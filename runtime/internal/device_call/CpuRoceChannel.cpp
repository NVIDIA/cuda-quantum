/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// CpuRoceChannel — a DeviceCallChannel that carries device_call RPCs over a
// pure-CPU RoCEv2 RDMA wire using the CpuRoceTransceiver.  It is the caller
// (device_call origin / requester) end of the wire and plays the FPGA role on
// the wire; the service end is a separate process (the test daemon, or
// eventually a real decoder/bridge).  A real HSB-enabled FPGA could stand in
// for this channel unchanged, since the channel speaks the FPGA's wire pattern.
//
// Wire pattern (asymmetric, mirrors a real FPGA <-> bridge):
//   - caller -> service : IBV_WR_RDMA_WRITE_WITH_IMM into the service's rx
//                         ring (the caller pushes requests the way an FPGA
//                         RDMA-Writes syndromes).  Our transceiver TX uses
//                         tx_mode=kRdmaWriteWithImm, so we need the service's
//                         rx_data rkey (from the rendezvous) at connect().
//   - service -> caller : IBV_WR_SEND into our pre-posted recv WQEs (the
//                         service Sends responses the way a bridge Sends
//                         corrections to an FPGA, which receives Sends only).
//                         Our transceiver RX consumes those.
//
// QP/rkey rendezvous: connected (UC) QPs require each end to know the other's
// QP number before any traffic flows, so a one-way "daemon prints, caller
// reads" handshake is insufficient.  We do a minimal bidirectional TCP swap of
// {qp, rkey, roce-ipv4} inside initialize(), between the transceiver's setup()
// (mints our QP/rkey) and connect() (needs the peer's QP + rx_data rkey).  No
// HSB / no Hololink dependency.  For a real FPGA caller the same setup()/
// connect() seam is driven instead by the HSB control plane
// (authenticate/configure_roce); only this rendezvous step changes, not the
// data-plane wire.
//
// The frame lifecycle and v1 slot-correlation protocol (round-robin slot
// assignment, serialized response-bearing dispatch, fire-and-forget response
// draining) are the shared RingSlotChannel implementation; see
// RingSlotChannel.h.  True concurrent in-flight dispatch (imm-decode RX +
// daemon tx-slot=request_id) is a documented follow-up; see
// phase2_imm_convention.

#include "cudaq_internal/device_call/RingSlotChannel.h"

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

namespace {

using namespace cudaq_internal::device_call;

// Wire-format of one direction of the QP/rkey rendezvous.  All fields are in
// network byte order on the socket.  rx_base is omitted because each end's
// rx_data MR is registered with iova=0, so the writer addresses the peer's
// slots by offset alone (we Write requests using the peer's rkey below).
struct RendezvousInfo {
  std::uint32_t qp_number = 0;
  std::uint32_t rkey = 0;
  std::uint32_t roce_ipv4 = 0; // network-order IPv4 of this end's RoCE GID
};

class CpuRoceChannel : public RingSlotChannel {
public:
  ~CpuRoceChannel() override { stop(); }

  void initialize(DeviceCallChannelCreateArgs &&args) override {
    channelConfig = args.channelConfig;
    numSlots = channelConfig.numSlots;
    slotSize = channelConfig.slotSize;
    timeoutMs = channelConfig.timeoutMs;

    CUDAQ_INFO("[device-call] cpu_roce channel initialize slots={} slotSize={} "
               "timeoutMs={} args={}",
               numSlots, slotSize, timeoutMs, args.arguments.size());

    if (numSlots == 0 || slotSize < CUDAQ_RPC_HEADER_SIZE)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "cpu_roce channel needs numSlots>0 and "
                            "slotSize>=RPC header size");

    parseArguments(args.arguments);
    if (ibDevice.empty())
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "cpu_roce channel requires ib-device=...");
    if (rendezvousHost.empty() || rendezvousPort == 0)
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "cpu_roce channel requires rendezvous-host=... rendezvous-port=...");

    // The logical frame size carried by each slot is the RPC payload region;
    // the transceiver's per-slot stride is the full slotSize.
    const std::size_t frameSize = slotSize;

    // Construct the caller-end transceiver via the public C wrapper.
    // tx_mode=RDMA_WRITE_WITH_IMM: we Write requests into the service's rx ring
    // (like an FPGA pushing syndromes); we receive the service's Sends as
    // responses.  The peer QP and rx_data rkey are not known until the
    // rendezvous, so we pass 0 here and supply the real rkey to
    // cpu_roce_connect() (cpu_roce_start() is not used in the setup()/connect()
    // flow).  peer_rx_base_addr stays 0: the service registers its rx_data MR
    // with iova=0, so we address its slots by offset alone.
    xcvr = cpu_roce_create_transceiver(
        ibDevice.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/0u, frameSize, slotSize,
        numSlots, /*peer_ip=*/"0.0.0.0", /*forward=*/0, /*rx_only=*/0,
        /*tx_only=*/0, /*unified=*/0, CPU_ROCE_TX_MODE_RDMA_WRITE_WITH_IMM,
        /*peer_rx_base_addr=*/0, /*peer_rx_rkey=*/0);
    if (!xcvr)
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "cpu_roce channel: transceiver create failed");

    // Pin the source GID to our configured local address so we pick the right
    // SGID even on a multi-IP port.
    cpu_roce_set_local_ip(xcvr, localIp.c_str());

    if (!cpu_roce_setup(xcvr))
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "cpu_roce channel: transceiver setup() failed");

    // Bidirectional QP/rkey/IP swap with the service.
    RendezvousInfo self;
    self.qp_number = cpu_roce_get_qp_number(xcvr);
    self.rkey = cpu_roce_get_rkey(xcvr);
    self.roce_ipv4 = parseIpv4(localIp);
    const RendezvousInfo peer = exchangeRendezvous(self);

    char peerIpStr[INET_ADDRSTRLEN] = {0};
    in_addr peerAddr{};
    peerAddr.s_addr = peer.roce_ipv4;
    if (!inet_ntop(AF_INET, &peerAddr, peerIpStr, sizeof(peerIpStr)))
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "cpu_roce channel: bad peer RoCE IPv4 in rendezvous");

    CUDAQ_INFO("[device-call] cpu_roce rendezvous done: self{{qp={} rkey={}}} "
               "peer{{qp={} rkey={} ip={}}}",
               self.qp_number, self.rkey, peer.qp_number, peer.rkey, peerIpStr);

    // connect(): INIT -> RTR -> RTS using the peer's QP + IP.  We Write
    // requests into the service's rx ring, so we pass the service's rx_data
    // rkey (learned from the rendezvous) here.
    if (!cpu_roce_connect(xcvr, peer.qp_number, peerIpStr,
                          /*peer_rx_rkey=*/peer.rkey))
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "cpu_roce channel: transceiver connect() failed");

    // Cache ring pointers for the slot handshake.  RX and TX share the same
    // per-slot stride (page size).
    txData = static_cast<std::uint8_t *>(cpu_roce_get_tx_ring_data_addr(xcvr));
    txFlags = cpu_roce_get_tx_ring_flag_addr(xcvr);
    txStride = cpu_roce_get_page_size(xcvr);
    rxData = static_cast<std::uint8_t *>(cpu_roce_get_rx_ring_data_addr(xcvr));
    rxFlags = cpu_roce_get_rx_ring_flag_addr(xcvr);
    rxStride = cpu_roce_get_page_size(xcvr);

    initRingState();

    // Run the transceiver RX/TX loops on a background thread.
    monitorThread = std::thread([this] { cpu_roce_blocking_monitor(xcvr); });
    started = true;
  }

  void stop() noexcept override {
    if (xcvr)
      cpu_roce_close(xcvr);
    if (monitorThread.joinable())
      monitorThread.join();
    if (xcvr) {
      cpu_roce_destroy_transceiver(xcvr);
      xcvr = nullptr;
    }
    started = false;
  }

private:
  const char *channelTag() const noexcept override { return "cpu_roce"; }

  // Latency-sensitive RDMA wire: busy-spin between flag polls.
  void relax() const override {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
  }

  // ---- argument parsing -----------------------------------------------------

  void parseArguments(const std::vector<std::string> &arguments) {
    forEachKeyValue(arguments,
                    [this](const std::string &key, const std::string &value) {
                      if (key == "ib-device")
                        ibDevice = value;
                      else if (key == "local-ip")
                        localIp = value;
                      else if (key == "rendezvous-host")
                        rendezvousHost = value;
                      else if (key == "rendezvous-port")
                        rendezvousPort = parsePort(value, "rendezvous-port");
                    });
  }

  static std::uint32_t parseIpv4(const std::string &ip) {
    in_addr addr{};
    if (ip.empty() || inet_pton(AF_INET, ip.c_str(), &addr) != 1)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "cpu_roce channel requires a valid local-ip=...");
    return addr.s_addr; // network order
  }

  // ---- minimal TCP rendezvous (client) --------------------------------------

  RendezvousInfo exchangeRendezvous(const RendezvousInfo &self) {
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(rendezvousPort);
    if (inet_pton(AF_INET, rendezvousHost.c_str(), &addr.sin_addr) != 1)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "cpu_roce channel: bad rendezvous-host");

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    int fd = -1;
    for (;;) {
      fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0)
        throw DeviceCallError(DeviceCallStatus::NotInitialized,
                              "cpu_roce rendezvous: socket() failed");
      if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0)
        break;
      ::close(fd);
      fd = -1;
      if (std::chrono::steady_clock::now() > deadline)
        throw DeviceCallError(DeviceCallStatus::Timeout,
                              "cpu_roce rendezvous: could not reach service");
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

    // Client speaks first (request), then reads the service's reply.
    RendezvousInfo wire{htonl(self.qp_number), htonl(self.rkey),
                        self.roce_ipv4};
    if (!writeAll(fd, &wire, sizeof(wire)) ||
        !readAll(fd, &wire, sizeof(wire))) {
      ::close(fd);
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            "cpu_roce rendezvous: exchange failed");
    }
    ::close(fd);
    return RendezvousInfo{ntohl(wire.qp_number), ntohl(wire.rkey),
                          wire.roce_ipv4};
  }

  static bool writeAll(int fd, const void *buf, std::size_t len) {
    const auto *p = static_cast<const std::uint8_t *>(buf);
    while (len > 0) {
      const ssize_t n = ::write(fd, p, len);
      if (n <= 0) {
        if (n < 0 && errno == EINTR)
          continue;
        return false;
      }
      p += n;
      len -= static_cast<std::size_t>(n);
    }
    return true;
  }

  static bool readAll(int fd, void *buf, std::size_t len) {
    auto *p = static_cast<std::uint8_t *>(buf);
    while (len > 0) {
      const ssize_t n = ::read(fd, p, len);
      if (n <= 0) {
        if (n < 0 && errno == EINTR)
          continue;
        return false;
      }
      p += n;
      len -= static_cast<std::size_t>(n);
    }
    return true;
  }

  // ---- configuration / state ------------------------------------------------

  DeviceCallChannelConfig channelConfig;

  std::string ibDevice;
  std::string localIp;
  std::string rendezvousHost;
  std::uint16_t rendezvousPort = 0;

  cpu_roce_transceiver_t xcvr = nullptr;
  std::thread monitorThread;
  bool started = false;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, CpuRoceChannel, cpu_roce)

} // namespace
