/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// CpuRoceChannel — a DeviceCallChannel that carries device_call RPCs over a
// pure-CPU RoCEv2 RDMA wire using the Phase 1 CpuRoceTransceiver.  It is the
// caller (device_call origin) end of the wire; the service end is a separate
// process (the test daemon, or eventually a real HSB-enabled FPGA).
//
// Wire pattern (asymmetric, FPGA-compatible):
//   - caller -> service : IBV_WR_SEND  (the service, e.g. an FPGA, can only
//                         *receive* Sends).  Our transceiver TX therefore uses
//                         tx_mode=kSendForFpga.
//   - service -> caller : IBV_WR_RDMA_WRITE_WITH_IMM into our rx ring (the
//                         service, e.g. an FPGA, *transmits* Writes).  Our
//                         transceiver RX consumes those.
//
// QP/rkey rendezvous: connected (UC) QPs require each end to know the other's
// QP number before any traffic flows, so a one-way "daemon prints, caller
// reads" handshake is insufficient.  We do a minimal bidirectional TCP swap of
// {qp, rkey, roce-ipv4} inside initialize(), between the transceiver's setup()
// (mints our QP/rkey) and connect() (needs the peer's QP/rkey).  No HSB / no
// Hololink dependency.  For a real FPGA service the same setup()/connect() seam
// is driven instead by the HSB control plane (authenticate/configure_roce);
// only this rendezvous step changes, not the data-plane wire.
//
// Slot correlation (v1): the daemon's host dispatcher mirrors its rx slot to
// its tx slot, and our transceiver's RX currently decodes the slot from the
// recv-WQE wr_id (FIFO order) rather than the Write-With-Imm imm_data.  Both
// are only correct for *in-order* traffic.  To keep this correct without
// changing the (already-merged) host dispatcher or the transceiver,
// dispatchFrame serializes response-bearing dispatch and assigns the ring slot
// at dispatch time in round-robin order, so the daemon's FIFO rx slot always
// equals our slot.  request_id stays globally monotonic for
// uniqueness/validation.  True concurrent in-flight dispatch (imm-decode RX +
// daemon tx-slot=request_id) is a documented follow-up; see
// phase2_imm_convention.

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/RpcFrame.h"

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/runtime/logger/logger.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace cudaq_internal::device_call;

// Wire-format of one direction of the QP/rkey rendezvous.  All fields are in
// network byte order on the socket.  rx_base is omitted because our rx_data MR
// is registered with iova=0, so the peer addresses slots by offset alone.
struct RendezvousInfo {
  std::uint32_t qp_number = 0;
  std::uint32_t rkey = 0;
  std::uint32_t roce_ipv4 = 0; // network-order IPv4 of this end's RoCE GID
};

// Per-lease state hung off DeviceCallFrame::channelPrivate.  Holds host-visible
// scratch the caller writes args into / reads results out of; dispatchFrame
// copies between this scratch and the RDMA-registered ring slot it picks.
struct FrameState {
  std::uint32_t functionId = 0;
  std::uint32_t requestId = 0;
  std::uint64_t requestBytes = 0;
  std::uint64_t responseCapacity = 0;
  bool inUse = false;
  std::vector<std::byte> requestScratch;  // [RPCHeader | args]
  std::vector<std::byte> responseScratch; // [RPCResponse | result]
};

class CpuRoceChannel : public DeviceCallChannel {
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
    // tx_mode=SEND_FOR_FPGA: we Send requests; we receive the service's
    // Write-With-Imm responses.  The peer QP is not known until the rendezvous,
    // so we pass 0 here and supply the real value to cpu_roce_connect()
    // (cpu_roce_start() is not used in the setup()/connect() flow).
    xcvr = cpu_roce_create_transceiver(
        ibDevice.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/0u, frameSize, slotSize,
        numSlots, /*peer_ip=*/"0.0.0.0", /*forward=*/0, /*rx_only=*/0,
        /*tx_only=*/0, /*unified=*/0, CPU_ROCE_TX_MODE_SEND_FOR_FPGA,
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

    // connect(): INIT -> RTR -> RTS using the peer's QP + IP.  We Send (don't
    // Write) so we do not need the peer's rkey here.
    if (!cpu_roce_connect(xcvr, peer.qp_number, peerIpStr, /*peer_rx_rkey=*/0))
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

    // Run the transceiver RX/TX loops on a background thread.
    monitorThread = std::thread([this] { cpu_roce_blocking_monitor(xcvr); });
    started = true;
  }

  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    frame = {};
    if (CUDAQ_RPC_HEADER_SIZE + requestBytes > slotSize)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "cpu_roce request exceeds slot size");
    if (sizeof(cudaq::realtime::RPCResponse) + responseCapacity > slotSize)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "cpu_roce response capacity exceeds slot size");

    auto state = std::make_unique<FrameState>();
    state->functionId = functionId;
    state->requestBytes = requestBytes;
    state->responseCapacity = responseCapacity;
    state->requestId = requestIdCounter.fetch_add(1, std::memory_order_relaxed);
    state->requestScratch.assign(CUDAQ_RPC_HEADER_SIZE + requestBytes,
                                 std::byte{0});
    state->responseScratch.assign(
        sizeof(cudaq::realtime::RPCResponse) + responseCapacity, std::byte{0});
    state->inUse = true;

    // Lay down the request header in the scratch; args follow it.
    auto *hdr = reinterpret_cast<cudaq::realtime::RPCHeader *>(
        state->requestScratch.data());
    hdr->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    hdr->function_id = functionId;
    hdr->arg_len = static_cast<std::uint32_t>(requestBytes);
    hdr->request_id = state->requestId;
    hdr->ptp_timestamp = 0;

    frame.functionId = functionId;
    frame.request.data = requestPayload(state->requestScratch.data());
    frame.request.capacity = requestBytes;
    frame.response.data = responsePayload(state->responseScratch.data());
    frame.response.capacity = responseCapacity;
    frame.channelPrivate = state.release();

    CUDAQ_DBG("[device-call] cpu_roce acquire functionId={} requestBytes={} "
              "responseCapacity={} requestId={}",
              functionId, requestBytes, responseCapacity,
              reinterpret_cast<FrameState *>(frame.channelPrivate)->requestId);
  }

  std::uint64_t dispatchFrame(DeviceCallFrame &frame) override {
    auto *state = static_cast<FrameState *>(frame.channelPrivate);
    if (!state || !state->inUse)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "cpu_roce dispatch on invalid frame");

    // v1: serialize the wire round-trip so the daemon's FIFO rx slot tracks the
    // ring slot we pick here.  See file header.
    std::lock_guard<std::mutex> lock(dispatchMutex);

    const std::uint32_t slot = rrCounter % numSlots;
    rrCounter = (rrCounter + 1u) % numSlots;

    const std::uint64_t txAddr =
        reinterpret_cast<std::uint64_t>(txData) + slot * txStride;
    const std::uint64_t rxAddr =
        reinterpret_cast<std::uint64_t>(rxData) + slot * rxStride;

    // Back-pressure: wait for the TX thread to have drained any prior send in
    // this slot before we overwrite it.
    waitFlagZero(txFlags[slot], "tx");
    // Defensive: clear any stale response flag for this slot before reuse.
    __atomic_store_n(&rxFlags[slot], std::uint64_t{0}, __ATOMIC_RELEASE);

    // Zero the full transmitted range before writing.  The transceiver's TX
    // SGE length is the whole slot stride, so any bytes beyond [RPCHeader |
    // args] would otherwise carry stale ring contents from a previous message
    // onto the wire.  The daemon only reads arg_len, but we must not transmit
    // uninitialized/stale memory.
    std::memset(reinterpret_cast<void *>(txAddr), 0, txStride);
    // Copy [RPCHeader | args] into the TX ring slot and publish it.
    std::memcpy(reinterpret_cast<void *>(txAddr), state->requestScratch.data(),
                state->requestScratch.size());
    __atomic_store_n(&txFlags[slot], txAddr, __ATOMIC_RELEASE);

    CUDAQ_DBG("[device-call] cpu_roce dispatch slot={} requestId={} "
              "functionId={} fireAndForget={}",
              slot, state->requestId, state->functionId,
              state->responseCapacity == 0);

    if (state->responseCapacity == 0)
      return 0; // fire-and-forget

    // Wait for the service's Write-With-Imm response to land in our rx slot.
    if (!waitFlagNonZero(rxFlags[slot], "rx"))
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "cpu_roce timed out waiting for response");

    void *respFrame = reinterpret_cast<void *>(rxAddr);
    std::uint64_t resultLen = 0;
    try {
      resultLen = validateResponseFrame(respFrame, state->requestId,
                                        state->responseCapacity, rxStride);
      // Hand the validated result bytes back through the caller's scratch view.
      std::memcpy(state->responseScratch.data(), respFrame,
                  sizeof(cudaq::realtime::RPCResponse) + resultLen);
    } catch (...) {
      __atomic_store_n(&rxFlags[slot], std::uint64_t{0}, __ATOMIC_RELEASE);
      throw;
    }

    // Release the rx slot so the RX thread can re-arm its recv WQE.
    __atomic_store_n(&rxFlags[slot], std::uint64_t{0}, __ATOMIC_RELEASE);

    CUDAQ_DBG("[device-call] cpu_roce dispatch complete slot={} requestId={} "
              "resultLen={}",
              slot, state->requestId, resultLen);
    return resultLen;
  }

  void releaseFrame(DeviceCallFrame &frame) noexcept override {
    auto *state = static_cast<FrameState *>(frame.channelPrivate);
    if (state) {
      state->inUse = false;
      delete state;
    }
    frame = {};
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
  // ---- argument parsing -----------------------------------------------------

  void parseArguments(const std::vector<std::string> &arguments) {
    for (const auto &arg : arguments) {
      const auto eq = arg.find('=');
      if (eq == std::string::npos)
        continue;
      const std::string key = arg.substr(0, eq);
      const std::string value = arg.substr(eq + 1);
      if (key == "ib-device")
        ibDevice = value;
      else if (key == "local-ip")
        localIp = value;
      else if (key == "rendezvous-host")
        rendezvousHost = value;
      else if (key == "rendezvous-port")
        rendezvousPort = static_cast<std::uint16_t>(std::stoul(value));
    }
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

  // ---- flag handshakes ------------------------------------------------------

  void waitFlagZero(std::uint64_t &flag, const char *which) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (__atomic_load_n(&flag, __ATOMIC_ACQUIRE) != 0) {
      if (std::chrono::steady_clock::now() > deadline)
        throw DeviceCallError(DeviceCallStatus::Timeout,
                              std::string("cpu_roce ") + which +
                                  " slot back-pressure timed out");
      cpuRelax();
    }
  }

  bool waitFlagNonZero(std::uint64_t &flag, const char *) {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (__atomic_load_n(&flag, __ATOMIC_ACQUIRE) == 0) {
      if (std::chrono::steady_clock::now() > deadline)
        return false;
      cpuRelax();
    }
    return true;
  }

  static inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
  }

  // ---- configuration / state ------------------------------------------------

  DeviceCallChannelConfig channelConfig;
  std::uint32_t numSlots = 0;
  std::uint64_t slotSize = 0;
  std::uint64_t timeoutMs = DefaultTimeoutMs;

  std::string ibDevice;
  std::string localIp;
  std::string rendezvousHost;
  std::uint16_t rendezvousPort = 0;

  cpu_roce_transceiver_t xcvr = nullptr;
  std::thread monitorThread;
  bool started = false;

  std::uint8_t *txData = nullptr;
  std::uint64_t *txFlags = nullptr;
  std::size_t txStride = 0;
  std::uint8_t *rxData = nullptr;
  std::uint64_t *rxFlags = nullptr;
  std::size_t rxStride = 0;

  std::mutex dispatchMutex; // v1: serialize wire round-trips
  std::uint32_t rrCounter = 0;
  std::atomic<std::uint32_t> requestIdCounter{1};
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, CpuRoceChannel, cpu_roce)

} // namespace
