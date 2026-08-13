/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// UdpChannel — a DeviceCallChannel that carries device_call RPCs over UDP
// datagrams using the cpu_transport UdpTransceiver.  It is the plain-UDP
// counterpart of CpuRoceChannel for systems without an RDMA NIC — anything
// from loopback development to a real UDP network: same wire frames
// ([RPCHeader | args] request, [RPCResponse | result] response), same ring
// contract, and the same caller-side frame lifecycle (RingSlotChannel), so
// the service end (a daemon wiring the transceiver rings into
// libcudaq-realtime's HOST_CALL dispatcher) is transport-agnostic between UDP
// and RoCE.
//
// No QP/rkey rendezvous is needed (UDP addressing replaces it): the channel
// just connects to the daemon's UDP endpoint.
//
// Slot correlation is the shared v1 protocol (see RingSlotChannel.h).  Known
// v1 limitation on this transport: correlation is purely positional, and UDP
// -- unlike the RoCE RC wire -- gives no delivery guarantee, so a datagram
// that is lost (e.g. socket-buffer overflow under a large-numSlots burst) or
// delayed past timeoutMs (e.g. a fire-and-forget handler stalling longer
// than the drain timeout) permanently shifts the arrival cursor against the
// round-robin slot counter; every later response-bearing dispatch then fails
// validation or times out until the channel is torn down and re-created.
// Default configurations cannot reach this state (in-flight traffic is
// capped at numSlots datagrams, far below any realistic socket buffer); the
// phase-2 request_id-keyed correlation removes the fragility outright.
//
// Channel arguments (key=value tokens on the device_call command line):
//   udp-host=<ipv4>   service daemon address (required)
//   udp-port=<port>   service daemon UDP port (required)
//
// Both ends must use the same slot stride: each datagram carries one full
// slot (see udp_wrapper.h, "Wire behavior"), so the daemon's --slot-size must
// equal this channel's configured slot size or the receiving end silently
// drops every request.

#include "cudaq_internal/device_call/RingSlotChannel.h"

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"

#include <chrono>
#include <string>
#include <thread>

namespace {

using namespace cudaq_internal::device_call;

class UdpChannel : public RingSlotChannel {
public:
  ~UdpChannel() override { stop(); }

  void initialize(DeviceCallChannelCreateArgs &&args) override {
    numSlots = args.channelConfig.numSlots;
    slotSize = args.channelConfig.slotSize;
    timeoutMs = args.channelConfig.timeoutMs;
    parseArguments(args.arguments);
    CUDAQ_INFO("[device-call] udp channel initialize host={} port={} "
               "slots={} slotSize={} timeoutMs={}",
               serviceHost, servicePort, numSlots, slotSize, timeoutMs);
    if (serviceHost.empty() || servicePort == 0)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "udp channel requires udp-host=... udp-port=...");

    xcvr = cpu_udp_create_transceiver(slotSize, numSlots);
    if (!xcvr)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "udp channel: transceiver create failed");
    if (!cpu_udp_connect(xcvr, serviceHost.c_str(), servicePort)) {
      stop();
      throw DeviceCallError(DeviceCallStatus::RemoteError,
                            "udp channel: transceiver connect failed");
    }
    if (!cpu_udp_start(xcvr)) {
      stop();
      throw DeviceCallError(DeviceCallStatus::RemoteError,
                            "udp channel: transceiver start failed");
    }

    rxFlags =
        reinterpret_cast<std::uint64_t *>(cpu_udp_get_rx_ring_flag_addr(xcvr));
    txFlags =
        reinterpret_cast<std::uint64_t *>(cpu_udp_get_tx_ring_flag_addr(xcvr));
    rxData =
        reinterpret_cast<std::uint8_t *>(cpu_udp_get_rx_ring_data_addr(xcvr));
    txData =
        reinterpret_cast<std::uint8_t *>(cpu_udp_get_tx_ring_data_addr(xcvr));
    rxStride = slotSize;
    txStride = slotSize;
    initRingState();
  }

  void stop() noexcept override {
    if (xcvr) {
      cpu_udp_close(xcvr);
      cpu_udp_destroy_transceiver(xcvr);
      xcvr = nullptr;
    }
  }

private:
  const char *channelTag() const noexcept override { return "udp"; }

  // A 5us sleep keeps the per-poll latency floor small without pinning a
  // core the way the RDMA channel's cpuRelax spin does.
  void relax() const override {
    std::this_thread::sleep_for(std::chrono::microseconds(5));
  }

  void parseArguments(const std::vector<std::string> &arguments) {
    forEachKeyValue(arguments,
                    [this](const std::string &key, const std::string &value) {
                      if (key == "udp-host")
                        serviceHost = value;
                      else if (key == "udp-port")
                        servicePort = parsePort(value, "udp-port");
                    });
  }

  std::string serviceHost;
  std::uint16_t servicePort = 0;
  cpu_udp_transceiver_t xcvr = nullptr;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, UdpChannel, udp)

} // namespace
