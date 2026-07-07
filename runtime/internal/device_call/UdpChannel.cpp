/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// UdpChannel — a DeviceCallChannel that carries device_call RPCs over UDP
// datagrams using the cpu_transport UdpTransceiver.  It is the loopback/CI
// stand-in for CpuRoceChannel on hosts without an RDMA NIC: same wire frames
// ([RPCHeader | args] request, [RPCResponse | result] response), same ring
// contract, and the same caller-side frame lifecycle, so the service end (a
// daemon wiring the transceiver rings into libcudaq-realtime's HOST_CALL
// dispatcher) is transport-agnostic between UDP and RoCE.
//
// No QP/rkey rendezvous is needed (UDP addressing replaces it): the channel
// just connects to the daemon's UDP endpoint.
//
// Slot correlation (v1, matching CpuRoceChannel): the daemon's host dispatcher
// mirrors its rx slot to its tx slot, and both transceivers fill RX slots in
// strict arrival order, so serializing response-bearing dispatch and assigning
// ring slots round-robin keeps the daemon's FIFO rx slot equal to ours.
// request_id stays globally monotonic for uniqueness/validation.
//
// Channel arguments (key=value tokens on the device_call command line):
//   udp-host=<ipv4>   service daemon address (required)
//   udp-port=<port>   service daemon UDP port (required)
//
// Both ends must use the same slot stride (each datagram carries one full
// slot, mirroring the RoCE TX SGE); the daemon's --slot-size must equal this
// channel's configured slot size.

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/RpcFrame.h"

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/runtime/logger/logger.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace cudaq_internal::device_call;

class UdpChannel : public DeviceCallChannel {
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

    rxFlags = reinterpret_cast<volatile std::uint64_t *>(
        cpu_udp_get_rx_ring_flag_addr(xcvr));
    txFlags = reinterpret_cast<volatile std::uint64_t *>(
        cpu_udp_get_tx_ring_flag_addr(xcvr));
    rxData =
        reinterpret_cast<std::uint8_t *>(cpu_udp_get_rx_ring_data_addr(xcvr));
    txData =
        reinterpret_cast<std::uint8_t *>(cpu_udp_get_tx_ring_data_addr(xcvr));
    ffPending.assign(numSlots, 0);
  }

  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    frame = {};
    if (CUDAQ_RPC_HEADER_SIZE + requestBytes > slotSize)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "udp request exceeds slot size");
    if (sizeof(cudaq::realtime::RPCResponse) + responseCapacity > slotSize)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "udp response capacity exceeds slot size");

    auto state = std::make_unique<FrameState>();
    state->requestId = requestIdCounter.fetch_add(1, std::memory_order_relaxed);
    state->responseCapacity = responseCapacity;
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

    CUDAQ_DBG("[device-call] udp acquire functionId={} requestBytes={} "
              "responseCapacity={} requestId={}",
              functionId, requestBytes, responseCapacity,
              static_cast<FrameState *>(frame.channelPrivate)->requestId);
  }

  std::uint64_t dispatchFrame(DeviceCallFrame &frame) override {
    auto *state = static_cast<FrameState *>(frame.channelPrivate);
    if (!state || !state->inUse)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "udp dispatch on invalid frame");

    // v1: serialize the wire round-trip so the daemon's FIFO rx slot tracks
    // the ring slot we pick here (see file header).
    std::lock_guard<std::mutex> lock(dispatchMutex);

    const std::uint32_t slot = rrCounter % numSlots;
    rrCounter = (rrCounter + 1u) % numSlots;

    // If this slot still has an outstanding fire-and-forget whose response we
    // never read, drain that late (zero-length) response before reuse.
    if (ffPending[slot]) {
      if (waitFlagNonZero(&rxFlags[slot], "rx-drain"))
        clearFlag(&rxFlags[slot]);
      else
        CUDAQ_DBG("[device-call] udp fire-and-forget drain timed out slot={}",
                  slot);
      ffPending[slot] = 0;
    }

    // Back-pressure: wait for the TX thread to have shipped any prior send in
    // this slot before overwriting it, and clear any stale response flag.
    waitFlagZero(&txFlags[slot], "tx");
    clearFlag(&rxFlags[slot]);

    std::uint8_t *tx_slot = txData + slot * slotSize;
    // Zero the full transmitted range: the datagram carries the whole slot
    // stride (mirroring the RoCE TX SGE), so bytes beyond [RPCHeader | args]
    // must not leak stale ring contents onto the wire.
    std::memset(tx_slot, 0, slotSize);
    std::memcpy(tx_slot, state->requestScratch.data(),
                state->requestScratch.size());
    publishFlag(&txFlags[slot], reinterpret_cast<std::uint64_t>(tx_slot));

    CUDAQ_DBG("[device-call] udp dispatch slot={} requestId={} "
              "fireAndForget={}",
              slot, state->requestId, state->responseCapacity == 0);

    if (state->responseCapacity == 0) {
      // Async fire-and-forget: return immediately per the device_call
      // contract; the service still Sends a zero-length response, drained on
      // this slot's next reuse (see above).
      ffPending[slot] = 1;
      return 0;
    }

    // Wait for the service's response datagram to land in our rx slot.
    if (!waitFlagNonZero(&rxFlags[slot], "rx"))
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            "udp timed out waiting for response");

    void *respFrame = rxData + slot * slotSize;
    std::uint64_t resultLen = 0;
    try {
      resultLen = validateResponseFrame(respFrame, state->requestId,
                                        state->responseCapacity, slotSize);
      std::memcpy(state->responseScratch.data(), respFrame,
                  sizeof(cudaq::realtime::RPCResponse) + resultLen);
    } catch (...) {
      clearFlag(&rxFlags[slot]);
      throw;
    }

    // Release the rx slot so the transceiver's RX thread can reuse it.
    clearFlag(&rxFlags[slot]);

    CUDAQ_DBG("[device-call] udp dispatch complete slot={} requestId={} "
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
    if (xcvr) {
      cpu_udp_close(xcvr);
      cpu_udp_destroy_transceiver(xcvr);
      xcvr = nullptr;
    }
  }

private:
  struct FrameState {
    std::uint32_t requestId = 0;
    std::uint64_t responseCapacity = 0;
    std::vector<std::byte> requestScratch;
    std::vector<std::byte> responseScratch;
    bool inUse = false;
  };

  static std::uint64_t loadFlag(const volatile std::uint64_t *flag) {
    return __atomic_load_n(const_cast<const std::uint64_t *>(flag),
                           __ATOMIC_ACQUIRE);
  }
  static void publishFlag(volatile std::uint64_t *flag, std::uint64_t value) {
    __atomic_store_n(const_cast<std::uint64_t *>(flag), value,
                     __ATOMIC_RELEASE);
  }
  static void clearFlag(volatile std::uint64_t *flag) {
    publishFlag(flag, 0);
  }

  bool waitFlagNonZero(const volatile std::uint64_t *flag, const char *what) {
    return waitFlag(flag, /*wantZero=*/false, what);
  }
  bool waitFlagZero(const volatile std::uint64_t *flag, const char *what) {
    return waitFlag(flag, /*wantZero=*/true, what);
  }

  bool waitFlag(const volatile std::uint64_t *flag, bool wantZero,
                const char *what) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeoutMs);
    while ((loadFlag(flag) == 0) != wantZero) {
      if (std::chrono::steady_clock::now() > deadline) {
        CUDAQ_DBG("[device-call] udp wait timed out on {}", what);
        return false;
      }
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return true;
  }

  void parseArguments(const std::vector<std::string> &arguments) {
    for (const auto &arg : arguments) {
      const auto eq = arg.find('=');
      if (eq == std::string::npos)
        continue;
      const std::string key = arg.substr(0, eq);
      const std::string value = arg.substr(eq + 1);
      if (key == "udp-host")
        serviceHost = value;
      else if (key == "udp-port")
        servicePort = static_cast<std::uint16_t>(std::stoul(value));
    }
  }

  std::string serviceHost;
  std::uint16_t servicePort = 0;
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint64_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
  cpu_udp_transceiver_t xcvr = nullptr;
  volatile std::uint64_t *rxFlags = nullptr;
  volatile std::uint64_t *txFlags = nullptr;
  std::uint8_t *rxData = nullptr;
  std::uint8_t *txData = nullptr;
  std::atomic<std::uint32_t> requestIdCounter{1};
  std::mutex dispatchMutex;
  std::uint32_t rrCounter = 0;
  std::vector<std::uint8_t> ffPending;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, UdpChannel, udp)

} // namespace
