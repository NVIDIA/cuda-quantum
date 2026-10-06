/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// RingSlotChannel -- the transport-independent caller side of the v1
// ring-slot device_call RPC protocol, shared by CpuRoceChannel (RDMA wire)
// and UdpChannel (loopback/Ethernet UDP wire).  Everything protocol-shaped
// lives here so a fix or protocol evolution lands in exactly one place; the
// subclasses supply only transport setup/teardown and argument parsing.
//
// Protocol (v1): the service's host dispatcher mirrors its rx slot to its tx
// slot, and both ends fill RX slots in strict arrival order, so serializing
// response-bearing dispatch and assigning ring slots round-robin keeps the
// service's FIFO rx slot equal to ours.  request_id stays globally monotonic
// for uniqueness/validation.  The service sends a response for *every*
// request, including fire-and-forget; a fire-and-forget's (ignored)
// zero-length response is drained on the slot's next reuse so it is never
// read as a later request's response.
//
// Subclass contract: initialize() must fill the protected ring/config members
// (numSlots, slotSize, timeoutMs, tx/rx data + flags + strides) and call
// initRingState() before the first dispatch.  Flags are host-visible
// std::uint64_t ring entries accessed with acquire/release atomics; a
// non-zero value means "slot published/fresh", zero means "free".

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/RpcFrame.h"

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/runtime/logger/logger.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace cudaq_internal::device_call {

class RingSlotChannel : public DeviceCallChannel {
public:
  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    frame = {};
    if (CUDAQ_RPC_HEADER_SIZE + requestBytes > slotSize)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            std::string(channelTag()) +
                                " request exceeds slot size");
    if (sizeof(cudaq::realtime::RPCResponse) + responseCapacity > slotSize)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            std::string(channelTag()) +
                                " response capacity exceeds slot size");

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

    CUDAQ_DBG("[device-call] {} acquire functionId={} requestBytes={} "
              "responseCapacity={} requestId={}",
              channelTag(), functionId, requestBytes, responseCapacity,
              static_cast<FrameState *>(frame.channelPrivate)->requestId);
  }

  std::uint64_t dispatchFrame(DeviceCallFrame &frame) override {
    auto *state = static_cast<FrameState *>(frame.channelPrivate);
    if (!state || !state->inUse)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            std::string(channelTag()) +
                                " dispatch on invalid frame");

    // v1: serialize the wire round-trip so the service's FIFO rx slot tracks
    // the ring slot we pick here.  See file header.
    std::lock_guard<std::mutex> lock(dispatchMutex);

    const std::uint32_t slot = rrCounter % numSlots;
    rrCounter = (rrCounter + 1u) % numSlots;

    // If this slot still has an outstanding fire-and-forget whose response we
    // never read, drain that late (zero-length) response before reuse.
    if (ffPending[slot]) {
      if (waitFlagNonZero(rxFlags[slot]))
        clearFlag(rxFlags[slot]);
      else
        CUDAQ_DBG("[device-call] {} fire-and-forget drain timed out slot={} "
                  "requestId={}",
                  channelTag(), slot, ffPendingRequestId[slot]);
      ffPending[slot] = 0;
    }

    std::uint8_t *const txSlot = txData + slot * txStride;
    std::uint8_t *const rxSlot = rxData + slot * rxStride;

    // Back-pressure: the prior send in this slot must have been shipped by
    // the transport before we overwrite it.  A still-published flag past the
    // timeout means the transport TX path is wedged; overwriting anyway would
    // corrupt an in-flight message, so fail the dispatch instead.
    waitFlagZeroOrThrow(txFlags[slot], "tx");
    // Defensive: clear any stale response flag for this slot before reuse.
    clearFlag(rxFlags[slot]);

    // Zero the full transmitted range before writing.  The transport ships
    // the whole per-slot stride, so any bytes beyond [RPCHeader | args] would
    // otherwise carry stale ring contents from a previous message onto the
    // wire.  The service only reads arg_len, but we must not transmit
    // uninitialized/stale memory.
    std::memset(txSlot, 0, txStride);
    std::memcpy(txSlot, state->requestScratch.data(),
                state->requestScratch.size());
    publishFlag(txFlags[slot], reinterpret_cast<std::uint64_t>(txSlot));

    CUDAQ_DBG("[device-call] {} dispatch slot={} requestId={} functionId={} "
              "fireAndForget={}",
              channelTag(), slot, state->requestId, state->functionId,
              state->responseCapacity == 0);

    if (state->responseCapacity == 0) {
      // Async fire-and-forget: return immediately per the device_call
      // contract.  Mark the slot so its next reuse drains the service's late
      // zero-length response (see the drain above) before overwriting it.
      ffPending[slot] = 1;
      ffPendingRequestId[slot] = state->requestId;
      return 0;
    }

    // Wait for the service's response to land in our rx slot.
    if (!waitFlagNonZero(rxFlags[slot]))
      throw DeviceCallError(DeviceCallStatus::Timeout,
                            std::string(channelTag()) +
                                " timed out waiting for response");

    std::uint64_t resultLen = 0;
    try {
      resultLen = validateResponseFrame(rxSlot, state->requestId,
                                        state->responseCapacity, rxStride);
      // Hand the validated result bytes back through the caller's scratch.
      std::memcpy(state->responseScratch.data(), rxSlot,
                  sizeof(cudaq::realtime::RPCResponse) + resultLen);
    } catch (...) {
      clearFlag(rxFlags[slot]);
      throw;
    }

    // Release the rx slot so the transport's RX path can reuse it.
    clearFlag(rxFlags[slot]);

    CUDAQ_DBG("[device-call] {} dispatch complete slot={} requestId={} "
              "resultLen={}",
              channelTag(), slot, state->requestId, resultLen);
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

protected:
  // Per-lease state hung off DeviceCallFrame::channelPrivate.  Holds
  // host-visible scratch the caller writes args into / reads results out of;
  // dispatchFrame copies between this scratch and the ring slot it picks.
  struct FrameState {
    std::uint32_t functionId = 0;
    std::uint32_t requestId = 0;
    std::uint64_t requestBytes = 0;
    std::uint64_t responseCapacity = 0;
    bool inUse = false;
    std::vector<std::byte> requestScratch;  // [RPCHeader | args]
    std::vector<std::byte> responseScratch; // [RPCResponse | result]
  };

  // Short transport tag used in log and error messages ("cpu_roce", "udp").
  virtual const char *channelTag() const noexcept = 0;

  // Called between flag polls.  Transports pick their wait posture: busy-spin
  // (cpuRelax) for latency-sensitive RDMA, a short sleep for the UDP
  // transport.
  virtual void relax() const = 0;

  // Subclass initialize() fills these, then calls initRingState().
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint64_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
  std::uint8_t *txData = nullptr;
  std::uint64_t *txFlags = nullptr;
  std::size_t txStride = 0;
  std::uint8_t *rxData = nullptr;
  std::uint64_t *rxFlags = nullptr;
  std::size_t rxStride = 0;

  void initRingState() {
    ffPending.assign(numSlots, 0);
    ffPendingRequestId.assign(numSlots, 0);
  }

  // ---- flag handshakes (acquire/release atomics on ring flag entries) ----

  static std::uint64_t loadFlag(std::uint64_t &flag) {
    return __atomic_load_n(&flag, __ATOMIC_ACQUIRE);
  }
  static void publishFlag(std::uint64_t &flag, std::uint64_t value) {
    __atomic_store_n(&flag, value, __ATOMIC_RELEASE);
  }
  static void clearFlag(std::uint64_t &flag) { publishFlag(flag, 0); }

  void waitFlagZeroOrThrow(std::uint64_t &flag, const char *which) const {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (loadFlag(flag) != 0) {
      if (std::chrono::steady_clock::now() > deadline)
        throw DeviceCallError(DeviceCallStatus::Timeout,
                              std::string(channelTag()) + " " + which +
                                  " slot back-pressure timed out");
      relax();
    }
  }

  bool waitFlagNonZero(std::uint64_t &flag) const {
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    while (loadFlag(flag) == 0) {
      if (std::chrono::steady_clock::now() > deadline)
        return false;
      relax();
    }
    return true;
  }

  // ---- argument parsing helpers ----

  // Invoke fn(key, value) for every "key=value" token; other tokens are
  // ignored (they may belong to other components on the command line).
  template <typename Fn>
  static void forEachKeyValue(const std::vector<std::string> &arguments,
                              Fn &&fn) {
    for (const auto &arg : arguments) {
      const auto eq = arg.find('=');
      if (eq == std::string::npos)
        continue;
      fn(arg.substr(0, eq), arg.substr(eq + 1));
    }
  }

  // Parse a UDP/TCP port argument, rejecting non-numeric input and values
  // outside [1, 65535] with a DeviceCallError instead of letting std::stoul's
  // raw exceptions (or a silent uint16_t truncation) escape.
  std::uint16_t parsePort(const std::string &value, const char *key) const {
    unsigned long port = 0;
    std::size_t consumed = 0;
    try {
      port = std::stoul(value, &consumed);
    } catch (const std::exception &) {
      consumed = 0;
    }
    if (consumed != value.size() || value.empty() || port == 0 || port > 65535)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            std::string(channelTag()) + " channel: " + key +
                                "=" + value + " is not a valid port (1-65535)");
    return static_cast<std::uint16_t>(port);
  }

private:
  std::mutex dispatchMutex; // v1: serialize wire round-trips
  std::uint32_t rrCounter = 0;
  // Per-slot tracking of an outstanding fire-and-forget whose (ignored)
  // zero-length response has not yet been drained.  Sized in initRingState().
  std::vector<char> ffPending;
  std::vector<std::uint32_t> ffPendingRequestId;
  std::atomic<std::uint32_t> requestIdCounter{1};
};

} // namespace cudaq_internal::device_call
