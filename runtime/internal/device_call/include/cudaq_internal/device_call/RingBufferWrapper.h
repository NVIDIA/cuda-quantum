/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace cudaq_internal::device_call {

// Helper to compute the required ring buffer slot size for a given request and
// response size.
std::uint64_t requiredRingSlotSize(std::uint64_t requestBytes,
                                   std::uint64_t responseCapacity);

DeviceCallStatus statusFromCudaqStatus(cudaq_status_t status);

// Helper to check the return status of dispatcher API calls
void checkDispatcherStatus(cudaq_status_t status, const char *call,
                           const char *file, int line, const char *function);

#define CUDAQ_CHECK_DISPATCHER_STATUS(expr)                                    \
  ::cudaq_internal::device_call::checkDispatcherStatus(                        \
      (expr), #expr, __FILE__, __LINE__, __func__)

class RingBufferWrapper {
public:
  // `FrameState` tracks one slot of the underlying `cudaq_ringbuffer_t` while a
  // host caller holds it between `claimSlot` and `releaseFrame`.
  struct FrameState {
    // Index of this slot in the ring buffer.
    std::uint32_t slot = 0;
    // Unique id written into the RX RPC header and echoed back by the
    // dispatcher kernel in `cudaq::realtime::RPCResponse::request_id`.
    // Used to confirm that the TX payload corresponds to this caller's request.
    std::uint32_t requestId = 0;
    // True between `claimSlot` and `releaseFrame`. While true, this slot is
    // owned by the caller. Cleared once the frame is released.
    bool inUse = false;
    // Tracks an in-flight fire-and-forget request (responseCapacity == 0)
    // whose dispatcher-side work may still be running. Set in `dispatchFrame`
    // right after the slot is signaled via `signalHostSlot`/`signalDeviceSlot`,
    // so `dispatchFrame` knows not to poll the TX flag for completion. Stays
    // set across `releaseFrame` to mark that the slot cannot be reused until
    // the dispatcher finishes. Cleared by `tryGetReadySlot` once
    // `clearSlot`/`clearDeviceSlot` has reclaimed the slot.
    bool fireAndForgetPending = false;
  };

  // Check function to determine if a slot is ready for reuse.
  using IsSlotReadyFn = std::function<bool(FrameState &)>;

  // Poll interval used between slot availability checks.
  static constexpr std::chrono::microseconds PollInterval{50};

  RingBufferWrapper() = default;
  RingBufferWrapper(const RingBufferWrapper &) = delete;
  RingBufferWrapper &operator=(const RingBufferWrapper &) = delete;
  ~RingBufferWrapper() { reset(); }

  void configure(const char *channelName, int deviceId,
                 const DeviceCallChannelConfig &config);

  void reset() noexcept;

  // Find the first ready slot without blocking. Returns nullopt if none is
  // ready. Must be called under the caller's lock and immediately followed by
  // `claimSlot` in the same lock scope to avoid a race with other callers.
  std::optional<std::uint32_t>
  tryGetReadySlot(const IsSlotReadyFn &isSlotReady);

  // Write the RPC request header into `slot` and bind `frame` to its payload
  // regions.
  FrameState &claimSlot(std::uint32_t slot, std::uint32_t functionId,
                        std::uint64_t requestBytes,
                        std::uint64_t responseCapacity,
                        DeviceCallChannel::DeviceCallFrame &frame,
                        std::uint64_t txClearBytes);

  // Poll the slot's TX flag until the dispatcher publishes a response, then
  // validate it and return the response payload size in bytes.
  //   - state:             frame state returned by `claimSlot`.
  //   - frame:             frame whose response.capacity bounds the validated
  //   payload.
  //   - captureCudaError:  when true, surface the dispatcher's cudaError_t
  //                        (host-graph path); otherwise ignore it.
  std::uint64_t
  waitForResponseAndValidate(const FrameState &state,
                             DeviceCallChannel::DeviceCallFrame &frame,
                             bool captureCudaError = false);

  void clearSlot(std::uint32_t slot);
  void clearDeviceSlot(std::uint32_t slot);
  void signalDeviceSlot(std::uint32_t slot);
  void signalHostSlot(std::uint32_t slot);

  bool configured() const { return slotSizeValue != 0; }

  cudaq_ringbuffer_t *ringbuffer() { return &ringbufferValue; }
  const cudaq_ringbuffer_t *ringbuffer() const { return &ringbufferValue; }

  std::uint32_t numSlots() const { return numSlotsValue; }
  std::uint64_t slotSize() const { return slotSizeValue; }
  std::uint64_t timeoutMs() const { return timeoutMsValue; }

private:
  // Simple RAII wrapper for a pinned host allocation.
  class PinnedAllocation {
  public:
    PinnedAllocation() = default;
    PinnedAllocation(const PinnedAllocation &) = delete;
    PinnedAllocation &operator=(const PinnedAllocation &) = delete;
    ~PinnedAllocation() { reset(); }

    bool allocate(std::size_t bytes);
    void reset() noexcept;

    void *hostPtr() const { return host; }
    void *devicePtr() const { return device; }

  private:
    void *host = nullptr;
    void *device = nullptr;
  };

  bool allocateStorage(const char *channelName, std::uint32_t slots,
                       std::uint64_t bytesPerSlot);

  std::uint8_t *rxHostSlot(std::uint32_t slot) const;
  std::uint8_t *txHostSlot(std::uint32_t slot) const;
  std::uint64_t rxDeviceSlotAddress(std::uint32_t slot) const;

  PinnedAllocation rxFlags;
  PinnedAllocation txFlags;
  PinnedAllocation rxData;
  PinnedAllocation txData;
  cudaq_ringbuffer_t ringbufferValue{};
  std::vector<FrameState> frameStates;
  std::uint32_t numSlotsValue = DefaultNumSlots;
  std::uint64_t slotSizeValue = 0;
  std::uint64_t timeoutMsValue = DefaultTimeoutMs;
  std::uint32_t nextSlot = 0;
  std::uint32_t nextRequestId = 1;
};

} // namespace cudaq_internal::device_call
