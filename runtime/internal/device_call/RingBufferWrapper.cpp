/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/RingBufferWrapper.h"
#include "cudaq_internal/device_call/RpcFrame.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/runtime/logger/logger.h"

#include <cuda_runtime.h>

#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

namespace cudaq_internal::device_call {

constexpr std::uint32_t InitialSlot = 0;
constexpr std::uint32_t InitialRequestId = 1;
constexpr std::uint64_t EmptyRingFlag = 0;
constexpr std::uint64_t NoPtpTimestamp = 0;
constexpr std::uint64_t RingSlotAlignmentBytes = 256;
constexpr std::chrono::microseconds PollInterval{50};
//==============================================================================
// Ring buffer slot helpers
//==============================================================================
static inline void storeFlag(volatile std::uint64_t *flag,
                             std::uint64_t value) {

  auto *atomicFlag = reinterpret_cast<std::atomic<std::uint64_t> *>(
      const_cast<std::uint64_t *>(flag));
  atomicFlag->store(value, std::memory_order_release);
}

std::uint64_t requiredRingSlotSize(std::uint64_t requestBytes,
                                   std::uint64_t responseCapacity) {
  assert((requestBytes <=
          std::numeric_limits<std::uint32_t>::max() - CUDAQ_RPC_HEADER_SIZE) &&
         (responseCapacity <= std::numeric_limits<std::uint32_t>::max() -
                                  sizeof(cudaq::realtime::RPCResponse)) &&
         "device_call frame length exceeds 32 bits (overflow)");

  return std::max<std::uint64_t>(CUDAQ_RPC_HEADER_SIZE + requestBytes,
                                 sizeof(cudaq::realtime::RPCResponse) +
                                     responseCapacity);
}

DeviceCallStatus statusFromCudaqStatus(cudaq_status_t status) {
  switch (status) {
  case CUDAQ_OK:
    return DeviceCallStatus::Success;
  case CUDAQ_ERR_INVALID_ARG:
    return DeviceCallStatus::InvalidArgument;
  case CUDAQ_ERR_CUDA:
    return DeviceCallStatus::CudaError;
  case CUDAQ_ERR_INTERNAL:
  default:
    return DeviceCallStatus::RemoteError;
  }
}

void checkDispatcherStatus(cudaq_status_t status, const char *call,
                           const char *file, int line, const char *function) {
  if (status == CUDAQ_OK)
    return;
  CUDAQ_INFO("[device-call] dispatcher status failure status={} call={} "
             "location={}:{} function={}",
             static_cast<int>(status), call, file, line, function);
  const std::string message =
      std::string("device_call dispatcher call failed: ") + call + " at " +
      file + ":" + std::to_string(line);
  throw DeviceCallError(statusFromCudaqStatus(status), message);
}

bool RingBufferWrapper::PinnedAllocation::allocate(std::size_t bytes) {
  reset();

  void *hostPtr = nullptr;
  cudaError_t err = cudaHostAlloc(&hostPtr, bytes, cudaHostAllocMapped);
  if (err != cudaSuccess)
    return false;

  void *devicePtr = nullptr;
  err = cudaHostGetDevicePointer(&devicePtr, hostPtr, /*flags=*/0);
  if (err != cudaSuccess) {
    cudaFreeHost(hostPtr);
    return false;
  }

  std::memset(hostPtr, 0, bytes);
  host = hostPtr;
  device = devicePtr;
  return true;
}

void RingBufferWrapper::PinnedAllocation::reset() noexcept {
  if (host)
    cudaFreeHost(host);
  host = nullptr;
  device = nullptr;
}

void RingBufferWrapper::configure(const char *channelName, int deviceId,
                                  const DeviceCallChannelConfig &config) {
  CUDAQ_DBG("[device-call] {} channel configure ring configuredSlots={} "
            "configuredSlotSize={} timeoutMs={}",
            channelName, config.numSlots, config.slotSize, config.timeoutMs);
  reset();

  numSlotsValue = static_cast<std::uint32_t>(
      std::max<std::uint64_t>(DefaultNumSlots, config.numSlots));
  slotSizeValue = llvm::alignTo(config.slotSize, RingSlotAlignmentBytes);
  timeoutMsValue = config.timeoutMs;

  if (cudaSetDevice(deviceId) != cudaSuccess)
    throw DeviceCallError(DeviceCallStatus::CudaError,
                          "failed to set CUDA device for device_call");

  if (!allocateStorage(channelName, numSlotsValue, slotSizeValue))
    throw DeviceCallError(DeviceCallStatus::CudaError,
                          "failed to allocate device_call ring storage");

  nextSlot = InitialSlot;
  nextRequestId = InitialRequestId;
  frameStates.assign(numSlotsValue, FrameState{});

  CUDAQ_DBG("[device-call] {} channel ring configured device={} slots={} "
            "slotSize={} timeoutMs={}",
            channelName, deviceId, numSlotsValue, slotSizeValue,
            timeoutMsValue);
}

void RingBufferWrapper::reset() noexcept {
  rxFlags.reset();
  txFlags.reset();
  rxData.reset();
  txData.reset();
  ringbufferValue = {};
  frameStates.clear();
  numSlotsValue = DefaultNumSlots;
  slotSizeValue = 0;
  timeoutMsValue = DefaultTimeoutMs;
  nextSlot = InitialSlot;
  nextRequestId = InitialRequestId;
}

std::optional<std::uint32_t>
RingBufferWrapper::tryGetReadySlot(const IsSlotReadyFn &isSlotReady) {
  for (std::uint32_t attempt = 0; attempt < numSlotsValue; ++attempt) {
    const std::uint32_t candidate = (nextSlot + attempt) % numSlotsValue;
    auto &state = frameStates[candidate];
    if (state.inUse)
      continue;
    state.slot = candidate;
    if (isSlotReady(state)) {
      nextSlot = (candidate + 1) % numSlotsValue;
      return candidate;
    }
  }
  return std::nullopt;
}

RingBufferWrapper::FrameState &RingBufferWrapper::claimSlot(
    std::uint32_t slot, std::uint32_t functionId, std::uint64_t requestBytes,
    std::uint64_t responseCapacity, DeviceCallChannel::DeviceCallFrame &frame,
    std::uint64_t txClearBytes) {
  const std::uint32_t requestId = nextRequestId++;
  std::uint8_t *const rxSlot = rxHostSlot(slot);
  std::uint8_t *const txSlot = txHostSlot(slot);
  std::memset(rxSlot, 0, slotSizeValue);
  std::memset(txSlot, 0, std::min(txClearBytes, slotSizeValue));
  CUDAQ_CHECK_DISPATCHER_STATUS(cudaq_host_ringbuffer_write_rpc_request(
      &ringbufferValue, slot, functionId, nullptr,
      static_cast<std::uint32_t>(requestBytes), requestId, NoPtpTimestamp));

  auto &state = frameStates[slot];
  state.slot = slot;
  state.requestId = requestId;
  state.inUse = true;
  state.fireAndForgetPending = false;

  frame.functionId = functionId;
  frame.request.data = requestPayload(rxSlot);
  frame.request.capacity = requestBytes;
  frame.response.data = responsePayload(txSlot);
  frame.response.capacity = responseCapacity;
  frame.channelPrivate = &state;
  return state;
}

std::uint64_t RingBufferWrapper::waitForResponseAndValidate(
    const FrameState &state, DeviceCallChannel::DeviceCallFrame &frame,
    bool captureCudaError) {
  constexpr const char *txErrorMessage =
      "device_call dispatch reported an error";
  constexpr const char *timeoutMessage =
      "timed out waiting for device_call response";
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(timeoutMsValue);
  while (true) {
    int cudaError = 0;
    int *const cudaErrorPtr = captureCudaError ? &cudaError : nullptr;
    switch (cudaq_host_ringbuffer_poll_tx_flag(&ringbufferValue, state.slot,
                                               cudaErrorPtr)) {
    case CUDAQ_TX_EMPTY:
    case CUDAQ_TX_IN_FLIGHT:
      break;
    case CUDAQ_TX_READY:
      return validateResponseFrame(txHostSlot(state.slot), state.requestId,
                                   frame.response.capacity, slotSizeValue);
    case CUDAQ_TX_ERROR:
      throw DeviceCallError(DeviceCallStatus::CudaError, txErrorMessage);
    }

    if (std::chrono::steady_clock::now() > deadline)
      throw DeviceCallError(DeviceCallStatus::Timeout, timeoutMessage);
    std::this_thread::sleep_for(PollInterval);
  }
}

void RingBufferWrapper::clearSlot(std::uint32_t slot) {
  cudaq_host_ringbuffer_clear_slot(&ringbufferValue, slot);
}

void RingBufferWrapper::clearDeviceSlot(std::uint32_t slot) {
  clearSlot(slot);
  storeFlag(ringbufferValue.rx_flags_host + slot, EmptyRingFlag);
}

void RingBufferWrapper::signalDeviceSlot(std::uint32_t slot) {
  storeFlag(ringbufferValue.rx_flags_host + slot, rxDeviceSlotAddress(slot));
}

void RingBufferWrapper::signalHostSlot(std::uint32_t slot) {
  cudaq_host_ringbuffer_signal_slot(&ringbufferValue, slot);
}

bool RingBufferWrapper::allocateStorage(const char *channelName,
                                        std::uint32_t slots,
                                        std::uint64_t bytesPerSlot) {
  CUDAQ_DBG("[device-call] {} ring storage allocate slots={} "
            "bytesPerSlot={}",
            channelName, slots, bytesPerSlot);
  const auto flagsBytes =
      static_cast<std::size_t>(slots) * sizeof(std::uint64_t);
  const auto dataBytes =
      static_cast<std::size_t>(slots) * static_cast<std::size_t>(bytesPerSlot);
  if (!rxFlags.allocate(flagsBytes) || !txFlags.allocate(flagsBytes) ||
      !rxData.allocate(dataBytes) || !txData.allocate(dataBytes)) {
    CUDAQ_INFO("[device-call] {} ring storage allocation failed", channelName);
    reset();
    return false;
  }

  ringbufferValue.rx_flags =
      static_cast<volatile std::uint64_t *>(rxFlags.devicePtr());
  ringbufferValue.tx_flags =
      static_cast<volatile std::uint64_t *>(txFlags.devicePtr());
  ringbufferValue.rx_data = static_cast<std::uint8_t *>(rxData.devicePtr());
  ringbufferValue.tx_data = static_cast<std::uint8_t *>(txData.devicePtr());
  ringbufferValue.rx_stride_sz = bytesPerSlot;
  ringbufferValue.tx_stride_sz = bytesPerSlot;
  ringbufferValue.rx_flags_host =
      static_cast<volatile std::uint64_t *>(rxFlags.hostPtr());
  ringbufferValue.tx_flags_host =
      static_cast<volatile std::uint64_t *>(txFlags.hostPtr());
  ringbufferValue.rx_data_host = static_cast<std::uint8_t *>(rxData.hostPtr());
  ringbufferValue.tx_data_host = static_cast<std::uint8_t *>(txData.hostPtr());
  return true;
}

std::uint8_t *RingBufferWrapper::rxHostSlot(std::uint32_t slot) const {
  return ringbufferValue.rx_data_host + slot * slotSizeValue;
}

std::uint8_t *RingBufferWrapper::txHostSlot(std::uint32_t slot) const {
  return ringbufferValue.tx_data_host + slot * slotSizeValue;
}

std::uint64_t RingBufferWrapper::rxDeviceSlotAddress(std::uint32_t slot) const {
  return reinterpret_cast<std::uint64_t>(ringbufferValue.rx_data +
                                         slot * slotSizeValue);
}

} // namespace cudaq_internal::device_call
