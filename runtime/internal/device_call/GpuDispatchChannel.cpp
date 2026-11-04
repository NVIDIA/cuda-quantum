/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/RingBufferWrapper.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/runtime/logger/logger.h"

#include <cuda_runtime.h>

#include <chrono>
#include <thread>
#include <tuple>

#include <mutex>
#include <stdexcept>

namespace {

using namespace cudaq_internal::device_call;

class GpuDispatchChannel : public DeviceCallChannel {
public:
  ~GpuDispatchChannel() override { stop(); }

  void initialize(DeviceCallChannelCreateArgs &&args) override {
    CUDAQ_INFO("[device-call] gpu channel initialize functionCount={} "
               "device={} slots={} slotSize={} timeoutMs={} launchFn={} "
               "synchronizeFn={}",
               args.functionCount, args.deviceId, args.channelConfig.numSlots,
               args.channelConfig.slotSize, args.channelConfig.timeoutMs,
               static_cast<bool>(args.launchFn),
               static_cast<bool>(args.synchronizeFn));
    if (!args.functionTable || args.functionCount == 0)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "device_call function table is empty");
    if (!args.launchFn)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "device_call dispatch launch hook is missing");
    functionTable = args.functionTable;
    functionCount = args.functionCount;
    deviceId = args.deviceId;
    channelConfig = args.channelConfig;
    launchFn = args.launchFn;
    synchronizeFn = args.synchronizeFn;
  }

  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    CUDAQ_DBG("[device-call] gpu channel acquire frame functionId={} "
              "requestBytes={} responseCapacity={}",
              functionId, requestBytes, responseCapacity);
    frame = {};
    const std::uint64_t requiredBytes =
        requiredRingSlotSize(requestBytes, responseCapacity);

    const auto isReady = [this](RingBufferWrapper::FrameState &state) -> bool {
      return isSlotReady(state);
    };

    RingBufferWrapper::FrameState *state = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex);
      ensureDispatcherStarted();
      if (requiredBytes > ringBuffer.slotSize())
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "device_call frame exceeds configured slot size");
      if (auto slot = ringBuffer.tryGetReadySlot(isReady))
        state = &ringBuffer.claimSlot(*slot, functionId, requestBytes,
                                      responseCapacity, frame,
                                      sizeof(cudaq::realtime::RPCResponse));
    }

    if (!state) {
      const auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::milliseconds(ringBuffer.timeoutMs());
      while (!state) {
        std::this_thread::sleep_for(RingBufferWrapper::PollInterval);
        {
          std::lock_guard<std::mutex> lock(mutex);
          if (auto slot = ringBuffer.tryGetReadySlot(isReady))
            state = &ringBuffer.claimSlot(*slot, functionId, requestBytes,
                                          responseCapacity, frame,
                                          sizeof(cudaq::realtime::RPCResponse));
        }
        if (std::chrono::steady_clock::now() > deadline)
          throw DeviceCallError(DeviceCallStatus::Timeout,
                                "no reusable device_call ring slot available");
      }
    }
    CUDAQ_DBG("[device-call] gpu channel acquired slot={} requestId={} "
              "requiredBytes={}",
              state->slot, state->requestId, requiredBytes);
  }

  std::uint64_t dispatchFrame(DeviceCallFrame &frame) override {
    CUDAQ_DBG("[device-call] gpu channel dispatch frame functionId={} "
              "requestCapacity={} responseCapacity={}",
              frame.functionId, frame.request.capacity,
              frame.response.capacity);
    const auto [state, slot, fireAndForget] = [&] {
      std::lock_guard<std::mutex> lock(mutex);
      if (!frame.channelPrivate)
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "invalid device_call frame");
      auto *const state =
          static_cast<RingBufferWrapper::FrameState *>(frame.channelPrivate);
      if (!state->inUse)
        throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                              "device_call frame is not active");

      const auto slot = state->slot;
      CUDAQ_DBG("[device-call] gpu channel signal slot={} requestId={}", slot,
                state->requestId);
      ringBuffer.signalDeviceSlot(slot);

      if (frame.response.capacity == 0) {
        CUDAQ_DBG("[device-call] gpu channel fire-and-forget dispatch slot={} "
                  "requestId={}",
                  slot, state->requestId);
        state->fireAndForgetPending = true;
        return std::tuple{state, slot, true};
      }
      return std::tuple{state, slot, false};
    }(); // release before blocking; each slot is independent

    if (fireAndForget)
      return 0;

    const auto responseBytes = [&]() -> std::uint64_t {
      try {
        return ringBuffer.waitForResponseAndValidate(*state, frame);
      } catch (const DeviceCallError &err) {
        std::lock_guard<std::mutex> lock(
            mutex); // re-acquire for shared state cleanup
        if (err.status() == DeviceCallStatus::Timeout) {
          frame.channelPrivate = nullptr;
          stopNoLock();
        }
        throw;
      }
    }();

    CUDAQ_DBG("[device-call] gpu channel dispatch complete slot={} "
              "requestId={} responseBytes={}",
              slot, state->requestId, responseBytes);
    return responseBytes;
  }

  void releaseFrame(DeviceCallFrame &frame) noexcept override {
    CUDAQ_DBG("[device-call] gpu channel release frame functionId={} "
              "hasFrame={}",
              frame.functionId, frame.channelPrivate != nullptr);
    std::lock_guard<std::mutex> lock(mutex);
    if (!frame.channelPrivate)
      return;
    auto *const state =
        static_cast<RingBufferWrapper::FrameState *>(frame.channelPrivate);
    if (state->inUse) {
      if (!state->fireAndForgetPending)
        ringBuffer.clearDeviceSlot(state->slot);
      state->inUse = false;
    }
    frame = {};
  }

  void stop() noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    try {
      stopNoLock();
    } catch (...) {
    }
  }

private:
  bool isSlotReady(RingBufferWrapper::FrameState &state) {
    if (state.fireAndForgetPending) {
      const std::uint64_t rxFlag =
          ringBuffer.ringbuffer()->rx_flags_host[state.slot];
      if (rxFlag != 0)
        return false;
      ringBuffer.clearDeviceSlot(state.slot);
      state.fireAndForgetPending = false;
    }
    return cudaq_host_ringbuffer_slot_available(ringBuffer.ringbuffer(),
                                                state.slot);
  }

  void ensureDispatcherStarted() {
    if (started)
      return;
    CUDAQ_DBG("[device-call] gpu channel start dispatcher");

    try {
      ringBuffer.configure("gpu", deviceId, channelConfig);

      CUDAQ_DBG("[device-call] gpu dispatcher starting device={} slots={} "
                "slotSize={} functions={}",
                deviceId, ringBuffer.numSlots(), ringBuffer.slotSize(),
                functionCount);

      const cudaq_dispatcher_config_t dispatchConfig = [&] {
        cudaq_dispatcher_config_t result{};
        result.device_id = deviceId;
        result.num_blocks = /*num_blocks=*/1;
        result.threads_per_block = /*threads_per_block=*/64;
        result.num_slots = ringBuffer.numSlots();
        result.slot_size = static_cast<std::uint32_t>(ringBuffer.slotSize());
        result.kernel_type = CUDAQ_KERNEL_REGULAR;
        result.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
        result.dispatch_path = CUDAQ_DISPATCH_PATH_DEVICE;
        return result;
      }();

      CUDAQ_CHECK_DISPATCHER_STATUS(
          cudaq_dispatch_manager_create(&dispatchManager));
      CUDAQ_CHECK_DISPATCHER_STATUS(cudaq_dispatcher_create(
          dispatchManager, &dispatchConfig, &dispatcher));
      CUDAQ_CHECK_DISPATCHER_STATUS(
          cudaq_dispatcher_set_ringbuffer(dispatcher, ringBuffer.ringbuffer()));

      const cudaq_function_table_t table = [&] {
        cudaq_function_table_t result{};
        result.entries = functionTable;
        result.count = functionCount;
        return result;
      }();
      CUDAQ_CHECK_DISPATCHER_STATUS(
          cudaq_dispatcher_set_function_table(dispatcher, &table));
      CUDAQ_CHECK_DISPATCHER_STATUS(
          cudaq_dispatcher_set_launch_fn(dispatcher, launchFn));
      CUDAQ_CHECK_DISPATCHER_STATUS(cudaq_dispatcher_start(dispatcher));

      started = true;
      CUDAQ_DBG("[device-call] gpu dispatcher started device={} slots={} "
                "slotSize={}",
                deviceId, ringBuffer.numSlots(), ringBuffer.slotSize());
    } catch (...) {
      stopNoLock();
      throw;
    }
  }

  void stopNoLock() {
    const bool hasResources =
        started || dispatcher || dispatchManager || ringBuffer.configured();
    if (!hasResources)
      return;
    CUDAQ_DBG("[device-call] gpu dispatcher stop started={}", started);
    if (started && dispatcher) {
      cudaq_dispatcher_stop(dispatcher);
      if (synchronizeFn)
        synchronizeFn();
    }
    if (dispatcher) {
      cudaq_dispatcher_destroy(dispatcher);
      dispatcher = nullptr;
    }
    if (dispatchManager) {
      cudaq_dispatch_manager_destroy(dispatchManager);
      dispatchManager = nullptr;
    }

    ringBuffer.reset();
    started = false;
    CUDAQ_DBG("[device-call] gpu dispatcher stopped");
  }

  cudaq_function_entry_t *functionTable = nullptr;
  std::uint32_t functionCount = 0;
  int deviceId = 0;
  DeviceCallChannelConfig channelConfig;
  cudaq_dispatch_launch_fn_t launchFn = nullptr;
  cudaq::realtime::DeviceCallDispatchSynchronizeFn synchronizeFn = nullptr;
  RingBufferWrapper ringBuffer;
  bool started = false;
  cudaq_dispatch_manager_t *dispatchManager = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;
  mutable std::mutex mutex;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, GpuDispatchChannel, device_dispatch)

} // namespace
