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

#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>

namespace {

using namespace cudaq_internal::device_call;

class HostDispatchChannel : public DeviceCallChannel {
public:
  ~HostDispatchChannel() override { stop(); }

  void initialize(DeviceCallChannelCreateArgs &&args) override {
    CUDAQ_INFO("[device-call] host channel initialize functionCount={} "
               "device={} hasMailbox={} slots={} slotSize={} timeoutMs={}",
               args.functionCount, args.deviceId, args.mailbox != nullptr,
               args.channelConfig.numSlots, args.channelConfig.slotSize,
               args.channelConfig.timeoutMs);
    if (!args.functionTable || args.functionCount == 0)
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "device_call function table is empty");
    functionTable = args.functionTable;
    functionCount = args.functionCount;
    deviceId = args.deviceId;
    channelConfig = args.channelConfig;
    mailbox = args.mailbox;

    if (!hasSupportedEntry())
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "host device_call dispatch requires a HOST_CALL or graph-launch "
          "table entry");

    // Graph-launch workers require a pinned, mapped mailbox. HOST_CALL
    // handlers execute inline on the host dispatcher thread and do not use it.
    if (hasGraphLaunchEntry() && !mailbox)
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "host device_call graph-launch entries require a pinned, mapped "
          "mailbox "
          "(cudaHostAlloc with cudaHostAllocMapped) sized for the "
          "GRAPH_LAUNCH worker count");
  }

  void acquireFrame(std::uint32_t functionId, std::uint64_t requestBytes,
                    std::uint64_t responseCapacity,
                    DeviceCallFrame &frame) override {
    CUDAQ_DBG("[device-call] host channel acquire frame functionId={} "
              "requestBytes={} responseCapacity={}",
              functionId, requestBytes, responseCapacity);
    frame = {};
    const std::uint64_t requiredBytes =
        requiredRingSlotSize(requestBytes, responseCapacity);

    const auto isReady = [this](RingBufferWrapper::FrameState &state) -> bool {
      if (state.fireAndForgetPending) {
        const cudaq_tx_status_t txStatus = cudaq_host_ringbuffer_poll_tx_flag(
            ringBuffer.ringbuffer(), state.slot, nullptr);
        if (txStatus == CUDAQ_TX_EMPTY || txStatus == CUDAQ_TX_IN_FLIGHT)
          return false;
        ringBuffer.clearSlot(state.slot);
        state.fireAndForgetPending = false;
      }
      return cudaq_host_ringbuffer_slot_available(ringBuffer.ringbuffer(),
                                                  state.slot);
    };

    RingBufferWrapper::FrameState *state = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex);
      const auto *const entry = lookup(functionId);
      CUDAQ_DBG("[device-call] host channel lookup functionId={} found={}",
                functionId, static_cast<bool>(entry));
      if (!entry || !isSupportedEntry(*entry))
        throw DeviceCallError(
            DeviceCallStatus::InvalidArgument,
            "host device_call dispatch requires a HOST_CALL or graph-launch "
            "table entry");
      ensureDispatcherStarted();
      if (requiredBytes > ringBuffer.slotSize())
        throw DeviceCallError(
            DeviceCallStatus::InvalidArgument,
            "host device_call frame exceeds configured slot size");
      if (auto slot = ringBuffer.tryGetReadySlot(isReady))
        state = &ringBuffer.claimSlot(*slot, functionId, requestBytes,
                                      responseCapacity, frame,
                                      ringBuffer.slotSize());
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
                                          ringBuffer.slotSize());
        }
        if (std::chrono::steady_clock::now() > deadline)
          throw DeviceCallError(
              DeviceCallStatus::Timeout,
              "no reusable host device_call ring slot available");
      }
    }
    CUDAQ_DBG("[device-call] host channel acquired slot={} requestId={} "
              "requiredBytes={}",
              state->slot, state->requestId, requiredBytes);
  }

  std::uint64_t dispatchFrame(DeviceCallFrame &frame) override {
    CUDAQ_DBG("[device-call] host channel dispatch frame functionId={} "
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
      CUDAQ_DBG("[device-call] host channel signal slot={} requestId={}", slot,
                state->requestId);
      ringBuffer.signalHostSlot(slot);

      if (frame.response.capacity == 0) {
        CUDAQ_DBG("[device-call] host channel fire-and-forget dispatch slot={} "
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
        return ringBuffer.waitForResponseAndValidate(*state, frame, true);
      } catch (const DeviceCallError &err) {
        // Slot cleanup is the caller's responsibility via releaseFrame, so
        // for recoverable errors (ResponseTooLarge, RemoteError, ...) leave
        // the frame intact and let releaseFrame reclaim the slot for reuse.
        // For Timeout/CudaError we tear the dispatcher down, which frees the
        // ring buffer's FrameState storage — null out channelPrivate so the
        // caller's releaseFrame doesn't dereference a dangling pointer.
        if (err.status() == DeviceCallStatus::Timeout ||
            err.status() == DeviceCallStatus::CudaError) {
          std::lock_guard<std::mutex> lock(mutex);
          frame.channelPrivate = nullptr;
          stopDispatcher();
        }
        throw;
      }
    }();

    CUDAQ_DBG("[device-call] host channel dispatch complete slot={} "
              "requestId={} responseBytes={}",
              slot, state->requestId, responseBytes);
    return responseBytes;
  }

  void releaseFrame(DeviceCallFrame &frame) noexcept override {
    CUDAQ_DBG("[device-call] host channel release frame functionId={} "
              "hasFrame={}",
              frame.functionId, frame.channelPrivate != nullptr);
    std::lock_guard<std::mutex> lock(mutex);
    if (!frame.channelPrivate)
      return;
    auto *const state =
        static_cast<RingBufferWrapper::FrameState *>(frame.channelPrivate);
    if (state->inUse) {
      if (!state->fireAndForgetPending)
        ringBuffer.clearSlot(state->slot);
      state->inUse = false;
    }
    frame = {};
  }

  void stop() noexcept override {
    std::lock_guard<std::mutex> lock(mutex);
    try {
      stopDispatcher();
    } catch (...) {
    }
  }

private:
  static bool isSupportedEntry(const cudaq_function_entry_t &entry) {
    if (entry.dispatch_mode == CUDAQ_DISPATCH_HOST_CALL)
      return entry.handler.host_fn != nullptr;
    if (entry.dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
      return entry.handler.graph_exec != nullptr;
    return false;
  }

  const cudaq_function_entry_t *lookup(std::uint32_t functionId) const {
    for (std::uint32_t i = 0; i < functionCount; ++i)
      if (functionTable[i].function_id == functionId)
        return &functionTable[i];
    return nullptr;
  }

  bool hasGraphLaunchEntry() const {
    for (std::uint32_t i = 0; i < functionCount; ++i)
      if (functionTable[i].dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH &&
          functionTable[i].handler.graph_exec)
        return true;
    return false;
  }

  bool hasSupportedEntry() const {
    for (std::uint32_t i = 0; i < functionCount; ++i)
      if (isSupportedEntry(functionTable[i]))
        return true;
    return false;
  }

  void ensureDispatcherStarted() {
    if (dispatcherStarted)
      return;
    CUDAQ_DBG("[device-call] host channel start dispatcher");

    try {
      ringBuffer.configure("host", deviceId, channelConfig);

      const bool usesGraphLaunch = hasGraphLaunchEntry();

      CUDAQ_DBG("[device-call] host dispatcher starting device={} "
                "slots={} slotSize={} functions={} hasMailbox={}",
                deviceId, ringBuffer.numSlots(), ringBuffer.slotSize(),
                functionCount, mailbox != nullptr);

      const cudaq_dispatcher_config_t dispatchConfig = [&] {
        cudaq_dispatcher_config_t result{};
        result.device_id = deviceId;
        result.num_slots = ringBuffer.numSlots();
        result.slot_size = static_cast<std::uint32_t>(ringBuffer.slotSize());
        result.kernel_type = CUDAQ_KERNEL_REGULAR;
        result.dispatch_mode = usesGraphLaunch ? CUDAQ_DISPATCH_GRAPH_LAUNCH
                                               : CUDAQ_DISPATCH_HOST_CALL;
        result.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
        return result;
      }();

      CUDAQ_CHECK_DISPATCHER_STATUS(
          cudaq_dispatch_manager_create(&dispatchManager));
      CUDAQ_CHECK_DISPATCHER_STATUS(cudaq_dispatcher_create(
          dispatchManager, &dispatchConfig, &graphDispatcher));
      CUDAQ_CHECK_DISPATCHER_STATUS(cudaq_dispatcher_set_ringbuffer(
          graphDispatcher, ringBuffer.ringbuffer()));

      const cudaq_function_table_t table = [&] {
        cudaq_function_table_t result{};
        result.entries = functionTable;
        result.count = functionCount;
        return result;
      }();
      CUDAQ_CHECK_DISPATCHER_STATUS(
          cudaq_dispatcher_set_function_table(graphDispatcher, &table));

      shutdownFlag = 0;
      stats = 0;
      CUDAQ_CHECK_DISPATCHER_STATUS(
          cudaq_dispatcher_set_control(graphDispatcher, &shutdownFlag, &stats));
      if (mailbox)
        CUDAQ_CHECK_DISPATCHER_STATUS(
            cudaq_dispatcher_set_mailbox(graphDispatcher, mailbox));
      CUDAQ_CHECK_DISPATCHER_STATUS(cudaq_dispatcher_start(graphDispatcher));

      dispatcherStarted = true;
      CUDAQ_DBG("[device-call] host dispatcher started device={} "
                "slots={} slotSize={}",
                deviceId, ringBuffer.numSlots(), ringBuffer.slotSize());
    } catch (...) {
      stopDispatcher();
      throw;
    }
  }

  void stopDispatcher() {
    const bool hasResources = dispatcherStarted || graphDispatcher ||
                              dispatchManager || ringBuffer.configured();
    if (!hasResources)
      return;
    CUDAQ_DBG("[device-call] host dispatcher stop started={}",
              dispatcherStarted);
    if (dispatcherStarted && graphDispatcher)
      cudaq_dispatcher_stop(graphDispatcher);
    if (graphDispatcher) {
      cudaq_dispatcher_destroy(graphDispatcher);
      graphDispatcher = nullptr;
    }
    if (dispatchManager) {
      cudaq_dispatch_manager_destroy(dispatchManager);
      dispatchManager = nullptr;
    }

    ringBuffer.reset();
    dispatcherStarted = false;
    CUDAQ_DBG("[device-call] host dispatcher stopped");
  }

  cudaq_function_entry_t *functionTable = nullptr;
  std::uint32_t functionCount = 0;
  int deviceId = 0;
  DeviceCallChannelConfig channelConfig;
  void **mailbox = nullptr;
  RingBufferWrapper ringBuffer;
  bool dispatcherStarted = false;
  cudaq_dispatch_manager_t *dispatchManager = nullptr;
  cudaq_dispatcher_t *graphDispatcher = nullptr;
  volatile int shutdownFlag = 0;
  std::uint64_t stats = 0;
  mutable std::mutex mutex;
};

CUDAQ_REGISTER_TYPE(DeviceCallChannel, HostDispatchChannel, host_dispatch)

} // namespace
