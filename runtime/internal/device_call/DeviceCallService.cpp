/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallService.h"
#include "cudaq_internal/device_call/DeviceCallError.h"

#include "cudaq/runtime/logger/logger.h"

#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>

namespace {
using namespace cudaq_internal::device_call;

class DefaultDeviceCallServiceSession : public DeviceCallServiceSession {
public:
  DefaultDeviceCallServiceSession(DeviceCallService &service,
                                  DeviceCallDispatchMode mode)
      : service(service) {
    initialize(mode);
  }

  DefaultDeviceCallServiceSession(const DefaultDeviceCallServiceSession &) =
      delete;
  DefaultDeviceCallServiceSession &
  operator=(const DefaultDeviceCallServiceSession &) = delete;

  ~DefaultDeviceCallServiceSession() override {
    try {
      reset();
    } catch (...) {
    }
  }

  const DeviceCallDispatchTable &dispatchTable() const noexcept override {
    return table;
  }

  void start() override {
    CUDAQ_DBG("[device-call] service session start started={}", started);
    if (started)
      return;

    const int hookStatus = service.start();
    CUDAQ_DBG("[device-call] service start hook status={}", hookStatus);
    if (hookStatus != 0)
      throw std::runtime_error("device_call service start hook failed");

    started = true;
  }

  void stop() noexcept override {
    try {
      reset();
    } catch (...) {
    }
  }

private:
  void initialize(DeviceCallDispatchMode mode) {
    CUDAQ_INFO("[device-call] service session initialize mode={}",
               mode == DeviceCallDispatchMode::Host ? "host" : "gpu");

    if (int hookStatus = service.create(nullptr, 0)) {
      CUDAQ_DBG("[device-call] service create hook status={}", hookStatus);
      reset();
      throw std::runtime_error("device_call service create hook failed");
    }
    created = true;

    if (mode == DeviceCallDispatchMode::Host)
      initializeHostDispatch();
    else
      initializeGpuDispatch();
  }

  void initializeGpuDispatch() {
    const std::uint32_t functionCount = service.getFunctionCount();
    CUDAQ_DBG("[device-call] service reports {} GPU dispatch functions",
              functionCount);
    if (functionCount == 0) {
      reset();
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            "device_call service exported no functions");
    }

    const auto bytes = functionCount * sizeof(cudaq_function_entry_t);
    CUDAQ_DBG("[device-call] allocating GPU function table bytes={}", bytes);
    void *allocated = nullptr;
    if (cudaMalloc(&allocated, bytes) != cudaSuccess) {
      reset();
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to allocate device_call function table");
    }
    d_ownedFunctionEntries = static_cast<cudaq_function_entry_t *>(allocated);

    if (cudaMemset(d_ownedFunctionEntries, 0, bytes) != cudaSuccess) {
      reset();
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to initialize device_call function table");
    }

    CUDAQ_DBG("[device-call] populating GPU function table entries={}",
              functionCount);
    if (service.populateTable(d_ownedFunctionEntries, functionCount, nullptr) !=
            0 ||
        cudaDeviceSynchronize() != cudaSuccess) {
      reset();
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            "failed to populate device_call function table");
    }

    const cudaq_dispatch_launch_fn_t launchFn =
        service.getDeviceDispatchLaunch();
    if (!launchFn) {
      reset();
      throw DeviceCallError(
          DeviceCallStatus::InvalidArgument,
          "device_call service is missing dispatch launch hook");
    }

    table.mode = DeviceCallDispatchMode::Gpu;
    table.entries = d_ownedFunctionEntries;
    table.count = functionCount;
    table.launchFn = launchFn;
    table.synchronizeFn = service.getDeviceDispatchSynchronize();
    CUDAQ_INFO("[device-call] service GPU dispatch initialized launchFn={} "
               "synchronizeFn={}",
               static_cast<bool>(table.launchFn),
               static_cast<bool>(table.synchronizeFn));
  }

  void initializeHostDispatch() {
    const DeviceCallHostDispatchTable hostTable = [&] {
      DeviceCallHostDispatchTable result{};
      CUDAQ_DBG("[device-call] requesting host dispatch function table");
      if (service.getHostDispatchTable(result) != 0) {
        reset();
        throw std::runtime_error(
            "host device_call service graph table hook failed");
      }
      if (!result.entries || result.count == 0) {
        reset();
        throw DeviceCallError(
            DeviceCallStatus::InvalidArgument,
            "host device_call service exported no graph table entries");
      }
      return result;
    }();

    table.mode = DeviceCallDispatchMode::Host;
    table.entries = hostTable.entries;
    table.count = hostTable.count;
    table.deviceId = static_cast<int>(hostTable.deviceId);
    table.mailbox = hostTable.mailbox;
    CUDAQ_INFO("[device-call] host dispatch table count={} device={} "
               "hasMailbox={}",
               table.count, table.deviceId, table.mailbox != nullptr);
  }

  void reset() {
    CUDAQ_INFO(
        "[device-call] service session reset started={} functionCount={}",
        started, table.count);
    if (started) {
      CUDAQ_DBG("[device-call] calling service stop hook");
      service.stop();
    }
    started = false;

    if (d_ownedFunctionEntries) {
      CUDAQ_DBG("[device-call] freeing GPU function table");
      cudaFree(d_ownedFunctionEntries);
      d_ownedFunctionEntries = nullptr;
    }

    if (created) {
      CUDAQ_DBG("[device-call] calling service destroy hook");
      service.destroy();
    }
    created = false;
    table = {};
  }

  DeviceCallService &service;
  DeviceCallDispatchTable table;
  cudaq_function_entry_t *d_ownedFunctionEntries = nullptr;
  bool created = false;
  bool started = false;
};

} // namespace

namespace cudaq_internal::device_call {
std::unique_ptr<DeviceCallServiceSession>
DeviceCallService::createDispatchSession(DeviceCallDispatchMode mode) {
  return std::make_unique<DefaultDeviceCallServiceSession>(*this, mode);
}

} // namespace cudaq_internal::device_call
