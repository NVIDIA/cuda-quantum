/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallRuntimePlugin.h"

#include "common/RuntimeTarget.h"
#include "cudaq/runtime/logger/logger.h"

#include <stdexcept>

namespace cudaq_internal::device_call {
void initializeDeviceCallRuntime(int argc, char **argv);
bool isGpuDispatchRuntimeConfigured();
void finalizeDeviceCallRuntime();
} // namespace cudaq_internal::device_call

namespace {

class DeviceCallRuntimePluginImpl
    : public cudaq_internal::device_call::DeviceCallRuntimePlugin {
public:
  void initialize(int argc, char **argv) override {
    CUDAQ_INFO("[device-call] realtime plugin initialize argc={}", argc);
    cudaq_internal::device_call::initializeDeviceCallRuntime(argc, argv);
  }

  void validate(const cudaq::RuntimeTarget *target) override {
    if (!target || !target->config.GpuRequired ||
        !cudaq_internal::device_call::isGpuDispatchRuntimeConfigured())
      return;

    // GPU simulator targets synchronize with CUDA internally, which conflicts
    // with the realtime GPU dispatch loop. Stop the dispatcher before throwing
    // so users see the configuration error instead of callback timeouts.
    cudaq_internal::device_call::finalizeDeviceCallRuntime();
    throw std::runtime_error(
        "cudaq::realtime GPU dispatch is not supported with GPU simulator "
        "target '" +
        target->name + "'. Use host dispatch or a CPU simulator target.");
  }

  void finalize() override {
    CUDAQ_INFO("[device-call] realtime plugin finalize");
    cudaq_internal::device_call::finalizeDeviceCallRuntime();
  }
};

cudaq_internal::device_call::DeviceCallRuntimePlugin *
getDeviceCallRuntimePlugin() {
  static DeviceCallRuntimePluginImpl plugin;
  return &plugin;
}

} // namespace

extern "C" cudaq_internal::device_call::DeviceCallRuntimePlugin *
getCudaqDeviceCallRuntime_device_call() {
  return getDeviceCallRuntimePlugin();
}
