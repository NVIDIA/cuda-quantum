/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime.h"
#include "cudaq/runtime/logger/logger.h"

#if defined(CUDAQ_ENABLE_REALTIME)
#include "common/PluginUtils.h"
#include "cudaq_internal/device_call/DeviceCallRuntimePlugin.h"
#include "cudaq/platform.h"
#endif

#include <stdexcept>
#include <string>

namespace {

void warnRealtimeNotCompiled(const char *functionName) {
  CUDAQ_WARN("cudaq::realtime::{}() ignored because this CUDA-Q build was "
             "compiled without realtime support.",
             functionName);
}

#if defined(CUDAQ_ENABLE_REALTIME)
constexpr const char *DeviceCallRuntimePluginSymbol =
    "getCudaqDeviceCallRuntime_device_call";

cudaq_internal::device_call::DeviceCallRuntimePlugin &
getRealtimeRuntimePlugin() {
  try {
    auto *const plugin = cudaq::getUniquePluginInstance<
        cudaq_internal::device_call::DeviceCallRuntimePlugin>(
        DeviceCallRuntimePluginSymbol);
    if (plugin)
      return *plugin;
  } catch (const std::exception &err) {
    throw std::runtime_error(
        "cudaq::realtime requires the realtime device-call runtime library to "
        "be linked into this process: " +
        std::string(err.what()));
  }

  throw std::runtime_error(
      "cudaq::realtime requires the realtime device-call runtime library to "
      "be linked into this process, but the runtime plugin factory returned "
      "null.");
}
#endif

} // namespace

namespace cudaq::realtime {

void initialize(int argc, char **argv) {
#if defined(CUDAQ_ENABLE_REALTIME)
  getRealtimeRuntimePlugin().initialize(argc, argv);
  // Validate the initialized runtime against the selected CUDA-Q target.
  getRealtimeRuntimePlugin().validate(
      cudaq::get_platform().get_runtime_target());
#else
  (void)argc;
  (void)argv;
  warnRealtimeNotCompiled("initialize");
#endif
}

void initialize() { initialize(0, nullptr); }

void finalize() {
#if defined(CUDAQ_ENABLE_REALTIME)
  getRealtimeRuntimePlugin().finalize();
#else
  warnRealtimeNotCompiled("finalize");
#endif
}

} // namespace cudaq::realtime
