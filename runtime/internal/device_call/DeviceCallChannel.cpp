/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallChannel.h"
#include "cudaq_internal/device_call/DeviceCallError.h"

#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::vector<std::string> defaultPluginSearchPaths() {
  // CUDAQ_DEVICE_CALL_PLUGIN_PATH: a ':'-separated list of directories that
  // contain channel plugins. These directories are searched before the runtime
  // library's own plugin locations so tests and deployments can override or
  // extend the available channel implementations.
  std::vector<std::string> paths = [] {
    std::vector<std::string> paths;
    const char *pathList = std::getenv("CUDAQ_DEVICE_CALL_PLUGIN_PATH");
    if (!pathList || !*pathList) {
      return paths;
    }
    CUDAQ_DBG("[device-call] CUDAQ_DEVICE_CALL_PLUGIN_PATH={}", pathList);

    llvm::SmallVector<llvm::StringRef> entries;
    llvm::StringRef(pathList).split(entries, ':', -1, false);
    for (llvm::StringRef entry : entries) {
      paths.push_back(entry.str());
      CUDAQ_DBG("[device-call] added plugin search path '{}'", paths.back());
    }
    return paths;
  }();

  // By default, search the `plugins` directory in the CUDA-Q install.
  const std::string cudaqLibPath = cudaq::getCUDAQLibraryPath();
  llvm::StringRef libDir = llvm::sys::path::parent_path(cudaqLibPath);
  if (!libDir.empty()) {
    paths.push_back([libDir] {
      llvm::SmallString<256> path(libDir);
      llvm::sys::path::append(path, "plugins");
      return path.str().str();
    }());
    paths.push_back(libDir.str());
  }

  return paths;
}

bool tryLoadChannelPlugin(const std::string &name) {
  CUDAQ_DBG("[device-call] trying to load channel plugin '{}'", name);
  static std::mutex mutex;
  static std::vector<void *> loadedPlugins;

  std::lock_guard<std::mutex> lock(mutex);
  if (cudaq::registry::isRegistered<
          cudaq_internal::device_call::DeviceCallChannel>(name)) {
    CUDAQ_DBG("[device-call] channel '{}' is already registered", name);
    return true;
  }

  const auto canonicalPluginName = [](const std::string &name) {
    return "libcudaq-device-call-channel-" + name + ".so";
  };
  const std::string libraryName = canonicalPluginName(name);
  std::vector<std::string> candidates;
  for (const std::string &dir : defaultPluginSearchPaths())
    candidates.push_back([&dir, &libraryName] {
      llvm::SmallString<256> path(dir);
      llvm::sys::path::append(path, libraryName);
      return path.str().str();
    }());
  // Keep a bare library-name candidate last so the platform loader can still
  // resolve plugins through its configured search path.
  candidates.push_back(libraryName);

  for (const std::string &candidate : candidates) {
    CUDAQ_DBG("[device-call] loading channel plugin candidate '{}'", candidate);
    // Clear any existing error state
    ::dlerror();
    void *const handle = ::dlopen(candidate.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      CUDAQ_DBG("[device-call] channel plugin candidate '{}' was not loaded",
                candidate);
      continue;
    }
    CUDAQ_DBG("[device-call] loaded channel plugin candidate '{}'", candidate);
    loadedPlugins.push_back(handle);
    if (cudaq::registry::isRegistered<
            cudaq_internal::device_call::DeviceCallChannel>(name)) {
      CUDAQ_DBG("[device-call] channel '{}' registered by plugin '{}'", name,
                candidate);
      return true;
    }
  }

  CUDAQ_DBG("[device-call] no plugin registered channel '{}'", name);
  return false;
}

} // namespace

namespace cudaq_internal::device_call {

std::unique_ptr<DeviceCallChannel>
createDeviceCallChannel(const std::string &name,
                        DeviceCallChannelCreateArgs args) {
  CUDAQ_DBG("[device-call] create channel '{}' functions={} device={} "
            "slots={} slotSize={} timeoutMs={} args={}",
            name, args.functionCount, args.deviceId,
            args.channelConfig.numSlots, args.channelConfig.slotSize,
            args.channelConfig.timeoutMs, args.arguments.size());
  // Channel implementations may either be linked in already or provided by a
  // runtime plugin. Check the registry first, then try to load the plugin
  // library
  if (!cudaq::registry::isRegistered<DeviceCallChannel>(name) &&
      !tryLoadChannelPlugin(name))
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "unknown CUDA-Q device_call channel '" + name + "'");

  CUDAQ_DBG("[device-call] instantiating registered channel '{}'", name);
  auto nextChannel = cudaq::registry::get<DeviceCallChannel>(name);
  if (!nextChannel)
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "failed to create CUDA-Q device_call channel '" +
                              name + "'");

  nextChannel->initialize(std::move(args));
  CUDAQ_INFO("[device-call] channel '{}' initialized", name);
  return nextChannel;
}

} // namespace cudaq_internal::device_call

CUDAQ_INSTANTIATE_REGISTRY(
    cudaq_internal::device_call::DeviceCallChannel::RegistryType)
