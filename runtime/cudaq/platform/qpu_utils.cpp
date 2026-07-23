/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu_utils.h"
#include "common/Executor.h"
#include "common/RuntimeTarget.h"
#include "common/ServerHelper.h"
#include "nvqpp_config.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Target/TargetConfig.h"
#include "cudaq/Target/TargetConfigYaml.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/Support/Base64.h"
#include <algorithm>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace cudaq;

void detail::parseTargetConfigYml(const std::string &yamlContent,
                                  config::TargetConfig &targetConfig) {
  targetConfig = config::parseTargetConfig(yamlContent);
}

std::string detail::decodeBase64(const std::string &encoded) {
  std::vector<char> decoded_vec;
  if (auto err = llvm::decodeBase64(encoded, decoded_vec))
    throw std::runtime_error("DecodeBase64 error");
  return std::string(decoded_vec.data(), decoded_vec.size());
}

std::optional<std::string>
detail::getBackendConfigOption(const std::string &backend,
                               std::string_view key) {
  auto split = cudaq::split(backend, ';');
  for (std::size_t i = 1; i + 1 < split.size(); i += 2) {
    if (split[i] != key)
      continue;

    auto value = split[i + 1];
    if (value.starts_with("base64_")) {
      value.erase(0, 7);
      return decodeBase64(value);
    }
    return value;
  }
  return std::nullopt;
}

std::filesystem::path
detail::getTargetConfigPath(const std::string &backend,
                            const std::filesystem::path &fallback) {
  if (auto path = getBackendConfigOption(backend, "__yml_path"))
    return *path;
  return fallback;
}

namespace {
void loadTargetPluginLibrary(const std::filesystem::path &path,
                             const std::string &targetName) {
  static std::mutex mutex;
  static std::unordered_map<std::string, void *> handles;
  const auto pathString = path.string();
  {
    std::lock_guard lock(mutex);
    if (handles.contains(pathString))
      return;
  }

  CUDAQ_INFO("Loading plugin library '{}' for target '{}'.", pathString,
             targetName);
  void *handle = dlopen(pathString.c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (!handle) {
    const char *error = dlerror();
    throw std::runtime_error("Unable to load plugin library '" + pathString +
                             "' for target '" + targetName +
                             "': " + (error ? error : "unknown error"));
  }

  std::lock_guard lock(mutex);
  handles.emplace(pathString, handle);
}
} // namespace

void detail::loadTargetPluginLibraries(
    const std::string &targetName, const std::filesystem::path &configPath,
    const config::TargetConfig &targetConfig) {
  const std::filesystem::path cudaqLibraryPath{cudaq::getCUDAQLibraryPath()};
  const auto cudaqLibDir = cudaqLibraryPath.parent_path();
  const auto configDir = configPath.parent_path();
  const auto pluginRoot =
      configDir.filename() == "targets" ? configDir.parent_path() : configDir;
  const auto pluginLibDir = pluginRoot / "lib";

  for (const auto &pluginLibrary : targetConfig.PluginLibraries) {
    const std::filesystem::path requestedPath(pluginLibrary);
    std::vector<std::filesystem::path> candidates;
    if (requestedPath.is_absolute()) {
      candidates.push_back(requestedPath);
    } else {
      candidates.push_back(cudaqLibDir / requestedPath);
      candidates.push_back(pluginLibDir / requestedPath);
    }

    const auto found = std::find_if(candidates.begin(), candidates.end(),
                                    [](const auto &candidate) {
                                      return std::filesystem::exists(candidate);
                                    });
    if (found == candidates.end()) {
      std::string searchedPaths;
      for (const auto &candidate : candidates) {
        if (!searchedPaths.empty())
          searchedPaths += ", ";
        searchedPaths += candidate.string();
      }
      throw std::runtime_error("Unable to find plugin library '" +
                               pluginLibrary + "' for target '" + targetName +
                               "'. Searched: " + searchedPaths);
    }
    loadTargetPluginLibrary(*found, targetName);
  }

  const auto serverHelperName =
      "libcudaq-serverhelper-" + targetName + PLATFORM_SHARED_LIBRARY_SUFFIX;
  for (const auto &candidate :
       {cudaqLibDir / serverHelperName, pluginLibDir / serverHelperName}) {
    if (!std::filesystem::exists(candidate))
      continue;
    loadTargetPluginLibrary(candidate, targetName);
    break;
  }

  // External custom QPU plugins ship libcudaq-<platform-qpu>-qpu.so (same
  // naming as in-tree Fermioniq). Load before registry::get<QPU>.
  if (targetConfig.BackendConfig.has_value() &&
      !targetConfig.BackendConfig->PlatformQpu.empty()) {
    const auto qpuLibName = "libcudaq-" +
                            targetConfig.BackendConfig->PlatformQpu + "-qpu" +
                            PLATFORM_SHARED_LIBRARY_SUFFIX;
    for (const auto &candidate :
         {cudaqLibDir / qpuLibName, pluginLibDir / qpuLibName}) {
      if (!std::filesystem::exists(candidate))
        continue;
      loadTargetPluginLibrary(candidate, targetName);
      break;
    }
  }
}

bool detail::isAnalogHamiltonianKernel(const std::string &kernelName) {
  return kernelName.find(cudaq::runtime::cudaqAHKPrefixName) == 0;
}

void detail::initServerHelperAndExecutor(
    const std::string &qpuName,
    const std::map<std::string, std::string> &backendConfig,
    const config::TargetConfig &targetConfig,
    owning_ptr<ServerHelper> &serverHelper,
    std::unique_ptr<Executor> &executor) {
  // Create the ServerHelper for this QPU and give it the backend config.
  // The registry hands back a `unique_ptr<ServerHelper>` with the default
  // deleter; rebind it into an owning_ptr<ServerHelper> so that destruction
  // goes through the out-of-line `opaque_deleter<ServerHelper>` defined in
  // ServerHelper.cpp.
  auto raw = cudaq::registry::get<cudaq::ServerHelper>(qpuName);
  if (!raw) {
    throw std::runtime_error("ServerHelper not found for target: " + qpuName);
  }
  serverHelper = owning_ptr<ServerHelper>(raw.release());

  serverHelper->initialize(backendConfig);
  CUDAQ_INFO("Retrieving executor with name {}", qpuName);
  CUDAQ_INFO("Is this executor registered? {}",
             cudaq::registry::isRegistered<cudaq::Executor>(qpuName));
  executor = cudaq::registry::isRegistered<cudaq::Executor>(qpuName)
                 ? cudaq::registry::get<cudaq::Executor>(qpuName)
                 : std::make_unique<cudaq::Executor>();

  // Give the server helper to the executor
  executor->setServerHelper(serverHelper.get());

  // Construct the runtime target
  RuntimeTarget runtimeTarget;
  runtimeTarget.config = targetConfig;
  runtimeTarget.name = qpuName;
  runtimeTarget.description = targetConfig.Description;
  runtimeTarget.runtimeConfig = backendConfig;
  serverHelper->setRuntimeTarget(runtimeTarget);
}
