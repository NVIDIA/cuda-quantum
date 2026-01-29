/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "cudaq.h"
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

// Shared mutex to guard concurrent access to global kernel data (e.g.,
// `quakeRegistry`, `kernelRegistry`, `argsCreators`, `lambdaNames`).
// These global variables might be accessed (write or read) concurrently, e.g.,
// async. execution of kernels or via CUDA Quantum API (e.g.,
// `get_quake_by_name`). Note: currently, we use a single mutex for all static
// global variables for simplicity since these containers are small and not
// frequently accessed.
static std::shared_mutex globalRegistryMutex;

//===----------------------------------------------------------------------===//
// Registry that maps device code keys to strings of device code. The map is
// created at program startup and can be used to find code to be
// compiled/executed at runtime.
//===----------------------------------------------------------------------===//

static std::vector<std::pair<std::string, std::string>> quakeRegistry;

void cudaq::registry::__cudaq_deviceCodeHolderAdd(const char *key,
                                                  const char *code) {
  std::unique_lock<std::shared_mutex> lock(globalRegistryMutex);
  auto it = std::find_if(quakeRegistry.begin(), quakeRegistry.end(),
                         [&](const auto &pair) { return pair.first == key; });
  if (it != quakeRegistry.end()) {
    CUDAQ_INFO("Replacing code for kernel {}", key);
    it->second = code;
    return;
  }
  quakeRegistry.emplace_back(key, code);
}

//===----------------------------------------------------------------------===//
// Registry of all kernels that have been generated. The vector of kernels is
// created at program startup time. This list can be consulted by the runtime to
// determine if a particular kernel has been processed for kernel execution,
// including adding the trampoline to call the runtime to launch the kernel.
//===----------------------------------------------------------------------===//

static std::vector<std::string> kernelRegistry;

static std::map<std::string, cudaq::KernelArgsCreator> argsCreators;
static std::map<std::string, std::string> lambdaNames;
static std::map<void *, std::pair<const char *, void *>> linkableKernelRegistry;
static std::map<std::string, void *> runnableKernelRegistry;

void cudaq::registry::cudaqRegisterKernelName(const char *kernelName) {
  std::unique_lock<std::shared_mutex> lock(globalRegistryMutex);
  kernelRegistry.emplace_back(kernelName);
}

void cudaq::registry::__cudaq_registerRunnableKernel(const char *kernelName,
                                                     void *runnableEntry) {
  std::unique_lock<std::shared_mutex> lock(globalRegistryMutex);
  runnableKernelRegistry.insert({std::string{kernelName}, runnableEntry});
}

void *cudaq::registry::getRunnableKernelOrNull(const std::string &kernelName) {
  auto iter = runnableKernelRegistry.find(kernelName);
  return (iter != runnableKernelRegistry.end()) ? iter->second : nullptr;
}

void *
cudaq::registry::__cudaq_getRunnableKernel(const std::string &kernelName) {
  void *result = getRunnableKernelOrNull(kernelName);
  if (!result)
    throw std::runtime_error("runnable kernel " + kernelName +
                             "is not present: kernel cannot be "
                             "called by cudaq::run");
  return result;
}

void cudaq::registry::__cudaq_registerLinkableKernel(void *hostSideFunc,
                                                     const char *kernelName,
                                                     void *deviceSideFunc) {
  std::unique_lock<std::shared_mutex> lock(globalRegistryMutex);
  linkableKernelRegistry.insert(
      {hostSideFunc, std::pair{kernelName, deviceSideFunc}});
}

std::intptr_t cudaq::registry::__cudaq_getLinkableKernelKey(void *p) {
  if (!p)
    throw std::runtime_error("cannot get kernel key, nullptr");
  const auto &qk = *reinterpret_cast<const cudaq::qkernel<void()> *>(p);
  return reinterpret_cast<std::intptr_t>(*qk.get_entry_kernel_from_holder());
}

const char *cudaq::registry::getLinkableKernelNameOrNull(std::intptr_t key) {
  auto iter = linkableKernelRegistry.find(reinterpret_cast<void *>(key));
  return (iter != linkableKernelRegistry.end()) ? iter->second.first : nullptr;
}

const char *cudaq::registry::__cudaq_getLinkableKernelName(std::intptr_t key) {
  auto *result = getLinkableKernelNameOrNull(key);
  if (!result)
    throw std::runtime_error("kernel key is not present: kernel name unknown");
  return result;
}

void *
cudaq::registry::__cudaq_getLinkableKernelDeviceFunction(std::intptr_t key) {
  if (key & 1) {
    // This is a python kernel decorator. The key is the function address | 1.
    // Python kernel decorators are never initialized via .init sections and are
    // not part of the C++ runtime.
    return reinterpret_cast<void *>(key ^ 1);
  }
  auto iter = linkableKernelRegistry.find(reinterpret_cast<void *>(key));
  if (iter != linkableKernelRegistry.end())
    return iter->second.second;
  throw std::runtime_error("kernel key is not present: kernel unknown");
}

void cudaq::registry::cudaqRegisterArgsCreator(const char *name,
                                               char *rawFunctor) {
  std::unique_lock<std::shared_mutex> lock(globalRegistryMutex);
  argsCreators.insert(
      {std::string(name), reinterpret_cast<KernelArgsCreator>(rawFunctor)});
}

void cudaq::registry::cudaqRegisterLambdaName(const char *name,
                                              const char *value) {
  std::unique_lock<std::shared_mutex> lock(globalRegistryMutex);
  lambdaNames.insert({std::string(name), std::string(value)});
}

bool cudaq::detail::isKernelGenerated(const std::string &kernelName) {
  std::shared_lock<std::shared_mutex> lock(globalRegistryMutex);
  return std::find(kernelRegistry.begin(), kernelRegistry.end(), kernelName) !=
         kernelRegistry.end();
}

bool cudaq::__internal__::isLibraryMode(const std::string &kernelname) {
  return !detail::isKernelGenerated(kernelname);
}

//===----------------------------------------------------------------------===//

namespace cudaq {

KernelArgsCreator getArgsCreator(const std::string &kernelName) {
  std::unique_lock<std::shared_mutex> lock(globalRegistryMutex);
  return argsCreators[kernelName];
}

std::string get_quake_by_name(const std::string &kernelName,
                              bool throwException,
                              std::optional<std::string> knownMangledArgs) {
  // A prefix name has a '.' before the C++ mangled name suffix.
  auto kernelNamePrefix = kernelName + '.';

  // Find the quake code
  std::optional<std::string> result;
  std::shared_lock<std::shared_mutex> lock(globalRegistryMutex);

  for (const auto &pair : quakeRegistry) {
    if (pair.first == kernelName) {
      // Exact match. Return the code.
      return pair.second;
    }

    if (pair.first.starts_with(kernelNamePrefix)) {
      // Prefix match. Record it and make sure that it is a unique prefix.
      if (result.has_value()) {
        if (throwException)
          throw std::runtime_error("Quake code for '" + kernelName +
                                   "' has multiple matches.\n");
      } else {
        result = pair.second;
        if (knownMangledArgs.has_value() &&
            pair.first.ends_with(*knownMangledArgs))
          break;
      }
    }
  }

  if (result.has_value())
    return *result;
  if (throwException)
    throw std::runtime_error("Quake code not found for '" + kernelName +
                             "'.\n");
  return {};
}

std::string get_quake_by_name(const std::string &kernelName) {
  return get_quake_by_name(kernelName, /*throwException=*/true);
}

std::string get_quake_by_name(const std::string &kernelName,
                              std::optional<std::string> knownMangledArgs) {
  return get_quake_by_name(kernelName, /*throwException=*/true,
                           knownMangledArgs);
}

bool kernelHasConditionalFeedback(const std::string &kernelName) {
  auto quakeCode = get_quake_by_name(kernelName, false);
  return !quakeCode.empty() &&
         quakeCode.find("qubitMeasurementFeedback = true") != std::string::npos;
}
} // namespace cudaq
