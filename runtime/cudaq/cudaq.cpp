/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1

#include "common/Logger.h"
#include "cudaq/platform.h"
#include "cudaq/utils/registry.h"
#include <dlfcn.h>
#include <map>
#include <regex>
#include <string>
#include <vector>

namespace cudaq::__internal__ {
std::map<std::string, std::string> runtime_registered_mlir;
std::string demangle_kernel(const char *name) {
  return quantum_platform::demangle(name);
}
bool globalFalse = false;
} // namespace cudaq::__internal__

//===----------------------------------------------------------------------===//
// Registry that maps device code keys to strings of device code. The map is
// created at program startup and can be used to find code to be
// compiled/executed at runtime.
//===----------------------------------------------------------------------===//

static std::vector<std::pair<std::string, std::string>> quakeRegistry;

void cudaq::registry::deviceCodeHolderAdd(const char *key, const char *code) {
  quakeRegistry.emplace_back(key, code);
}

//===----------------------------------------------------------------------===//
// Registry of all kernels that have been generated. The vector of kernels is
// created at program startup time. This list can be consulted by the runtime to
// determine if a particular kernel has been processed for kernel execution,
// including adding the trampoline to call the runtime to launch the kernel.
//===----------------------------------------------------------------------===//

static std::vector<std::string_view> kernelRegistry;

static std::map<std::string, cudaq::KernelArgsCreator> argsCreators;
static std::map<std::string, std::string> lambdaNames;

void cudaq::registry::cudaqRegisterKernelName(const char *kernelName) {
  kernelRegistry.emplace_back(kernelName);
}

void cudaq::registry::cudaqRegisterArgsCreator(const char *name,
                                               char *rawFunctor) {
  argsCreators.insert(
      {std::string(name), reinterpret_cast<KernelArgsCreator>(rawFunctor)});
}

void cudaq::registry::cudaqRegisterLambdaName(const char *name,
                                              const char *value) {
  lambdaNames.insert({std::string(name), std::string(value)});
}

bool cudaq::__internal__::isKernelGenerated(const std::string &kernelName) {
  for (auto regName : kernelRegistry)
    if (kernelName == regName)
      return true;
  return false;
}

bool cudaq::__internal__::isLibraryMode(const std::string &kernelname) {
  return !isKernelGenerated(kernelname);
}

//===----------------------------------------------------------------------===//

namespace cudaq {
void set_target_backend(const char *backend) {
  std::string backendName(backend);
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendName);
}

KernelArgsCreator getArgsCreator(const std::string &kernelName) {
  return argsCreators[kernelName];
}

std::string get_quake_by_name(const std::string &kernelName,
                              bool throwException) {
  // A prefix name has a '.' before the C++ mangled name suffix.
  auto kernelNamePrefix = kernelName + '.';

  // Find the quake code
  std::optional<std::string> result;
  for (auto [k, v] : quakeRegistry) {
    if (k == kernelName) {
      // Exact match. Return the code.
      return v;
    }
    if (k.starts_with(kernelNamePrefix)) {
      // Prefix match. Record it and make sure that it is a unique prefix.
      if (result.has_value()) {
        if (throwException)
          throw std::runtime_error("Quake code for '" + kernelName +
                                   "' has multiple matches.\n");
      } else {
        result = v;
      }
    }
  }
  if (result.has_value())
    return *result;
  auto msg = "Quake code not found for '" + kernelName + "'.\n";
  if (throwException)
    throw std::runtime_error(msg);
  return {};
}

std::string get_quake_by_name(const std::string &kernelName) {
  return get_quake_by_name(kernelName, true);
}

bool kernelHasConditionalFeedback(const std::string &kernelName) {
  auto quakeCode = get_quake_by_name(kernelName, false);
  return !quakeCode.empty() &&
         quakeCode.find("qubitMeasurementFeedback = true") != std::string::npos;
}

void set_shots(const std::size_t nShots) {
  auto &platform = cudaq::get_platform();
  platform.set_shots(nShots);
}
void clear_shots(const std::size_t nShots) {
  auto &platform = cudaq::get_platform();
  platform.clear_shots();
}

void set_noise(cudaq::noise_model &model) {
  auto &platform = cudaq::get_platform();
  platform.set_noise(&model);
}

void unset_noise() {
  auto &platform = cudaq::get_platform();
  platform.set_noise(nullptr);
}
} // namespace cudaq

namespace cudaq::support {
extern "C" {
void __nvqpp_initializer_list_to_vector_bool(std::vector<bool> &result,
                                             char *initList, std::size_t size) {
  // result is a sret return value. Make sure it is default initialized. Takes
  // advantage of default empty vector being all 0s.
  std::memset(&result, 0, sizeof(result));
  // Allocate space.
  result.reserve(size);
  // Copy in the initialization list data.
  char *p = initList;
  for (std::size_t i = 0; i < size; ++i, ++p)
    result.push_back(static_cast<bool>(*p));
  // Free the initialization list, which was stack allocated.
  free(initList);
}
}
} // namespace cudaq::support
