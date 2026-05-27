/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform.h"
#include <concepts>
#include <dlfcn.h>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq {
class noise_model;
} // namespace cudaq

namespace cudaq {

namespace details {

/// @brief `extern "C"` entry point for DEM analysis, defined in
/// `libcudaq-analysis.so`. Resolved at first call via `dlopen` + `dlsym` so
/// that consumers who never invoke `dem_from_kernel` pay zero startup cost
/// (no eager `libstim.so` load, no transitive plugin pulls).
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif
extern "C" std::string cudaq_runDemFromKernel(
    const std::string &kernelName, cudaq::quantum_platform &platform,
    const cudaq::noise_model *noise, const std::function<void()> &wrappedKernel,
    const std::string &plugin_name);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

/// @brief Type-erased core of `dem_from_kernel`. Header-only inline body
/// `dlopen`s `libcudaq-analysis.so` on demand and forwards to
/// `cudaq_runDemFromKernel`. The shared library is resolved once per
/// process (function-local static cache).
inline std::string runDemFromKernel(const std::string &kernelName,
                                    cudaq::quantum_platform &platform,
                                    const cudaq::noise_model *noise,
                                    const std::function<void()> &wrappedKernel,
                                    std::string plugin_name = "stim") {
  using FnT = decltype(&cudaq_runDemFromKernel);
  static FnT fn = []() -> FnT {
#if defined(__APPLE__)
    constexpr const char *kLib = "libcudaq-analysis.dylib";
#else
    constexpr const char *kLib = "libcudaq-analysis.so";
#endif
    constexpr const char *kSym = "cudaq_runDemFromKernel";
    void *handle = dlopen(kLib, RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
      const char *err = dlerror();
      throw std::runtime_error(
          std::string("`cudaq::dem_from_kernel`: failed to load '") + kLib +
          "' (does this CUDA-Q distribution include the DEM analysis "
          "engine?): " +
          (err ? err : "unknown dlerror"));
    }
    void *sym = dlsym(handle, kSym);
    if (!sym) {
      const char *err = dlerror();
      throw std::runtime_error(
          std::string("`cudaq::dem_from_kernel`: '") + kLib +
          "' is missing the entry point '" + kSym +
          "' (CUDA-Q version mismatch?): " + (err ? err : "unknown dlerror"));
    }
    return reinterpret_cast<FnT>(sym);
  }();
  return fn(kernelName, platform, noise, wrappedKernel, plugin_name);
}

} // namespace details

/// @brief Run DEM (Detector Error Model) analysis over a CUDA-Q kernel and
/// return the resulting model as a UTF-8 string in Stim's standard
/// [`.dem` file
/// format](https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem_detector_error_model.md).
///
///
/// @note The active CUDA-Q target / platform is not modified; the analysis
///       simulator is purely an internal override.
///
/// @param kernel  Any callable that can be invoked with @p args (CUDA-Q kernel,
///                lambda, or kernel-builder).
/// @param noise   Optional noise model layered on per `cudaq::noise_model`
///                semantics.
/// @param args    Arguments forwarded to the kernel invocation.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel,
                            const cudaq::noise_model *noise, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return details::runDemFromKernel(kernelName, platform, noise, [&]() mutable {
    kernel(std::forward<Args>(args)...);
  });
}

/// @brief Convenience overload for the no-noise case.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel, Args &&...args) {
  return dem_from_kernel(std::forward<QuantumKernel>(kernel),
                         /*noise=*/nullptr, std::forward<Args>(args)...);
}

} // namespace cudaq
