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
#include <functional>
#include <string>
#include <utility>

namespace cudaq {
class noise_model;
} // namespace cudaq

namespace cudaq::detail {

/// @brief Type-erased core of `dem_from_kernel`.
std::string runDemFromKernel(const std::string &kernelName,
                             cudaq::quantum_platform &platform,
                             const cudaq::noise_model *noise,
                             const std::function<void()> &wrappedKernel,
                             const std::string &plugin_name = "stim");

} // namespace cudaq::detail

namespace cudaq {

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
  return detail::runDemFromKernel(kernelName, platform, noise, [&]() mutable {
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
