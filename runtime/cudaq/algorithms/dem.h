/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include <concepts>
#include <cstddef>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq {

class noise_model;

/// @brief Sparse binary matrix mapping detectors (rows) to measurements
/// (columns). Returned alongside `M2OSparseMatrix` when
/// `return_measurement_matrices=True` is passed to `dem_from_kernel` (Python),
/// or via the `m2d_out` / `m2o_out` reference overloads (C++).
///
/// `rows[d]` lists the chronological measurement indices that contribute to
/// detector `d` (i.e. are XOR-ed together to form its syndrome bit).
/// `num_measurements` gives the total column count (shape is
/// `rows.size() × num_measurements`).
struct M2DSparseMatrix {
  std::size_t num_measurements = 0;
  std::vector<std::vector<std::size_t>> rows;
};

/// @brief Sparse binary matrix mapping observables (rows) to measurements
/// (columns). Returned alongside `M2DSparseMatrix` when
/// `return_measurement_matrices=True` is passed to `dem_from_kernel` (Python),
/// or via the `m2d_out` / `m2o_out` reference overloads (C++).
///
/// `rows[k]` lists the chronological measurement indices that contribute to
/// observable `k`. `num_measurements` gives the total column count (shape is
/// `rows.size() × num_measurements`).
struct M2OSparseMatrix {
  std::size_t num_measurements = 0;
  std::vector<std::vector<std::size_t>> rows;
};

} // namespace cudaq

namespace cudaq::detail {

/// @brief Type-erased core of `dem_from_kernel`.
///
/// @param m2d_out  Optional output for the m2d matrix.
/// @param m2o_out  Optional output for the m2o matrix.
///                 Pass `nullptr` (default) to skip either computation.
///                 Both are computed in a single circuit pass; requesting
///                 one automatically computes the other.
std::string runDemFromKernel(const std::string &kernelName,
                             cudaq::quantum_platform &platform,
                             const cudaq::noise_model *noise,
                             const std::function<void()> &wrappedKernel,
                             const cudaq::dem_options &options = {},
                             const std::string &plugin_name = "stim",
                             cudaq::M2DSparseMatrix *m2d_out = nullptr,
                             cudaq::M2OSparseMatrix *m2o_out = nullptr);

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
/// @param options Options forwarded to the Stim error analyzer (see
///                `cudaq::dem_options`).
/// @param args    Arguments forwarded to the kernel invocation.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel,
                            const cudaq::noise_model *noise,
                            const cudaq::dem_options &options, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return detail::runDemFromKernel(
      kernelName, platform, noise,
      [&]() mutable { kernel(std::forward<Args>(args)...); }, options);
}

/// @brief Convenience overload: noise model with default options, forwarding
/// @p args to the kernel.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel,
                            const cudaq::noise_model *noise, Args &&...args) {
  return dem_from_kernel(std::forward<QuantumKernel>(kernel), noise,
                         /*options=*/cudaq::dem_options{},
                         std::forward<Args>(args)...);
}

/// @brief Convenience overload for the no-noise case.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel, Args &&...args) {
  return dem_from_kernel(std::forward<QuantumKernel>(kernel),
                         /*noise=*/nullptr, /*options=*/cudaq::dem_options{},
                         std::forward<Args>(args)...);
}

/// @brief Overload that also returns the m2d and m2o sparse matrices,
/// with full control over analyzer options.
///
/// @param m2d_out  Populated with the measurements-to-detectors matrix.
/// @param m2o_out  Populated with the measurements-to-observables matrix.
///                 Both are filled in a single circuit pass.
/// @param options  Options forwarded to the Stim error analyzer.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel,
                            const cudaq::noise_model *noise,
                            const cudaq::dem_options &options,
                            cudaq::M2DSparseMatrix &m2d_out,
                            cudaq::M2OSparseMatrix &m2o_out, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return detail::runDemFromKernel(
      kernelName, platform, noise,
      [&]() mutable { kernel(std::forward<Args>(args)...); }, options,
      /*plugin_name=*/"stim", &m2d_out, &m2o_out);
}

/// @brief Convenience overload: m2d/m2o outputs with noise, default options.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel,
                            const cudaq::noise_model *noise,
                            cudaq::M2DSparseMatrix &m2d_out,
                            cudaq::M2OSparseMatrix &m2o_out, Args &&...args) {
  return dem_from_kernel(std::forward<QuantumKernel>(kernel), noise,
                         /*options=*/cudaq::dem_options{}, m2d_out, m2o_out,
                         std::forward<Args>(args)...);
}

/// @brief Convenience overload: m2d/m2o outputs, no noise, default options.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
std::string dem_from_kernel(QuantumKernel &&kernel,
                            cudaq::M2DSparseMatrix &m2d_out,
                            cudaq::M2OSparseMatrix &m2o_out, Args &&...args) {
  return dem_from_kernel(std::forward<QuantumKernel>(kernel),
                         /*noise=*/nullptr, /*options=*/cudaq::dem_options{},
                         m2d_out, m2o_out, std::forward<Args>(args)...);
}

} // namespace cudaq
