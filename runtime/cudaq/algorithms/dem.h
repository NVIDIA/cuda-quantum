/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/algorithms/dem/options.h"
#include "cudaq/algorithms/dem/policy.h"
#include "cudaq/algorithms/dem/result.h"
#include "cudaq/algorithms/launch.h"
#include "cudaq/platform.h"
#include <concepts>
#include <functional>
#include <string>
#include <utility>

namespace cudaq {
class noise_model;
}

namespace cudaq::detail {

using dem_policy_launcher = std::function<cudaq::dem_result(
    const cudaq::dem_policy &, cudaq::ExecutionContext &)>;

/// @brief Shared DEM core: conditional-feedback check + Stim analysis scope,
/// then delegates to @p launchPolicy. Reached through `launchDem`, the shared
/// helper behind both DEM entry points (C++ `runDemFromKernel`, Python
/// `launch_dem`).
cudaq::dem_result launchDemPolicy(const cudaq::dem_policy &policy,
                                  cudaq::ExecutionContext &ctx,
                                  const dem_policy_launcher &launchPolicy,
                                  const std::string &plugin_name = "stim");

/// @brief Shared launch core for C++ and Python DEM entry points: builds the
/// execution context, then runs the policy through `launchDemPolicy` +
/// `detail::launch`. `detail::launch` stays inline so JIT compiler references
/// remain in the caller TU.
inline cudaq::dem_result launchDem(const cudaq::dem_policy &policy,
                                   cudaq::quantum_platform &platform,
                                   const std::function<void()> &kernel,
                                   const std::string &plugin_name = "stim") {
  cudaq::ExecutionContext ctx(policy.name);
  ctx.qpuId = cudaq::getCurrentQpuId();
  // Mirror the noise model onto the `ExecutionContext`: the local kernel-launch
  // path reconfigures the simulator from the context, which would otherwise
  // overwrite the policy-derived model with null and silence in-kernel
  // `apply_noise` (dropping every error mechanism from the DEM).
  ctx.noiseModel = policy.noiseModel;
  return launchDemPolicy(
      policy, ctx,
      [&](const cudaq::dem_policy &activePolicy,
          cudaq::ExecutionContext &activeCtx) {
        return cudaq::detail::launch(activePolicy, activeCtx.qpuId, activeCtx,
                                     platform, kernel);
      },
      plugin_name);
}

inline std::string runDemFromKernel(const std::string &kernelName,
                                    cudaq::quantum_platform &platform,
                                    const cudaq::noise_model *noise,
                                    const std::function<void()> &wrappedKernel,
                                    const cudaq::dem_options &options = {},
                                    const std::string &plugin_name = "stim",
                                    cudaq::M2DSparseMatrix *m2d_out = nullptr,
                                    cudaq::M2OSparseMatrix *m2o_out = nullptr) {
  cudaq::dem_policy policy;
  policy.kernelName = kernelName;
  policy.noiseModel = noise;
  policy.options = options;
  // Pointer existence is authoritative: non-null enables matrix computation
  // even if the flag is false; null suppresses it even if the flag is set.
  policy.options.return_measurement_matrices = m2d_out || m2o_out;

  auto result = launchDem(policy, platform, wrappedKernel, plugin_name);

  if (m2d_out)
    *m2d_out = std::move(result.m2d);
  if (m2o_out)
    *m2o_out = std::move(result.m2o);
  return std::move(result.dem);
}

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
