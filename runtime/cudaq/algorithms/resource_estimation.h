/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/Resources.h"
#include "cudaq/algorithms/estimate/policy.h"
#include "cudaq/algorithms/launch.h"
#include "cudaq/platform.h"

namespace cudaq {
namespace detail {

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the resource estimation
/// process.
template <typename KernelFunctor>
estimate_result run_estimate_resources(KernelFunctor &&wrappedKernel,
                                       quantum_platform &platform,
                                       const std::string &kernelName,
                                       std::function<bool()> choice) {
  estimate_policy policy{.kernelName = kernelName, .choice = std::move(choice)};

  // Create the execution context.
  ExecutionContext ctx(estimate_policy::name, 1);
  ctx.kernelName = kernelName;

  // Indicate that this is not an async exec
  ctx.asyncExec = false;

  CUDAQ_INFO("Launching kernel with estimate policy");
  return detail::launch(policy, /*qpu_id=*/0, ctx, platform,
                        std::forward<KernelFunctor>(wrappedKernel));
}
} // namespace detail

////////////////////// Note //////////////////////////////////////////////////
// We currently have two redundant APIs for resource estimation that only
// differ in the return type:
//  - the older `resource_estimation()` functions return `Resources` types
//  directly,
//  - the newer `estimate()` functions return `estimate_result` types.
//
// The goal is to deprecate the `resource_estimation()` functions and migrate
// callers to the `estimate()` functions. However, the shape of
// `estimate_result` has not settled yet, so we keep both APIs for now and will
// make a clean break once we are ready.
//////////////////////////////////////////////////////////////////////////////

/// @brief Given any CUDA-Q kernel and its associated runtime arguments,
/// return the resources that this kernel will use. This does not execute the
/// circuit simulation, it only traces the quantum operation calls and returns
/// a `Resources` type that allows the programmer to query the number and
/// types of operations in the kernel. By default, any measurement will return
/// `true` or `false` with 50% probability. To estimate resources for specific
/// paths based on measurements, supply a choice function to the overloaded
/// version of this function.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
estimate_result estimate(QuantumKernel &&kernel, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  auto seed = cudaq::get_random_seed();
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> rand(0, 1);
  auto choice = [&]() { return rand(gen); };
  return detail::run_estimate_resources(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, choice);
}

/// @brief Given any CUDA-Q kernel and its associated runtime arguments,
/// return the resources that this kernel will use. This does not execute the
/// circuit simulation, it only traces the quantum operation calls and returns
/// a `estimate_result` type that allows the programmer to query the number and
/// types of operations in the kernel.
///
/// @param choice A function called to determine the result of measurements,
///               used to determine which path is taken when the kernel has
///               branches on mid-circuit measurement results. Invoking the
///               kernel from inside this function is forbidden.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
estimate_result estimate(std::function<bool()> choice, QuantumKernel &&kernel,
                         Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return detail::run_estimate_resources(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, choice);
}

/// @brief Given any CUDA-Q kernel and its associated runtime arguments,
/// return the resources that this kernel will use. This does not execute the
/// circuit simulation, it only traces the quantum operation calls and returns
/// a `Resources` type that allows the programmer to query the number and
/// types of operations in the kernel. By default, any measurement will return
/// `true` or `false` with 50% probability. To estimate resources for specific
/// paths based on measurements, supply a choice function to the overloaded
/// version of this function.
///
/// Returns a `Resources` type that stores the number and types of operations
/// in the kernel. Note that this return type is the same as the `resources`
/// attribute of `cudaq::estimate_result`, returned by `cudaq::estimate()`.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
Resources estimate_resources(QuantumKernel &&kernel, Args &&...args) {
  return estimate(std::forward<QuantumKernel>(kernel),
                  std::forward<Args>(args)...)
      .get_resources();
}

/// @brief Given any CUDA-Q kernel and its associated runtime arguments,
/// return the resources that this kernel will use. This does not execute the
/// circuit simulation, it only traces the quantum operation calls and returns
/// a `estimate_result` type that allows the programmer to query the number and
/// types of operations in the kernel.
///
/// @param choice A function called to determine the result of measurements,
///               used to determine which path is taken when the kernel has
///               branches on mid-circuit measurement results. Invoking the
///               kernel from inside this function is forbidden.
///
/// Returns a `Resources` type that stores the number and types of operations
/// in the kernel. Note that this return type is the same as the `resources`
/// attribute of `cudaq::estimate_result`, returned by `cudaq::estimate()`.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
Resources estimate_resources(std::function<bool()> choice,
                             QuantumKernel &&kernel, Args &&...args) {
  return estimate(std::move(choice), std::forward<QuantumKernel>(kernel),
                  std::forward<Args>(args)...)
      .get_resources();
}

} // namespace cudaq
