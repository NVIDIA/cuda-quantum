/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/Resources.h"
#include "cudaq/platform.h"

namespace nvqir {
void switchToResourceCounterSimulator();
void stopUsingResourceCounterSimulator();
void setChoiceFunction(std::function<bool()> choice);
cudaq::Resources *getResourceCounts();
} // namespace nvqir

namespace cudaq {
namespace details {

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the resource estimation
/// process.
template <typename KernelFunctor>
Resources run_estimate_resources(KernelFunctor &&wrappedKernel,
                                 quantum_platform &platform,
                                 const std::string &kernelName,
                                 std::function<bool()> choice) {
  // Create the execution context.
  auto ctx = std::make_unique<ExecutionContext>("resource-count", 1);
  ctx->kernelName = kernelName;

  // Indicate that this is not an async exec
  ctx->asyncExec = false;

  // Use the resource counter simulator
  nvqir::switchToResourceCounterSimulator();
  // Set the choice function for the simulator
  nvqir::setChoiceFunction(choice);

  // Set the platform
  platform.set_exec_ctx(ctx.get());

  wrappedKernel();

  platform.reset_exec_ctx();

  // Save and clone counts data
  auto counts = Resources(*nvqir::getResourceCounts());
  // Switch simulators back
  nvqir::stopUsingResourceCounterSimulator();

  return counts;
}
} // namespace details

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
Resources estimate_resources(QuantumKernel &&kernel, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  auto seed = cudaq::get_random_seed();
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> rand(0, 1);
  auto choice = [&]() { return rand(gen); };
  return details::run_estimate_resources(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, choice);
}

/// @brief Given any CUDA-Q kernel and its associated runtime arguments,
/// return the resources that this kernel will use. This does not execute the
/// circuit simulation, it only traces the quantum operation calls and returns
/// a `Resources` type that allows the programmer to query the number and
/// types of operations in the kernel.
///
/// @param choice A function called to determine the result of measurements,
///               used to determine which path is taken when the kernel has
///               branches on mid-circuit measurement results. Invoking the
///               kernel from inside this function is forbidden.
template <typename QuantumKernel, typename... Args>
  requires std::invocable<QuantumKernel &, Args...>
Resources estimate_resources(std::function<bool()> choice,
                             QuantumKernel &&kernel, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return details::run_estimate_resources(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, choice);
}

} // namespace cudaq
