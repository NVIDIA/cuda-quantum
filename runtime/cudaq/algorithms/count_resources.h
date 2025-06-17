/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/KernelWrapper.h"
#include "common/ResourceCounts.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/concepts.h"
#include "cudaq/host_config.h"

namespace nvqir {
void switchToResourceCounterSimulator();
void stopUsingResourceCounterSimulator();
void setChoiceFunction(std::function<bool()> choice);
cudaq::resource_counts *getResourceCounts();
} // namespace nvqir

namespace cudaq {
bool kernelHasConditionalFeedback(const std::string &);
namespace __internal__ {
bool isKernelGenerated(const std::string &);
}

/// @brief Performs resource counting on the given quantum kernel
/// expression and returns an accounting of how many times each gate
/// was applied, in addition to the total number of gates and qubits used.
///
/// @param choice A choice function called to determine the outcome of
///               measurements, in case control flow depends on measurements.
/// @param kernel The kernel expression, must contain final measurements.
/// @param args The variadic concrete arguments for evaluation of the kernel.
/// @returns The resource_counts object with the accounting of resource usage.
///
/// @details Given a quantum kernel and choice function, counts the number of
///          resources used along the control flow path determined by the
///          choice function.
#if CUDAQ_USE_STD20
template <typename QuantumKernel, typename... Args>
  requires SampleCallValid<QuantumKernel, Args...>
#else
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_r_v<void, QuantumKernel, Args...>>>
#endif
auto count_resources(std::function<bool()> choice, QuantumKernel &&kernel,
                     Args &&...args) {
  // Need the code to be lowered to llvm and the kernel to be registered
  // so that we can check for conditional feedback / mid circ measurement
  if constexpr (has_name<QuantumKernel>::value) {
    static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
  }

  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  auto hasConditionalFeedback = cudaq::kernelHasConditionalFeedback(kernelName);
  // Create the execution context.
  auto ctx = std::make_unique<ExecutionContext>("resource-count", 1);
  ctx->kernelName = kernelName;
  ctx->hasConditionalsOnMeasureResults = hasConditionalFeedback;

  // Indicate that this is not an async exec
  ctx->asyncExec = false;

  // Use the resource counter simulator
  nvqir::switchToResourceCounterSimulator();
  // Set the choice function for the simulator
  nvqir::setChoiceFunction(choice);

  // Set the platform
  platform.set_exec_ctx(ctx.get());

  kernel(std::forward<Args>(args)...);

  // Save and clone counts data
  auto counts = resource_counts(*nvqir::getResourceCounts());
  // Switch simulators back
  nvqir::stopUsingResourceCounterSimulator();

  return counts;
}
} // namespace cudaq
