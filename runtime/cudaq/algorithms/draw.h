/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "cudaq/platform.h"

namespace cudaq {

namespace __internal__ {

std::string draw(const Trace &trace);

}

namespace details {

/// @brief Execute the given kernel functor and extract the
/// state representation.
template <typename KernelFunctor>
std::string extractTrace(KernelFunctor &&kernel) {
  // Get the platform.
  auto &platform = cudaq::get_platform();

  // This can only be done in simulation
  if (!platform.is_simulator())
    throw std::runtime_error("Cannot use draw on a physical QPU.");

  // Create an execution context, indicate this is for tracing the execution
  // path
  ExecutionContext context("tracer");

  // Perform the usual pattern set the context, execute and then reset
  platform.set_exec_ctx(&context);
  kernel();
  platform.reset_exec_ctx();

  return __internal__::draw(context.kernelTrace);
}
// FIXME: Implement `runDrawAsync`?
} // namespace details

/// @brief Returns a drawing of the execution path, i.e., the trace, of the
/// kernel. The drawing is a UTF-8 encoded string.
template <typename QuantumKernel, typename... Args>
std::string draw(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext context("tracer");
  auto &platform = get_platform();
  platform.set_exec_ctx(&context);
  kernel(args...);
  platform.reset_exec_ctx();
  return __internal__::draw(context.kernelTrace);
}

/// @brief Outputs the drawing of a circuit to an output stream.
template <typename QuantumKernel, typename... Args>
void draw(std::ostream &os, QuantumKernel &&kernel, Args &&...args) {
  auto drawing = draw(kernel, std::forward<Args>(args)...);
  os << drawing;
}

} // namespace cudaq