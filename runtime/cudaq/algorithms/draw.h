/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
