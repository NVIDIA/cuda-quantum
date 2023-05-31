/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/MeasureCounts.h"
#include "cudaq/concepts.h"
#include "cudaq/platform.h"

namespace cudaq {

/// @brief Given any CUDA Quantum kernel and its associated runtime arguments,
/// return the resources that this kernel will use. This does not execute the
/// circuit simulation, it only traces the quantum operation calls and returns
/// a `resources` type that allows the programmer to query the number and types
/// of operations in the kernel.
template <typename QuantumKernel, typename... Args>
auto estimate_resources(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext context("tracer");
  auto &platform = get_platform();
  platform.set_exec_ctx(&context);
  kernel(args...);
  platform.reset_exec_ctx();
  return context.kernelResources;
}

} // namespace cudaq
