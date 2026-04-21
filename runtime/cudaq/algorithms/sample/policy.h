/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include "cudaq/algorithms/sample/options.h"

namespace nvqir {
class CircuitSimulator;
}

namespace cudaq {

class ExecutionManager;
class ExecutionContext;

/// @brief Tag and options for sampling quantum circuit measurements.
struct sample_policy {
  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = sample_result;

  /// Sampling  options.
  sample_options options;

  friend sample_result
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const sample_policy &policy,
                                  ExecutionContext &ctx);
  friend sample_result
  finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                   const sample_policy &policy,
                                   ExecutionContext &ctx);
};

} // namespace cudaq
