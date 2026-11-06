/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CompileOptions.h"
#include "common/Future.h"
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
  /// @brief The name of the policy.
  static constexpr char name[] = "sample";

  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = sample_result;

  /// Sampling  options.
  sample_options options;

  /// @brief The name of the kernel being executed.
  std::string kernelName;

  const noise_model *noiseModel = nullptr;

  /// @brief For a broadcast over a set of arguments, the index of this launch
  /// and the total number of launches. Lets the simulator reuse its state
  /// between launches. Both are 0 outside of a broadcast.
  std::size_t batchIteration = 0;
  std::size_t totalIterations = 0;

  friend sample_result
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const sample_policy &policy);
  friend sample_result
  finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                   const sample_policy &policy);

  friend CompileOptions get_compile_options_impl(const sample_policy &);
};

using async_sample_policy = async_policy_wrapper<sample_policy>;

} // namespace cudaq
