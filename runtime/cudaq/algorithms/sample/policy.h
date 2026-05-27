/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

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

  /// @brief Flag to indicate that a warning about named measurement registers
  /// in sampling context has already been emitted.
  bool warnedNamedMeasurements = false;

  /// @brief A vector containing information about how to reorder the global
  /// register after execution. Empty means no reordering.
  mutable std::vector<std::size_t> reorderIdx;

  mutable const noise_model *noiseModel = nullptr;

  friend sample_result
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const sample_policy &policy);
  friend sample_result
  finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                   const sample_policy &policy);
};

using async_sample_policy = async_policy_wrapper<sample_policy>;

} // namespace cudaq
