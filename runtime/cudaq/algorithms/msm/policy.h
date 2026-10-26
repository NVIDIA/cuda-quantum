/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SampleResult.h"
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nvqir {
class CircuitSimulator;
}

namespace cudaq {

class ExecutionManager;
class ExecutionContext;
class noise_model;

using msm_dimensions = std::pair<std::size_t, std::size_t>;

struct msm_result {
  sample_result samples;
  std::vector<double> probabilities;
  std::vector<std::size_t> probability_error_ids;
};

/// @brief Tag and inputs for computing Measurement Syndrome Matrix dimensions.
struct msm_size_policy {
  /// @brief The name of the policy.
  static constexpr char name[] = "msm_size";

  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = msm_dimensions;

  /// @brief The name of the kernel being executed.
  std::string kernelName;

  mutable const noise_model *noiseModel = nullptr;

  friend msm_dimensions
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const msm_size_policy &policy,
                                  ExecutionContext &ctx);
  friend msm_dimensions
  finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                   const msm_size_policy &policy);
};

/// @brief Tag and inputs for computing a Measurement Syndrome Matrix.
struct msm_policy {
  /// @brief The name of the policy.
  static constexpr char name[] = "msm";

  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = msm_result;

  /// @brief The name of the kernel being executed.
  std::string kernelName;

  std::optional<msm_dimensions> dimensions;

  mutable const noise_model *noiseModel = nullptr;

  friend msm_result finalize_execution_manager_impl(ExecutionManager &mgr,
                                                    const msm_policy &policy,
                                                    ExecutionContext &ctx);
  friend msm_result
  finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                   const msm_policy &policy);
};

} // namespace cudaq
