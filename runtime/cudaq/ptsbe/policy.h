/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/NoiseModel.h"
#include "cudaq/ptsbe/PTSBEOptions.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"
#include <cstddef>
#include <string>
#include <vector>

namespace cudaq {

class ExecutionManager;
class ExecutionContext;

namespace ptsbe {

/// @brief Tag and options for PTSBE (pre-trajectory sampling with batch
/// execution) sampling.
struct sample_policy {
  /// @brief The name of the policy. Must match the ExecutionContext name used
  /// for PTSBE execution.
  static constexpr char name[] = "ptsbe-sample";

  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = cudaq::ptsbe::sample_result;

  /// PTSBE configuration (strategy, shot allocation, execution-data flags).
  PTSBEOptions options;

  /// @brief Total number of shots to allocate across trajectories.
  std::size_t shots = 0;

  /// @brief The name of the kernel being executed.
  std::string kernelName;

  /// @brief Noise model used to resolve trajectory noise channels.
  ///
  /// PTSBE applies noise as explicit trajectory unitaries resolved into the
  /// trace before execution.
  mutable const noise_model *noiseModel = nullptr;

  friend result_type
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const sample_policy &policy);
};

} // namespace ptsbe
} // namespace cudaq
