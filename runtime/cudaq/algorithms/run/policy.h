/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Future.h"
#include <cstddef>
#include <string>

namespace nvqir {
class CircuitSimulator;
}

namespace cudaq {

class ExecutionManager;
class ExecutionContext;
class noise_model;

/// @brief Raw output produced by a runnable kernel.
struct run_result {
  std::string outputLog;
};

/// @brief Tag and inputs for executing a runnable kernel.
struct run_policy {
  /// @brief The name of the policy.
  static constexpr char name[] = "run";

  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = run_result;

  /// @brief The name of the kernel being executed.
  std::string kernelName;

  /// @brief The number of kernel executions requested.
  std::size_t shots = 1;

  mutable const noise_model *noiseModel = nullptr;

  friend run_result finalize_execution_manager_impl(ExecutionManager &mgr,
                                                    const run_policy &policy,
                                                    ExecutionContext &ctx);
  friend run_result
  finalize_simulation_circuit_impl(nvqir::CircuitSimulator &sim,
                                   const run_policy &policy);
};

using async_run_policy = async_policy_wrapper<run_policy>;

} // namespace cudaq
