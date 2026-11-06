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
#include "common/ObserveResult.h"
#include "cudaq/algorithms/observe/options.h"
#include "cudaq/operators.h"

namespace cudaq {

class ExecutionManager;
class ExecutionContext;

/// @brief Tag and options for computing expectation values.
struct observe_policy {
  /// @brief The name of the policy.
  static constexpr char name[] = "observe";

  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = observe_result;

  /// Observe options.
  observe_options options;

  /// @brief The name of the kernel being executed.
  std::string kernelName;

  /// @brief The spin operator being observed.
  spin_op spin;

  const noise_model *noiseModel = nullptr;

  /// @brief For a broadcast over a set of arguments, the index of this launch
  /// and the total number of launches. Lets the simulator reuse its state
  /// between launches. Both are 0 outside of a broadcast.
  std::size_t batchIteration = 0;
  std::size_t totalIterations = 0;

  friend observe_result
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const observe_policy &policy,
                                  ExecutionContext &ctx);

  friend CompileOptions get_compile_options_impl(const observe_policy &);
};

using async_observe_policy = async_policy_wrapper<observe_policy>;

} // namespace cudaq
