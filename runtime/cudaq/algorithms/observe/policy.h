/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Future.h"
#include "common/KernelExecution.h"
#include "common/ObserveResult.h"
#include "cudaq/algorithms/ExecutionMetadataPolicy.h"
#include "cudaq/algorithms/observe/options.h"
#include "cudaq/operators.h"

namespace cudaq {

class ExecutionManager;
class ExecutionContext;

/// @brief Tag and options for computing expectation values.
struct observe_policy : ExecutionMetadataPolicy {
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

  mutable const noise_model *noiseModel = nullptr;

  mutable bool canHandleObserve = false;

  friend observe_result
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const observe_policy &policy,
                                  ExecutionContext &ctx);
};

using async_observe_policy = async_policy_wrapper<observe_policy>;

} // namespace cudaq
