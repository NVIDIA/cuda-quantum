/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ObserveResult.h"
#include "cudaq/algorithms/observe/options.h"

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

  friend observe_result
  finalize_execution_manager_impl(ExecutionManager &mgr,
                                  const observe_policy &policy,
                                  ExecutionContext &ctx);
};

} // namespace cudaq
