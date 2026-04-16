/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Future.h"
#include "common/ObserveResult.h"
#include "cudaq/algorithms/any_policy.h"
#include "cudaq/algorithms/observe/options.h"

namespace cudaq {

/// @brief Tag and options for computing expectation values.
struct observe_policy : public any_policy {
  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = observe_result;

  /// Observe options.
  observe_options options;
};

namespace async {
/// @brief Tag and options for asynchronous statistical sampling.
///
struct observe_policy : public any_policy {
  /// Associated result type for asynchronous APIs keyed off this policy.
  using result_type = async_result<observe_result>;

  /// Observe options.
  observe_options options;
};
} // namespace async
} // namespace cudaq
