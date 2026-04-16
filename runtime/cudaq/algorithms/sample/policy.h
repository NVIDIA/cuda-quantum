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
#include "cudaq/algorithms/any_policy.h"
#include "cudaq/algorithms/sample/options.h"

namespace cudaq {

/// @brief Tag and options for sampling quantum circuit measurements.
struct sample_policy : public any_policy {
  /// Associated result type for synchronous APIs keyed off this policy.
  using result_type = sample_result;

  /// Sampling  options.
  sample_options options;
};

namespace async {
/// @brief Tag and options for asynchronous statistical sampling.
///
struct sample_policy : public any_policy {
  /// Associated result type for asynchronous APIs keyed off this policy.
  using result_type = async_result<sample_result>;

  /// Sampling  options.
  sample_options options;
};
} // namespace async
} // namespace cudaq
