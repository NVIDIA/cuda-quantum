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

namespace cudaq::orca {

struct sample_policy {
  static constexpr char name[] = "orca-sample";
  static constexpr char kernelName[] = "orca_launch";
  using result_type = cudaq::sample_result;
};

using async_sample_policy = cudaq::async_policy_wrapper<sample_policy>;

} // namespace cudaq::orca
