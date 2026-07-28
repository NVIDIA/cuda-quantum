/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/dem/policy.h"
#include "cudaq/algorithms/msm/policy.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/run/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/ptsbe/policy.h"

namespace cudaq {

/// @brief Fallback policy tag used when no specific policy matches.
struct other_policies {};

template <typename Policy>
std::string get_policy_name(const Policy &policy) {
  if constexpr (requires { policy.inner; }) {
    return std::string("async_") + policy.inner.name;
  } else {
    return Policy::name;
  }
}

} // namespace cudaq
