/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/Tuple.h"
#include "cudaq/algorithms/dem/policy.h"
#include "cudaq/algorithms/msm/policy.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/run/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/ptsbe/policy.h"
#include <tuple>

namespace cudaq {

/// @brief Fallback policy tag used when no specific policy matches.
struct other_policies {};

/// @brief List of all existing launch policies.
using all_policies =
    std::tuple<sample_policy, async_sample_policy, observe_policy,
               async_observe_policy, run_policy, async_run_policy,
               msm_size_policy, msm_policy, dem_policy, ptsbe::sample_policy>;

/// @brief Concept satisfied by any type registered in @c all_policies.
template <typename Policy>
concept launch_policy = detail::is_in_tuple_v<Policy, all_policies>;

template <launch_policy Policy>
std::string get_policy_name(const Policy &policy) {
  if constexpr (requires { policy.inner; }) {
    return std::string("async_") + policy.inner.name;
  } else {
    return Policy::name;
  }
}

} // namespace cudaq
