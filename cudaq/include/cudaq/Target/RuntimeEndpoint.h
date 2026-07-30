
/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/KernelArgs.h"
#include "cudaq/algorithms/policies.h"
#include <any>
#include <stdexcept>
#include <string>

namespace cudaq {

class QPU;
class CompiledModule;

namespace detail {

// The compile-time map from policy type to function pointer in RuntimeEndpoint.
template <typename Policy>
struct runtime_endpoint_fn {
  static_assert(sizeof(Policy) == 0, "Unsupported policy");
};

template <typename Policy>
using launch_fn_type = Policy::result_type (*)(std::any &impl,
                                               const Policy &policy,
                                               const CompiledModule &module,
                                               KernelArgs args);

} // namespace detail

struct RuntimeEndpoint {
  ///////////////////////////
  /// Function pointers for all supported launch policies.
  ///////////////////////////

  detail::launch_fn_type<sample_policy> sample = nullptr;
  detail::launch_fn_type<async_sample_policy> async_sample = nullptr;
  detail::launch_fn_type<observe_policy> observe = nullptr;
  detail::launch_fn_type<async_observe_policy> async_observe = nullptr;
  detail::launch_fn_type<run_policy> run = nullptr;
  detail::launch_fn_type<async_run_policy> async_run = nullptr;
  detail::launch_fn_type<msm_size_policy> msm_size = nullptr;
  detail::launch_fn_type<msm_policy> msm = nullptr;
  detail::launch_fn_type<dem_policy> dem = nullptr;
  detail::launch_fn_type<ptsbe::sample_policy> ptsbe_sample = nullptr;

  /// Store any RuntimeEndpoint state here. Passed by mutable reference to each
  /// launch invocation.
  std::any impl;

  template <typename Policy>
  typename Policy::result_type launchKernel(const Policy &policy,
                                            const CompiledModule &module,
                                            KernelArgs args) {
    auto fn = this->*detail::runtime_endpoint_fn<Policy>::member;
    if (!fn) {
      throw std::runtime_error(std::string("Unsupported policy: '") +
                               get_policy_name(policy) + "'");
    }
    return fn(impl, policy, module, args);
  }

  /// Create a RuntimeEndpoint from a QPU instance.
  static RuntimeEndpoint wrapQPU(QPU &qpu);
};

namespace detail {
#define CUDAQ_RUNTIME_ENDPOINT_FN(Policy, field)                               \
  template <>                                                                  \
  struct runtime_endpoint_fn<Policy> {                                         \
    static constexpr auto member = &RuntimeEndpoint::field;                    \
  };

CUDAQ_RUNTIME_ENDPOINT_FN(sample_policy, sample)
CUDAQ_RUNTIME_ENDPOINT_FN(async_sample_policy, async_sample)
CUDAQ_RUNTIME_ENDPOINT_FN(observe_policy, observe)
CUDAQ_RUNTIME_ENDPOINT_FN(async_observe_policy, async_observe)
CUDAQ_RUNTIME_ENDPOINT_FN(run_policy, run)
CUDAQ_RUNTIME_ENDPOINT_FN(async_run_policy, async_run)
CUDAQ_RUNTIME_ENDPOINT_FN(msm_size_policy, msm_size)
CUDAQ_RUNTIME_ENDPOINT_FN(msm_policy, msm)
CUDAQ_RUNTIME_ENDPOINT_FN(dem_policy, dem)
CUDAQ_RUNTIME_ENDPOINT_FN(ptsbe::sample_policy, ptsbe_sample)

#undef CUDAQ_RUNTIME_ENDPOINT_FN
} // namespace detail

} // namespace cudaq
