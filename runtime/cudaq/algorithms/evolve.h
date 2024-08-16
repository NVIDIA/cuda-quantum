/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EvolveResult.h"
#include "common/KernelWrapper.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/host_config.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"

namespace cudaq {

/// @brief Return type for asynchronous `evolve_async`.
using async_evolve_result = std::future<evolve_result>;

template <typename QuantumKernel>
evolve_result evolve(QuantumKernel &&kernel, std::vector<spin_op> observables = {}) {
  state final_state = get_state(std::forward<QuantumKernel>(kernel));
  if (observables.size() == 0) return evolve_result(final_state);

  auto prepare_state = [final_state]() { auto qs = qvector<2>(final_state); };
  std::vector<observe_result> final_expectations;
  for (auto observable : observables) {
    final_expectations.push_back(observe(prepare_state, observable));
  }
  return evolve_result(final_state, final_expectations);
}

template <typename QuantumKernel>
async_evolve_result evolve_async(QuantumKernel &&kernel, std::vector<spin_op> observables = {}, std::size_t qpu_id = 0) {
  auto &platform = cudaq::get_platform();
  std::promise<evolve_result> promise;
  auto f = promise.get_future();

  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), 
       func = std::forward<QuantumKernel>(kernel), 
       observables]() mutable {
        p.set_value(evolve(func, observables));
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return f;
}

} // namespace cudaq
