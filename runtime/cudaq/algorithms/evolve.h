/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EvolveResult.h"
#include "common/KernelWrapper.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/base_integrator.h"
#include "cudaq/evolution.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/schedule.h"

namespace cudaq {

/// @brief Return type for asynchronous `evolve_async`.
using async_evolve_result = std::future<evolve_result>;

namespace __internal__ {
// Internal methods for evolve implementation on circuit simulators.

/// @brief Evolve from an initial state to the final state, no intermediate
/// states.
template <typename QuantumKernel>
evolve_result evolve(state initial_state, QuantumKernel &&kernel,
                     const std::vector<spin_op> &observables = {},
                     int shots_count = -1) {
  state final_state =
      get_state(std::forward<QuantumKernel>(kernel), initial_state);
  if (observables.size() == 0)
    return evolve_result(final_state);

  auto prepare_state = [final_state]() { auto qs = qvector<2>(final_state); };
  std::vector<observe_result> final_expectations;
  for (auto observable : observables) {
    shots_count <= 0
        ? final_expectations.push_back(observe(prepare_state, observable))
        : final_expectations.push_back(
              observe(shots_count, prepare_state, observable));
  }
  return evolve_result(final_state, final_expectations);
}

/// @brief Evolve from an initial state to the final state and gather
/// intermediate states.
// Step evolution is provided as `kernels`.
template <typename QuantumKernel>
evolve_result evolve(state initial_state, std::vector<QuantumKernel> kernels,
                     const std::vector<std::vector<spin_op>> &observables = {},
                     int shots_count = -1) {
  std::vector<state> intermediate_states = {};
  std::vector<std::vector<observe_result>> expectation_values = {};
  int step_idx = -1;
  for (auto kernel : kernels) {
    if (intermediate_states.size() == 0) {
      intermediate_states.push_back(get_state(kernel, initial_state));
    } else {
      intermediate_states.push_back(
          get_state(kernel, intermediate_states.back()));
    }
    if (observables.size() > 0) {
      std::vector<observe_result> expectations = {};
      auto prepare_state = [intermediate_states]() {
        auto qs = qvector<2>(intermediate_states.back());
      };
      for (auto observable : observables[++step_idx]) {
        shots_count <= 0
            ? expectations.push_back(observe(prepare_state, observable))
            : expectations.push_back(
                  observe(shots_count, prepare_state, observable));
      }
      expectation_values.push_back(expectations);
    }
  }
  if (step_idx < 0)
    return evolve_result(intermediate_states);
  return evolve_result(intermediate_states, expectation_values);
}

template <typename QuantumKernel>
async_evolve_result
evolve_async(state initial_state, QuantumKernel &&kernel,
             const std::vector<spin_op> &observables = {},
             std::size_t qpu_id = 0,
             std::optional<cudaq::noise_model> noise_model = std::nullopt,
             int shots_count = -1) {
  auto &platform = cudaq::get_platform();
  std::promise<evolve_result> promise;
  auto f = promise.get_future();

  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), func = std::forward<QuantumKernel>(kernel),
       initial_state, observables, noise_model, shots_count,
       &platform]() mutable {
        if (noise_model.has_value())
          platform.set_noise(&noise_model.value());
        p.set_value(evolve(initial_state, func, observables, shots_count));
        if (noise_model.has_value())
          platform.set_noise(nullptr);
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return f;
}

template <typename QuantumKernel>
async_evolve_result
evolve_async(state initial_state, std::vector<QuantumKernel> kernels,
             const std::vector<std::vector<spin_op>> &observables = {},
             std::size_t qpu_id = 0,
             std::optional<cudaq::noise_model> noise_model = std::nullopt,
             int shots_count = -1) {
  auto &platform = cudaq::get_platform();
  std::promise<evolve_result> promise;
  auto f = promise.get_future();

  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), kernels, initial_state, observables, noise_model,
       shots_count, &platform]() mutable {
        if (noise_model.has_value())
          platform.set_noise(&noise_model.value());
        p.set_value(evolve(initial_state, kernels, observables, shots_count));
        if (noise_model.has_value())
          platform.set_noise(nullptr);
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return f;
}

inline async_evolve_result
evolve_async(std::function<evolve_result()> evolveFunctor,
             std::size_t qpu_id = 0) {
  auto &platform = cudaq::get_platform();
  if (qpu_id >= platform.num_qpus()) {
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");
  }
  std::promise<evolve_result> promise;
  auto f = promise.get_future();

  QuantumTask wrapped = detail::make_copyable_function(
      [p = std::move(promise), evolveFunctor]() mutable {
        p.set_value(evolveFunctor());
      });

  platform.enqueueAsyncTask(qpu_id, wrapped);
  return f;
}
} // namespace __internal__

template <typename OperatorTy,
          typename = std::enable_if_t<
              std::is_same_v<std::decay_t<OperatorTy>,
                             cudaq::operator_sum<cudaq::matrix_operator>> ||
              std::is_constructible_v<
                  cudaq::operator_sum<cudaq::matrix_operator>, OperatorTy>>>
evolve_result evolve(const OperatorTy &hamiltonian,
                     const std::map<int, int> &dimensions,
                     const Schedule &schedule, const state &initial_state,
                     std::shared_ptr<BaseIntegrator> integrator = {},
                     const std::vector<OperatorTy> &collapse_operators = {},
                     const std::vector<OperatorTy> &observables = {},
                     bool store_intermediate_results = false,
                     std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_DYNAMICS_TARGET)
  if constexpr (std::is_same_v<std::decay_t<OperatorTy>,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
    std::vector<cudaq::operator_sum<cudaq::matrix_operator> *> collapseOpsPtr;
    for (const auto &cOp : collapse_operators) {
      collapseOpsPtr.emplace_back(
          const_cast<cudaq::operator_sum<cudaq::matrix_operator> *>(&cOp));
    }
    std::vector<cudaq::operator_sum<cudaq::matrix_operator> *> observeOpsPtr;
    for (const auto &obsOp : observables) {
      observeOpsPtr.emplace_back(
          const_cast<cudaq::operator_sum<cudaq::matrix_operator> *>(&obsOp));
    }
    // FIXME: change signature of `evolve_single` so that we don't need to
    // create the list of pointers.
    return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                         *integrator, collapseOpsPtr, observeOpsPtr,
                         store_intermediate_results);
  } else {
    std::vector<cudaq::operator_sum<cudaq::matrix_operator>>
        convertedCollapseOps;
    for (const auto &cOp : collapse_operators) {
      convertedCollapseOps.emplace_back(cOp);
    }
    std::vector<cudaq::operator_sum<cudaq::matrix_operator>>
        convertedObserveOps;
    for (const auto &obsOp : observables) {
      convertedObserveOps.emplace_back(obsOp);
    }
    std::vector<cudaq::operator_sum<cudaq::matrix_operator> *> collapseOpsPtr;
    for (const auto &cOp : convertedCollapseOps) {
      collapseOpsPtr.emplace_back(
          const_cast<cudaq::operator_sum<cudaq::matrix_operator> *>(&cOp));
    }
    std::vector<cudaq::operator_sum<cudaq::matrix_operator> *> observeOpsPtr;
    for (const auto &obsOp : convertedObserveOps) {
      observeOpsPtr.emplace_back(
          const_cast<cudaq::operator_sum<cudaq::matrix_operator> *>(&obsOp));
    }
    return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                         *integrator, collapseOpsPtr, observeOpsPtr,
                         store_intermediate_results);
  }
#else
  throw std::runtime_error(
      "cudaq::evolve is only supported on the 'dynamics' target. Please "
      "recompile your application with '--target dynamics' flag.");
#endif
}
template <typename OperatorTy,
          typename = std::enable_if_t<
              std::is_same_v<std::decay_t<OperatorTy>,
                             cudaq::operator_sum<cudaq::matrix_operator>> ||
              std::is_constructible_v<
                  cudaq::operator_sum<cudaq::matrix_operator>, OperatorTy>>>
// Multiple input state
std::vector<evolve_result>
evolve(const OperatorTy &hamiltonian, const std::map<int, int> &dimensions,
       const Schedule &schedule, const std::vector<state> &initial_states,
       std::shared_ptr<BaseIntegrator> integrator = {},
       const std::vector<OperatorTy> &collapse_operators = {},
       const std::vector<OperatorTy> &observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
  std::vector<evolve_result> results;
  for (const auto &initial_state : initial_states)
    results.emplace_back(evolve(hamiltonian, dimensions, schedule,
                                initial_states, integrator, collapse_operators,
                                observables, store_intermediate_results,
                                shots_count));
  return results;
}
} // namespace cudaq
