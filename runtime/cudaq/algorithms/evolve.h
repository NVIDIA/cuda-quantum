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

template <typename HamTy,
          typename CollapseOpTy = cudaq::operator_sum<cudaq::matrix_operator>,
          typename ObserveOpTy = cudaq::operator_sum<cudaq::matrix_operator>>
evolve_result
evolve(const HamTy &hamiltonian, const std::map<int, int> &dimensions,
       const Schedule &schedule, const state &initial_state,
       std::shared_ptr<BaseIntegrator> integrator = {},
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
  if constexpr (std::is_convertible_v<
                    HamTy, cudaq::product_operator<cudaq::matrix_operator>>) {
    cudaq::operator_sum<cudaq::matrix_operator> convertedHam(hamiltonian);
    if (std::is_convertible_v<
            CollapseOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
      std::vector<cudaq::operator_sum<cudaq::matrix_operator>> cOpConverted;
      for (const auto &cOp : collapse_operators)
        cOpConverted.emplace_back(cOp);
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else if constexpr (std::is_convertible_v<
                             CollapseOpTy,
                             cudaq::operator_sum<cudaq::matrix_operator>>) {
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else {
      throw std::invalid_argument("Collapse operator type is not convertible "
                                  "to cudaq::matrix_operator");
    }
  } else if constexpr (std::is_convertible_v<
                           HamTy,
                           cudaq::operator_sum<cudaq::matrix_operator>>) {
    if (std::is_convertible_v<
            CollapseOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
      std::vector<cudaq::operator_sum<cudaq::matrix_operator>> cOpConverted;
      for (const auto &cOp : collapse_operators)
        cOpConverted.emplace_back(cOp);
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else if constexpr (std::is_convertible_v<
                             CollapseOpTy,
                             cudaq::operator_sum<cudaq::matrix_operator>>) {
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else {
      throw std::invalid_argument("Collapse operator type is not convertible "
                                  "to cudaq::matrix_operator");
    }
  } else {
    throw std::invalid_argument(
        "Hamiltonian type is not convertible to cudaq::matrix_operator");
  }
}

template <typename HamTy, typename CollapseOpTy, typename ObserveOpTy>
evolve_result evolve(const HamTy &hamiltonian,
                     const std::map<int, int> &dimensions,
                     const Schedule &schedule, const state &initial_state,
                     std::shared_ptr<BaseIntegrator> integrator = {},
                     const std::vector<CollapseOpTy> &collapse_operators = {},
                     const std::vector<ObserveOpTy> &observables = {},
                     bool store_intermediate_results = false,
                     std::optional<int> shots_count = std::nullopt) {
  if constexpr (std::is_convertible_v<
                    HamTy, cudaq::product_operator<cudaq::matrix_operator>>) {

    cudaq::operator_sum<cudaq::matrix_operator> convertedHam(hamiltonian);
    if (std::is_convertible_v<
            CollapseOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
      std::vector<cudaq::operator_sum<cudaq::matrix_operator>> cOpConverted;
      for (const auto &cOp : collapse_operators)
        cOpConverted.emplace_back(cOp);
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else if constexpr (std::is_convertible_v<
                             CollapseOpTy,
                             cudaq::operator_sum<cudaq::matrix_operator>>) {
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(convertedHam, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else {
      throw std::invalid_argument("Collapse operator type is not convertible "
                                  "to cudaq::matrix_operator");
    }
  } else if constexpr (std::is_convertible_v<
                           HamTy,
                           cudaq::operator_sum<cudaq::matrix_operator>>) {
    if (std::is_convertible_v<
            CollapseOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
      std::vector<cudaq::operator_sum<cudaq::matrix_operator>> cOpConverted;
      for (const auto &cOp : collapse_operators)
        cOpConverted.emplace_back(cOp);
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, cOpConverted, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else if constexpr (std::is_convertible_v<
                             CollapseOpTy,
                             cudaq::operator_sum<cudaq::matrix_operator>>) {
      if (std::is_convertible_v<
              ObserveOpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
        std::vector<cudaq::operator_sum<cudaq::matrix_operator>> obsOpConverted;
        for (const auto &obsOp : observables)
          obsOpConverted.emplace_back(obsOp);
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, obsOpConverted,
                             store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return evolve_single(hamiltonian, dimensions, schedule, initial_state,
                             *integrator, collapse_operators, observables,
                             store_intermediate_results);
      } else {
        throw std::invalid_argument("Observe operator type is not convertible "
                                    "to cudaq::matrix_operator");
      }
    } else {
      throw std::invalid_argument("Collapse operator type is not convertible "
                                  "to cudaq::matrix_operator");
    }
  } else {
    throw std::invalid_argument(
        "Hamiltonian type is not convertible to cudaq::matrix_operator");
  }
}

template <typename HamTy,
          typename CollapseOpTy = cudaq::operator_sum<cudaq::matrix_operator>,
          typename ObserveOpTy = cudaq::operator_sum<cudaq::matrix_operator>>
async_evolve_result
evolve_async(const HamTy &hamiltonian, const std::map<int, int> &dimensions,
             const Schedule &schedule, const state &initial_state,
             std::shared_ptr<BaseIntegrator> integrator = {},
             const std::vector<CollapseOpTy> &collapse_operators = {},
             const std::vector<ObserveOpTy> &observables = {},
             bool store_intermediate_results = false,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
  return __internal__::evolve_async(
      [=]() {
        ExecutionContext context("evolve");
        cudaq::get_platform().set_exec_ctx(&context, qpu_id);
        return evolve(hamiltonian, dimensions, schedule, initial_state,
                      integrator, collapse_operators, observables,
                      store_intermediate_results, shots_count);
      },
      qpu_id);
}
} // namespace cudaq
