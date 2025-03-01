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
#include "cudaq/BaseIntegrator.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/schedule.h"
#include "evolve_internal.h"

namespace cudaq {

/// @brief Return type for asynchronous `evolve_async`.
using async_evolve_result = std::future<evolve_result>;

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
#if defined(CUDAQ_DYNAMICS_TARGET)
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
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            cOpConverted, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            cOpConverted, observables, store_intermediate_results);
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
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            collapse_operators, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            collapse_operators, observables, store_intermediate_results);
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
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            cOpConverted, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            cOpConverted, observables, store_intermediate_results);
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
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            collapse_operators, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            collapse_operators, observables, store_intermediate_results);
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
#else
  throw std::runtime_error(
      "cudaq::evolve is only supported on the 'dynamics' target. Please "
      "recompile your application with '--target dynamics' flag.");
#endif
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
#if defined(CUDAQ_DYNAMICS_TARGET)
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
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            cOpConverted, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            cOpConverted, observables, store_intermediate_results);
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
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            collapse_operators, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            convertedHam, dimensions, schedule, initial_state, *integrator,
            collapse_operators, observables, store_intermediate_results);
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
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            cOpConverted, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            cOpConverted, observables, store_intermediate_results);
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
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            collapse_operators, obsOpConverted, store_intermediate_results);
      } else if constexpr (std::is_convertible_v<
                               ObserveOpTy,
                               cudaq::operator_sum<cudaq::matrix_operator>>) {
        return cudaq::__internal__::evolveSingle(
            hamiltonian, dimensions, schedule, initial_state, *integrator,
            collapse_operators, observables, store_intermediate_results);
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
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <typename HamTy,
          typename CollapseOpTy = cudaq::operator_sum<cudaq::matrix_operator>,
          typename ObserveOpTy = cudaq::operator_sum<cudaq::matrix_operator>>
std::vector<evolve_result>
evolve(const HamTy &hamiltonian, const std::map<int, int> &dimensions,
       const Schedule &schedule, const std::vector<state> &initial_states,
       std::shared_ptr<BaseIntegrator> integrator = {},
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_DYNAMICS_TARGET)
  std::vector<evolve_result> results;
  for (const auto &initial_state : initial_states)
    results.emplace_back(evolve(hamiltonian, dimensions, schedule,
                                initial_state, integrator, collapse_operators,
                                observables, store_intermediate_results,
                                shots_count));
  return results;
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <typename HamTy, typename CollapseOpTy, typename ObserveOpTy>
std::vector<evolve_result>
evolve(const HamTy &hamiltonian, const std::map<int, int> &dimensions,
       const Schedule &schedule, const std::vector<state> &initial_states,
       std::shared_ptr<BaseIntegrator> integrator = {},
       const std::vector<CollapseOpTy> &collapse_operators = {},
       const std::vector<ObserveOpTy> &observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_DYNAMICS_TARGET)
  std::vector<evolve_result> results;
  for (const auto &initial_state : initial_states)
    results.emplace_back(evolve(hamiltonian, dimensions, schedule,
                                initial_state, integrator, collapse_operators,
                                observables, store_intermediate_results,
                                shots_count));
  return results;
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <typename HamTy,
          typename CollapseOpTy = cudaq::operator_sum<cudaq::matrix_operator>,
          typename ObserveOpTy = cudaq::operator_sum<cudaq::matrix_operator>>
async_evolve_result
evolve_async(const HamTy &hamiltonian, const std::map<int, int> &dimensions,
             const Schedule &schedule, const state &initial_state,
             std::shared_ptr<BaseIntegrator> integrator = {},
             std::initializer_list<CollapseOpTy> collapse_operators = {},
             std::initializer_list<ObserveOpTy> observables = {},
             bool store_intermediate_results = false,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
#if defined(CUDAQ_DYNAMICS_TARGET)
  if (collapse_operators.size() > 0 && observables.size() > 0) {
    std::vector<CollapseOpTy> collapseOperators(collapse_operators);
    std::vector<ObserveOpTy> observableOperators(observables);
    return __internal__::evolve_async(
        [=, cOps = std::move(collapseOperators),
         obs = std::move(observableOperators)]() {
          ExecutionContext context("evolve");
          cudaq::get_platform().set_exec_ctx(&context, qpu_id);
          return evolve(hamiltonian, dimensions, schedule, initial_state,
                        integrator, cOps, obs, store_intermediate_results,
                        shots_count);
        },
        qpu_id);
  } else if (collapse_operators.size() > 0) {
    std::vector<CollapseOpTy> collapseOperators(collapse_operators);
    std::vector<CollapseOpTy> observableOperators;
    return __internal__::evolve_async(
        [=, cOps = std::move(collapseOperators),
         obs = std::move(observableOperators)]() {
          ExecutionContext context("evolve");
          cudaq::get_platform().set_exec_ctx(&context, qpu_id);
          return evolve(hamiltonian, dimensions, schedule, initial_state,
                        integrator, cOps, obs, store_intermediate_results,
                        shots_count);
        },
        qpu_id);
  } else if (observables.size()) {
    std::vector<ObserveOpTy> observableOperators(observables);
    std::vector<ObserveOpTy> collapseOperators;
    return __internal__::evolve_async(
        [=, cOps = std::move(collapseOperators),
         obs = std::move(observableOperators)]() {
          ExecutionContext context("evolve");
          cudaq::get_platform().set_exec_ctx(&context, qpu_id);
          return evolve(hamiltonian, dimensions, schedule, initial_state,
                        integrator, cOps, obs, store_intermediate_results,
                        shots_count);
        },
        qpu_id);
  } else {
    return __internal__::evolve_async(
        [=]() {
          ExecutionContext context("evolve");
          cudaq::get_platform().set_exec_ctx(&context, qpu_id);
          return evolve(hamiltonian, dimensions, schedule, initial_state,
                        integrator, {}, {}, store_intermediate_results,
                        shots_count);
        },
        qpu_id);
  }

#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <typename HamTy, typename CollapseOpTy, typename ObserveOpTy>
async_evolve_result
evolve_async(const HamTy &hamiltonian, const std::map<int, int> &dimensions,
             const Schedule &schedule, const state &initial_state,
             std::shared_ptr<BaseIntegrator> integrator = {},
             const std::vector<CollapseOpTy> &collapse_operators = {},
             const std::vector<ObserveOpTy> &observables = {},
             bool store_intermediate_results = false,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
#if defined(CUDAQ_DYNAMICS_TARGET)
  return __internal__::evolve_async(
      [=]() {
        ExecutionContext context("evolve");
        cudaq::get_platform().set_exec_ctx(&context, qpu_id);
        return evolve(hamiltonian, dimensions, schedule, initial_state,
                      integrator, collapse_operators, observables,
                      store_intermediate_results, shots_count);
      },
      qpu_id);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}
} // namespace cudaq
