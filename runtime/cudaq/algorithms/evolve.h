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

namespace __internal__ {
template <typename OpTy>
cudaq::operator_sum<cudaq::matrix_operator> convertOp(const OpTy &op) {
  if constexpr (std::is_convertible_v<
                    OpTy, cudaq::product_operator<cudaq::matrix_operator>>) {
    cudaq::operator_sum<cudaq::matrix_operator> convertedOp(op);
    return convertedOp;
  } else if constexpr (std::is_convertible_v<
                           OpTy, cudaq::operator_sum<cudaq::matrix_operator>>) {
    return op;
  } else {
    throw std::invalid_argument("Invalid operator type: cannot convert type " +
                                std::string(typeid(op).name()) +
                                " to cudaq::product_operator or "
                                "cudaq::operator_sum");
  }
}

template <typename OpTy>
std::vector<cudaq::operator_sum<cudaq::matrix_operator>>
convertOps(const std::vector<OpTy> &ops) {
  std::vector<cudaq::operator_sum<cudaq::matrix_operator>> converted;
  for (const auto &op : ops)
    converted.emplace_back(convertOp(op));
  return converted;
}

template <typename OpTy>
std::vector<cudaq::operator_sum<cudaq::matrix_operator>>
convertOps(const std::initializer_list<OpTy> &ops) {
  std::vector<cudaq::operator_sum<cudaq::matrix_operator>> converted;
  for (const auto &op : ops)
    converted.emplace_back(convertOp(op));
  return converted;
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
#if defined(CUDAQ_DYNAMICS_TARGET)
  return cudaq::__internal__::evolveSingle(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_state, *integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
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
  return cudaq::__internal__::evolveSingle(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_state, *integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
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