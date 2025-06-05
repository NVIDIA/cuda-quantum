/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EvolveResult.h"
#include "cudaq/algorithms/base_integrator.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include "cudaq/operators/operator_type.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/schedule.h"
#include "evolve_internal.h"

namespace cudaq {

/// @brief Return type for asynchronous `evolve_async`.
using async_evolve_result = std::future<evolve_result>;

namespace __internal__ {
template <typename OpTy>
cudaq::sum_op<cudaq::matrix_handler> convertOp(const OpTy &op) {
  if constexpr (std::is_convertible_v<
                    OpTy, cudaq::product_op<cudaq::matrix_handler>>) {
    cudaq::sum_op<cudaq::matrix_handler> convertedOp(op);
    return convertedOp;
  } else if constexpr (std::is_convertible_v<
                           OpTy, cudaq::sum_op<cudaq::matrix_handler>>) {
    return op;
  } else {
    throw std::invalid_argument("Invalid operator type: cannot convert type " +
                                std::string(typeid(op).name()) +
                                " to cudaq::product_op or "
                                "cudaq::sum_op");
  }
}

template <typename OpTy>
std::vector<cudaq::sum_op<cudaq::matrix_handler>>
convertOps(const std::vector<OpTy> &ops) {
  std::vector<cudaq::sum_op<cudaq::matrix_handler>> converted;
  for (const auto &op : ops)
    converted.emplace_back(convertOp(op));
  return converted;
}

template <typename OpTy>
std::vector<cudaq::sum_op<cudaq::matrix_handler>>
convertOps(const std::initializer_list<OpTy> &ops) {
  std::vector<cudaq::sum_op<cudaq::matrix_handler>> converted;
  for (const auto &op : ops)
    converted.emplace_back(convertOp(op));
  return converted;
}
} // namespace __internal__

#if CUDAQ_USE_STD20
template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
#else
template <typename HamTy,
          typename CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
evolve_result
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const state &initial_state,
       base_integrator &integrator,
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_state, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

#if CUDAQ_USE_STD20
template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
#else
template <typename HamTy,
          typename CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
evolve_result
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, InitialState initial_state,
       base_integrator &integrator,
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_state, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

#if CUDAQ_USE_STD20
template <operator_type HamTy, operator_type CollapseOpTy,
          operator_type ObserveOpTy>
#else
template <typename HamTy, typename CollapseOpTy, typename ObserveOpTy,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
evolve_result evolve(const HamTy &hamiltonian,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, const state &initial_state,
                     base_integrator &integrator,
                     const std::vector<CollapseOpTy> &collapse_operators = {},
                     const std::vector<ObserveOpTy> &observables = {},
                     bool store_intermediate_results = false,
                     std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_state, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

#if CUDAQ_USE_STD20
template <operator_type HamTy, operator_type CollapseOpTy,
          operator_type ObserveOpTy>
#else
template <typename HamTy, typename CollapseOpTy, typename ObserveOpTy,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
evolve_result evolve(const HamTy &hamiltonian,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, InitialState initial_state,
                     base_integrator &integrator,
                     const std::vector<CollapseOpTy> &collapse_operators = {},
                     const std::vector<ObserveOpTy> &observables = {},
                     bool store_intermediate_results = false,
                     std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_state, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

#if CUDAQ_USE_STD20
template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
#else
template <typename HamTy,
          typename CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
std::vector<evolve_result>
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const std::vector<state> &initial_states,
       base_integrator &integrator,
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveBatched(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_states, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      shots_count);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

#if CUDAQ_USE_STD20
template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
#else
template <typename HamTy,
          typename CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
std::vector<evolve_result>
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const std::vector<state> &initial_states,
       base_integrator &integrator,
       const std::vector<CollapseOpTy> &collapse_operators = {},
       const std::vector<ObserveOpTy> &observables = {},
       bool store_intermediate_results = false,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveBatched(
      cudaq::__internal__::convertOp(hamiltonian), dimensions, schedule,
      initial_states, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      shots_count);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

#if CUDAQ_USE_STD20
template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
#else
template <typename HamTy,
          typename CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
async_evolve_result
evolve_async(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
             const schedule &schedule, const state &initial_state,
             base_integrator &integrator,
             std::initializer_list<CollapseOpTy> collapse_operators = {},
             std::initializer_list<ObserveOpTy> observables = {},
             bool store_intermediate_results = false,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
#if defined(CUDAQ_ANALOG_TARGET)
  // Clone the integrator to extend its lifetime.
  auto cloneIntegrator = integrator.clone();
  auto collapseOperators = cudaq::__internal__::convertOps(collapse_operators);
  auto observableOperators = cudaq::__internal__::convertOps(observables);
  return __internal__::evolve_async(
      [=, cOps = std::move(collapseOperators),
       obs = std::move(observableOperators)]() {
        ExecutionContext context("evolve");
        cudaq::get_platform().set_exec_ctx(&context, qpu_id);
        state localizedState = cudaq::__internal__::migrateState(initial_state);
        return evolve(hamiltonian, dimensions, schedule, localizedState,
                      *cloneIntegrator, cOps, obs, store_intermediate_results,
                      shots_count);
      },
      qpu_id);

#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

#if CUDAQ_USE_STD20
template <operator_type HamTy, operator_type CollapseOpTy,
          operator_type ObserveOpTy>
#else
template <typename HamTy, typename CollapseOpTy, typename ObserveOpTy,
          typename = std::enable_if_t<cudaq::operator_type<HamTy> &&
                                      cudaq::operator_type<CollapseOpTy> &&
                                      cudaq::operator_type<ObserveOpTy>>>
#endif
async_evolve_result
evolve_async(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
             const schedule &schedule, const state &initial_state,
             base_integrator &integrator,
             const std::vector<CollapseOpTy> &collapse_operators = {},
             const std::vector<ObserveOpTy> &observables = {},
             bool store_intermediate_results = false,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
#if defined(CUDAQ_ANALOG_TARGET)
  // Clone the integrator to extend its lifetime.
  auto cloneIntegrator = integrator.clone();
  return __internal__::evolve_async(
      [=]() {
        ExecutionContext context("evolve");
        cudaq::get_platform().set_exec_ctx(&context, qpu_id);
        state localizedState = cudaq::__internal__::migrateState(initial_state);
        return evolve(hamiltonian, dimensions, schedule, localizedState,
                      *cloneIntegrator, collapse_operators, observables,
                      store_intermediate_results, shots_count);
      },
      qpu_id);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

// Rydberg Hamiltonian
evolve_result evolve(const cudaq::rydberg_hamiltonian &hamiltonian,
                     const cudaq::schedule &schedule,
                     std::optional<int> shots_count = std::nullopt) {
  return cudaq::__internal__::evolveSingle(hamiltonian, schedule, shots_count);
}

async_evolve_result evolve_async(const cudaq::rydberg_hamiltonian &hamiltonian,
                                 const cudaq::schedule &schedule,
                                 std::optional<int> shots_count = std::nullopt,
                                 int qpu_id = 0) {
  return cudaq::__internal__::evolve_async(
      [=]() {
        ExecutionContext context("evolve");
        cudaq::get_platform().set_exec_ctx(&context, qpu_id);
        return evolve(hamiltonian, schedule, shots_count);
      },
      qpu_id);
}

} // namespace cudaq
