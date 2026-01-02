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
std::vector<std::vector<cudaq::sum_op<cudaq::matrix_handler>>>
convertOps(const std::vector<std::vector<OpTy>> &opsList) {
  std::vector<std::vector<cudaq::sum_op<cudaq::matrix_handler>>> converted;
  for (const auto &ops : opsList) {
    std::vector<cudaq::sum_op<cudaq::matrix_handler>> convertedOps;
    for (const auto &op : ops)
      convertedOps.emplace_back(convertOp(op));
    converted.emplace_back(std::move(convertedOps));
  }
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

//===----------------------------------------------------------------------===//
// Single evolution API
// This API is used to evolve a single Hamiltonian with a single initial state.
//===----------------------------------------------------------------------===//

template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
evolve_result
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const state &initial_state,
       base_integrator &integrator,
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
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

template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
evolve_result
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, InitialState initial_state,
       base_integrator &integrator,
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
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

template <operator_type HamTy, operator_type CollapseOpTy,
          operator_type ObserveOpTy>
evolve_result evolve(const HamTy &hamiltonian,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, const state &initial_state,
                     base_integrator &integrator,
                     const std::vector<CollapseOpTy> &collapse_operators = {},
                     const std::vector<ObserveOpTy> &observables = {},
                     IntermediateResultSave store_intermediate_results =
                         IntermediateResultSave::None,
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

template <operator_type HamTy, operator_type CollapseOpTy,
          operator_type ObserveOpTy>
evolve_result evolve(const HamTy &hamiltonian,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, InitialState initial_state,
                     base_integrator &integrator,
                     const std::vector<CollapseOpTy> &collapse_operators = {},
                     const std::vector<ObserveOpTy> &observables = {},
                     IntermediateResultSave store_intermediate_results =
                         IntermediateResultSave::None,
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

// ===========================================================================
// Single super-operator evolution API
// ===========================================================================
// Overloads for both `std::vector` and `std::initializer_list` observables are
// provided to handle inline construction with braces at the call site.
template <operator_type ObserveOpTy>
evolve_result evolve(const super_op &super_op,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, const state &initial_state,
                     base_integrator &integrator,
                     const std::initializer_list<ObserveOpTy> &observables = {},
                     IntermediateResultSave store_intermediate_results =
                         IntermediateResultSave::None,
                     std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      super_op, dimensions, schedule, initial_state, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type ObserveOpTy>
evolve_result evolve(const super_op &super_op,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, InitialState initial_state,
                     base_integrator &integrator,
                     const std::initializer_list<ObserveOpTy> &observables = {},
                     IntermediateResultSave store_intermediate_results =
                         IntermediateResultSave::None,
                     std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      super_op, dimensions, schedule, initial_state, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type ObserveOpTy>
evolve_result evolve(const super_op &super_op,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, const state &initial_state,
                     base_integrator &integrator,
                     const std::vector<ObserveOpTy> &observables = {},
                     IntermediateResultSave store_intermediate_results =
                         IntermediateResultSave::None,
                     std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      super_op, dimensions, schedule, initial_state, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type ObserveOpTy>
evolve_result evolve(const super_op &super_op,
                     const cudaq::dimension_map &dimensions,
                     const schedule &schedule, InitialState initial_state,
                     base_integrator &integrator,
                     const std::vector<ObserveOpTy> &observables = {},
                     IntermediateResultSave store_intermediate_results =
                         IntermediateResultSave::None,
                     std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveSingle(
      super_op, dimensions, schedule, initial_state, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

//===----------------------------------------------------------------------===//
// Batched evolve functions are used to evolve multiple initial states with
// single/multiple Hamiltonian operators simultaneously.
//===----------------------------------------------------------------------===//

// ===========================================================================
// Single Hamiltonian with multiple initial states
// ===========================================================================
template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
std::vector<evolve_result>
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const std::vector<state> &initial_states,
       base_integrator &integrator,
       std::initializer_list<CollapseOpTy> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
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

template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
std::vector<evolve_result>
evolve(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const std::vector<state> &initial_states,
       base_integrator &integrator,
       const std::vector<CollapseOpTy> &collapse_operators = {},
       const std::vector<ObserveOpTy> &observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
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

// ===========================================================================
// Multiple Hamiltonians with multiple initial states
// ===========================================================================
// We provide overloads for both `std::vector` and `std::initializer_list` to
// handle inline Hamiltonian list construction with braces at the call site.
template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
std::vector<evolve_result>
evolve(const std::initializer_list<HamTy> &hamiltonians,
       const cudaq::dimension_map &dimensions, const schedule &schedule,
       const std::vector<state> &initial_states, base_integrator &integrator,
       std::vector<std::vector<CollapseOpTy>> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> batch_size = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  if (batch_size.has_value() && batch_size.value() < 1)
    throw std::invalid_argument(
        "Invalid batch size: " + std::to_string(batch_size.value()) +
        ". It must be at least 1.");
  return cudaq::__internal__::evolveBatched(
      cudaq::__internal__::convertOps(hamiltonians), dimensions, schedule,
      initial_states, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      batch_size);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
std::vector<evolve_result>
evolve(const std::initializer_list<HamTy> &hamiltonians,
       const cudaq::dimension_map &dimensions, const schedule &schedule,
       const std::vector<state> &initial_states, base_integrator &integrator,
       std::vector<std::vector<CollapseOpTy>> collapse_operators = {},
       std::vector<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> batch_size = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  if (batch_size.has_value() && batch_size.value() < 1)
    throw std::invalid_argument(
        "Invalid batch size: " + std::to_string(batch_size.value()) +
        ". It must be at least 1.");
  return cudaq::__internal__::evolveBatched(
      cudaq::__internal__::convertOps(hamiltonians), dimensions, schedule,
      initial_states, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      batch_size);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
std::vector<evolve_result>
evolve(const std::vector<HamTy> &hamiltonians,
       const cudaq::dimension_map &dimensions, const schedule &schedule,
       const std::vector<state> &initial_states, base_integrator &integrator,
       std::vector<std::vector<CollapseOpTy>> collapse_operators = {},
       std::initializer_list<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> batch_size = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  if (batch_size.has_value() && batch_size.value() < 1)
    throw std::invalid_argument(
        "Invalid batch size: " + std::to_string(batch_size.value()) +
        ". It must be at least 1.");
  return cudaq::__internal__::evolveBatched(
      cudaq::__internal__::convertOps(hamiltonians), dimensions, schedule,
      initial_states, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      batch_size);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
std::vector<evolve_result>
evolve(const std::vector<HamTy> &hamiltonians,
       const cudaq::dimension_map &dimensions, const schedule &schedule,
       const std::vector<state> &initial_states, base_integrator &integrator,
       const std::vector<std::vector<CollapseOpTy>> &collapse_operators = {},
       const std::vector<ObserveOpTy> &observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> batch_size = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  if (batch_size.has_value() && batch_size.value() < 1)
    throw std::invalid_argument(
        "Invalid batch size: " + std::to_string(batch_size.value()) +
        ". It must be at least 1.");
  return cudaq::__internal__::evolveBatched(
      cudaq::__internal__::convertOps(hamiltonians), dimensions, schedule,
      initial_states, integrator,
      cudaq::__internal__::convertOps(collapse_operators),
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      batch_size);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

// ===========================================================================
// Single super-operator with multiple initial states
// ===========================================================================
template <operator_type ObserveOpTy>
std::vector<evolve_result>
evolve(const super_op &super_op, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const std::vector<state> &initial_states,
       base_integrator &integrator,
       std::initializer_list<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveBatched(
      super_op, dimensions, schedule, initial_states, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      shots_count);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type ObserveOpTy>
std::vector<evolve_result>
evolve(const super_op &super_op, const cudaq::dimension_map &dimensions,
       const schedule &schedule, const std::vector<state> &initial_states,
       base_integrator &integrator, std::vector<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> shots_count = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  return cudaq::__internal__::evolveBatched(
      super_op, dimensions, schedule, initial_states, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      shots_count);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

// ===========================================================================
// Multiple super-operators with multiple initial states
// ===========================================================================
template <operator_type ObserveOpTy>
std::vector<evolve_result>
evolve(const std::vector<super_op> &super_ops,
       const cudaq::dimension_map &dimensions, const schedule &schedule,
       const std::vector<state> &initial_states, base_integrator &integrator,
       std::initializer_list<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> batch_size = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  if (batch_size.has_value() && batch_size.value() < 1)
    throw std::invalid_argument(
        "Invalid batch size: " + std::to_string(batch_size.value()) +
        ". It must be at least 1.");

  return cudaq::__internal__::evolveBatched(
      super_ops, dimensions, schedule, initial_states, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      batch_size);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type ObserveOpTy>
std::vector<evolve_result>
evolve(const std::vector<super_op> &super_ops,
       const cudaq::dimension_map &dimensions, const schedule &schedule,
       const std::vector<state> &initial_states, base_integrator &integrator,
       std::vector<ObserveOpTy> observables = {},
       IntermediateResultSave store_intermediate_results =
           IntermediateResultSave::None,
       std::optional<int> batch_size = std::nullopt) {
#if defined(CUDAQ_ANALOG_TARGET)
  if (batch_size.has_value() && batch_size.value() < 1)
    throw std::invalid_argument(
        "Invalid batch size: " + std::to_string(batch_size.value()) +
        ". It must be at least 1.");

  return cudaq::__internal__::evolveBatched(
      super_ops, dimensions, schedule, initial_states, integrator,
      cudaq::__internal__::convertOps(observables), store_intermediate_results,
      batch_size);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type HamTy,
          operator_type CollapseOpTy = cudaq::sum_op<cudaq::matrix_handler>,
          operator_type ObserveOpTy = cudaq::sum_op<cudaq::matrix_handler>>
async_evolve_result
evolve_async(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
             const schedule &schedule, const state &initial_state,
             base_integrator &integrator,
             std::initializer_list<CollapseOpTy> collapse_operators = {},
             std::initializer_list<ObserveOpTy> observables = {},
             IntermediateResultSave store_intermediate_results =
                 IntermediateResultSave::None,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
#if defined(CUDAQ_ANALOG_TARGET)
  // Clone the integrator to extend its lifetime.
  auto cloneIntegrator = integrator.clone();
  auto collapseOperators = cudaq::__internal__::convertOps(collapse_operators);
  auto observableOperators = cudaq::__internal__::convertOps(observables);
  return __internal__::evolve_async(
      [=, cOps = std::move(collapseOperators),
       obs = std::move(observableOperators)]() {
        // This is a dummy/unused execution context.
        // We used this to set the execution context for the QPU, which performs
        // GPU Id selection based on the QPU Id.
        ExecutionContext context("evolve");
        context.qpuId = qpu_id;
        cudaq::get_platform().set_exec_ctx(&context);
        state localizedState = cudaq::__internal__::migrateState(initial_state);
        const auto result = evolve(hamiltonian, dimensions, schedule,
                                   localizedState, *cloneIntegrator, cOps, obs,
                                   store_intermediate_results, shots_count);
        cudaq::get_platform().reset_exec_ctx();
        return result;
      },
      qpu_id);

#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type HamTy, operator_type CollapseOpTy,
          operator_type ObserveOpTy>
async_evolve_result
evolve_async(const HamTy &hamiltonian, const cudaq::dimension_map &dimensions,
             const schedule &schedule, const state &initial_state,
             base_integrator &integrator,
             const std::vector<CollapseOpTy> &collapse_operators = {},
             const std::vector<ObserveOpTy> &observables = {},
             IntermediateResultSave store_intermediate_results =
                 IntermediateResultSave::None,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
#if defined(CUDAQ_ANALOG_TARGET)
  // Clone the integrator to extend its lifetime.
  auto cloneIntegrator = integrator.clone();
  return __internal__::evolve_async(
      [=]() {
        // This is a dummy/unused execution context.
        // We used this to set the execution context for the QPU, which performs
        // GPU Id selection based on the QPU Id.
        ExecutionContext context("evolve");
        context.qpuId = qpu_id;
        cudaq::get_platform().set_exec_ctx(&context);
        state localizedState = cudaq::__internal__::migrateState(initial_state);
        auto result = evolve(hamiltonian, dimensions, schedule, localizedState,
                             *cloneIntegrator, collapse_operators, observables,
                             store_intermediate_results, shots_count);
        cudaq::get_platform().reset_exec_ctx();
        return result;
      },
      qpu_id);
#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

template <operator_type ObserveOpTy>
async_evolve_result
evolve_async(const super_op &super_op, const cudaq::dimension_map &dimensions,
             const schedule &schedule, const state &initial_state,
             base_integrator &integrator,
             std::initializer_list<ObserveOpTy> observables = {},
             IntermediateResultSave store_intermediate_results =
                 IntermediateResultSave::None,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
#if defined(CUDAQ_ANALOG_TARGET)
  // Clone the integrator to extend its lifetime.
  auto cloneIntegrator = integrator.clone();
  auto observableOperators = cudaq::__internal__::convertOps(observables);
  return __internal__::evolve_async(
      [=, obs = std::move(observableOperators)]() {
        // This is a dummy/unused execution context.
        // We used this to set the execution context for the QPU, which performs
        // GPU Id selection based on the QPU Id.
        ExecutionContext context("evolve");
        context.qpuId = qpu_id;
        cudaq::get_platform().set_exec_ctx(&context);
        state localizedState = cudaq::__internal__::migrateState(initial_state);
        auto result = evolve(super_op, dimensions, schedule, localizedState,
                             *cloneIntegrator, obs, store_intermediate_results,
                             shots_count);
        cudaq::get_platform().reset_exec_ctx();
        return result;
      },
      qpu_id);

#else
  static_assert(
      false, "cudaq::evolve is only supported on the 'dynamics' target. Please "
             "recompile your application with '--target dynamics' flag.");
#endif
}

// Rydberg Hamiltonian
inline evolve_result evolve(const cudaq::rydberg_hamiltonian &hamiltonian,
                            const cudaq::schedule &schedule,
                            std::optional<int> shots_count = std::nullopt) {
  return cudaq::__internal__::evolveSingle(hamiltonian, schedule, shots_count);
}

inline async_evolve_result
evolve_async(const cudaq::rydberg_hamiltonian &hamiltonian,
             const cudaq::schedule &schedule,
             std::optional<int> shots_count = std::nullopt, int qpu_id = 0) {
  return cudaq::__internal__::evolve_async(
      [=]() {
        ExecutionContext context("evolve");
        context.qpuId = qpu_id;
        cudaq::get_platform().set_exec_ctx(&context);
        auto result = evolve(hamiltonian, schedule, shots_count);
        cudaq::get_platform().reset_exec_ctx();
        return result;
      },
      qpu_id);
}

} // namespace cudaq
