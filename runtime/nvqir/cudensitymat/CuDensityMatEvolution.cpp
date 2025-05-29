/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatExpectation.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatUtils.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include <iterator>
#include <random>
#include <stdexcept>
namespace cudaq::__internal__ {
template <typename Key, typename Value>
std::map<Key, Value>
convertToOrderedMap(const std::unordered_map<Key, Value> &unorderedMap) {
  return std::map<Key, Value>(unorderedMap.begin(), unorderedMap.end());
}

state migrateState(const state &inputState) {
  const auto currentDeviceId =
      dynamics::Context::getCurrentContext()->getDeviceId();
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(
      cudaPointerGetAttributes(&attributes, inputState.get_tensor().data));
  const auto stateDeviceId = attributes.device;
  if (currentDeviceId == stateDeviceId)
    return inputState;

  cudaq::info("Migrate state data from device {} to {}\n", stateDeviceId,
              currentDeviceId);
  const int64_t dim = inputState.get_tensor().get_num_elements();
  const int64_t arraySizeBytes = dim * sizeof(std::complex<double>);
  auto localizedState =
      cudaq::dynamics::DeviceAllocator::allocate(arraySizeBytes);
  HANDLE_CUDA_ERROR(cudaMemcpy(localizedState, inputState.get_tensor().data,
                               arraySizeBytes, cudaMemcpyDefault));
  return state(new CuDensityMatState(dim, localizedState));
}

/// @brief Evolve the system for a single time step.
/// @param hamiltonian Hamiltonian operator.
/// @param dimensionsMap Dimension of the system.
/// @param schedule Time schedule.
/// @param initialState Initial state.
/// @param inIntegrator Integrator.
/// @param collapseOperators Collapse operators.
/// @param observables Observables.
/// @param storeIntermediateResults Store intermediate results.
/// @param shotsCount Number of shots.
/// @return evolve_result Result of the evolution.
evolve_result evolveSingle(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensionsMap, const schedule &schedule,
    const state &initialState, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapseOperators,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables,
    bool storeIntermediateResults, std::optional<int> shotsCount) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::map<std::size_t, int64_t> dimensions =
      convertToOrderedMap(dimensionsMap);
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : dimensions)
    dims.emplace_back(dim);
  const auto asCudmState = [](cudaq::state &cudaqState) -> CuDensityMatState * {
    auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    if (!castSimState)
      throw std::runtime_error("Invalid state.");
    return castSimState;
  };

  auto *cudmState = asCudmState(const_cast<state &>(initialState));
  if (!cudmState->is_initialized())
    cudmState->initialize_cudm(handle, dims, /*batchSize=*/1);

  state initial_State = [&]() {
    if (!collapseOperators.empty() && !cudmState->is_density_matrix())
      return state(new CuDensityMatState(cudmState->to_density_matrix()));
    return initialState;
  }();

  SystemDynamics system(dims, hamiltonian, collapseOperators);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);
  integrator.setState(initial_State, 0.0);
  std::vector<CuDensityMatExpectation> expectations;
  for (auto &obs : observables)
    expectations.emplace_back(CuDensityMatExpectation(
        handle, cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, obs, dims)));

  std::vector<std::vector<double>> expectationVals;
  std::vector<cudaq::state> intermediateStates;
  for (const auto &step : schedule) {
    integrator.integrate(step.real());
    auto [t, currentState] = integrator.getState();
    if (storeIntermediateResults) {
      std::vector<double> expVals;

      for (auto &expectation : expectations) {
        auto *cudmState = asCudmState(currentState);
        expectation.prepare(cudmState->get_impl());
        const auto expVal = expectation.compute(cudmState->get_impl(),
                                                step.real(), /*batchSize=*/1);
        assert(expVal.size() == 1);
        expVals.emplace_back(expVal.front().real());
      }
      expectationVals.emplace_back(std::move(expVals));
      intermediateStates.emplace_back(currentState);
    }
  }

  if (cudaq::details::should_log(cudaq::details::LogLevel::trace))
    cudaq::dynamics::dumpPerfTrace();

  if (storeIntermediateResults) {
    return evolve_result(intermediateStates, expectationVals);
  } else {
    // Only final state is needed
    auto [finalTime, finalState] = integrator.getState();
    std::vector<double> expVals;
    auto *cudmState = asCudmState(finalState);
    for (auto &expectation : expectations) {
      expectation.prepare(cudmState->get_impl());
      const auto expVal = expectation.compute(cudmState->get_impl(), finalTime,
                                              /*batchSize=*/1);
      assert(expVal.size() == 1);
      expVals.emplace_back(expVal.front().real());
    }
    return evolve_result(finalState, expVals);
  }
}

/// @brief Evolve the system for a single time step.
/// @param hamiltonian Hamiltonian operator.
/// @param dimensions Dimension of the system.
/// @param schedule Time schedule.
/// @param initial_state Initial state enum.
/// @param integrator Integrator.
/// @param collapse_operators Collapse operators.
/// @param observables Observables.
/// @param store_intermediate_results Store intermediate results.
/// @param shots_count Number of shots.
/// @return evolve_result Result of the evolution.
evolve_result evolveSingle(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensions, const schedule &schedule,
    InitialState initial_state, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapse_operators,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables,
    bool store_intermediate_results, std::optional<int> shots_count) {
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  auto cudmState = CuDensityMatState::createInitialState(
      handle, initial_state, dimensions, collapse_operators.size() > 0);
  return evolveSingle(
      hamiltonian, dimensions, schedule, state(cudmState.release()), integrator,
      collapse_operators, observables, store_intermediate_results, shots_count);
}

std::vector<evolve_result> evolveBatched(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensionsMap, const schedule &schedule,
    const std::vector<state> &initialStates, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapseOperators,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables,
    bool storeIntermediateResults, std::optional<int> shotsCount) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::map<std::size_t, int64_t> dimensions =
      convertToOrderedMap(dimensionsMap);
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : dimensions)
    dims.emplace_back(dim);
  const auto asCudmState = [](cudaq::state &cudaqState) -> CuDensityMatState * {
    auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    if (!castSimState)
      throw std::runtime_error("Invalid state.");
    return castSimState;
  };
  std::vector<CuDensityMatState *> states;
  for (auto &initialState : initialStates) {
    states.emplace_back(asCudmState(const_cast<state &>(initialState)));
  }
  auto batchedState = CuDensityMatState::createBatchedState(
      handle, states, dims, !collapseOperators.empty());
  SystemDynamics system(dims, hamiltonian, collapseOperators);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);
  integrator.setState(cudaq::state(batchedState.release()), 0.0);
  std::vector<CuDensityMatExpectation> expectations;
  auto &opConverter =
      cudaq::dynamics::Context::getCurrentContext()->getOpConverter();
  for (auto &obs : observables) {
    auto cudmObsOp = opConverter.convertToCudensitymatOperator({}, obs, dims);
    expectations.emplace_back(CuDensityMatExpectation(handle, cudmObsOp));
  }

  std::vector<std::vector<std::vector<double>>> expectationVals(
      initialStates.size());
  std::vector<std::vector<cudaq::state>> intermediateStates(
      initialStates.size());
  for (const auto &step : schedule) {
    integrator.integrate(step.real());
    auto [t, currentState] = integrator.getState();
    if (storeIntermediateResults) {
      auto *cudmState = asCudmState(currentState);
      std::vector<std::vector<double>> expVals(initialStates.size());
      for (auto &expectation : expectations) {
        expectation.prepare(cudmState->get_impl());
        const auto expVal = expectation.compute(
            cudmState->get_impl(), step.real(), initialStates.size());
        assert(expVal.size() == initialStates.size());
        for (int i = 0; i < expVal.size(); ++i) {
          expVals[i].emplace_back(expVal[i].real());
        }
      }
      auto states = CuDensityMatState::splitBatchedState(*cudmState);
      assert(states.size() == initialStates.size());
      for (int i = 0; i < initialStates.size(); ++i) {
        expectationVals[i].emplace_back(expVals[i]);
        intermediateStates[i].emplace_back(cudaq::state(states[i]));
      }
    }
  }

  if (storeIntermediateResults) {
    std::vector<evolve_result> results;
    for (int i = 0; i < initialStates.size(); ++i) {
      results.emplace_back(
          evolve_result(intermediateStates[i], expectationVals[i]));
    }
    return results;
  } else {
    // Only final state is needed
    auto [finalTime, finalState] = integrator.getState();
    auto *cudmState = asCudmState(finalState);
    std::vector<std::vector<double>> expVals(initialStates.size());
    for (auto &expectation : expectations) {
      expectation.prepare(cudmState->get_impl());
      const auto expVal = expectation.compute(cudmState->get_impl(), finalTime,
                                              initialStates.size());
      assert(expVal.size() == initialStates.size());
      for (int i = 0; i < expVal.size(); ++i) {
        expVals[i].emplace_back(expVal[i].real());
      }
    }
    auto states = CuDensityMatState::splitBatchedState(*cudmState);
    assert(states.size() == initialStates.size());
    std::vector<evolve_result> results;
    for (int i = 0; i < initialStates.size(); ++i) {
      results.emplace_back(evolve_result(cudaq::state(states[i]), expVals[i]));
    }
    return results;
  }
}

} // namespace cudaq::__internal__
