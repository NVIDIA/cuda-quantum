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
  cudmState->initialize_cudm(handle, dims);

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
        const auto expVal =
            expectation.compute(cudmState->get_impl(), step.real());
        expVals.emplace_back(expVal.real());
      }
      expectationVals.emplace_back(std::move(expVals));
      intermediateStates.emplace_back(currentState);
    }
  }

  if (storeIntermediateResults) {
    return evolve_result(intermediateStates, expectationVals);
  } else {
    // Only final state is needed
    auto [finalTime, finalState] = integrator.getState();
    std::vector<double> expVals;
    auto *cudmState = asCudmState(finalState);
    for (auto &expectation : expectations) {
      expectation.prepare(cudmState->get_impl());
      const auto expVal = expectation.compute(cudmState->get_impl(), finalTime);
      expVals.emplace_back(expVal.real());
    }
    return evolve_result(finalState, expVals);
  }
}
} // namespace cudaq::__internal__
