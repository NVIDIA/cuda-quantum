/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatState.h"
#include "cudaq/dynamics_integrators.h"
#include "cudaq/evolution.h"
#include "cudm_error_handling.h"
#include "cudm_expectation.h"
#include "cudm_time_stepper.h"
#include <random>
#include <stdexcept>
namespace cudaq {
evolve_result evolve_single(
    const operator_sum<cudaq::matrix_operator> &hamiltonian,
    const std::map<int, int> &dimensions, const Schedule &schedule,
    const state &initialState, BaseIntegrator &in_integrator,
    const std::vector<operator_sum<cudaq::matrix_operator>> &collapse_operators,
    const std::vector<operator_sum<cudaq::matrix_operator>> &observables,
    bool store_intermediate_results, std::optional<int> shots_count) {
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
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

  state initial_state = [&]() {
    if (!collapse_operators.empty() && !cudmState->is_density_matrix()) {
      return state(new CuDensityMatState(cudmState->to_density_matrix()));
    }
    return initialState;
  }();

  runge_kutta &integrator = dynamic_cast<runge_kutta &>(in_integrator);
  SystemDynamics system;
  system.hamiltonian =
      const_cast<operator_sum<cudaq::matrix_operator> *>(&hamiltonian);
  system.collapseOps = collapse_operators;
  system.modeExtents = dims;
  integrator.set_system(system, schedule);
  integrator.set_state(initial_state, 0.0);
  std::vector<cudm_expectation> expectations;
  for (auto &obs : observables)
    expectations.emplace_back(cudm_expectation(
        handle, cudaq::dynamics::Context::getCurrentContext()
                    ->getOpConverter()
                    .convertToCudensitymatOperator({}, obs, dims)));

  std::vector<std::vector<double>> expectationVals;
  std::vector<cudaq::state> intermediateStates;
  for (const auto &step : schedule) {
    integrator.integrate(step);
    auto [t, currentState] = integrator.get_state();
    if (store_intermediate_results) {
      std::vector<double> expVals;

      for (auto &expectation : expectations) {
        auto *cudmState = asCudmState(currentState);
        expectation.prepare(cudmState->get_impl());
        const auto expVal = expectation.compute(cudmState->get_impl(), step);
        expVals.emplace_back(expVal.real());
      }
      expectationVals.emplace_back(std::move(expVals));
      intermediateStates.emplace_back(currentState);
    }
  }

  if (store_intermediate_results) {
    return evolve_result(intermediateStates, expectationVals);
  } else {
    // Only final state is needed
    auto [finalTime, finalState] = integrator.get_state();
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

} // namespace cudaq