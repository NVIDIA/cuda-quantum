/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatState.h"
#include "cudaq/dynamics_integrators.h"
#include "cudaq/evolution.h"
#include "cudm_error_handling.h"
#include "cudm_expectation.h"
#include "cudm_helpers.h"
#include "cudm_time_stepper.h"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <stdexcept>
namespace cudaq {
evolve_result evolve_single(
    const operator_sum<cudaq::matrix_operator> &hamiltonian,
    const std::map<int, int> &dimensions, const Schedule &schedule,
    const state &initial_state, BaseIntegrator &in_integrator,
    const std::vector<operator_sum<cudaq::matrix_operator> *>
        &collapse_operators,
    const std::vector<operator_sum<cudaq::matrix_operator> *> &observables,
    bool store_intermediate_results, std::optional<int> shots_count) {
  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));
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
  asCudmState(const_cast<state &>(initial_state))
      ->initialize_cudm(handle, dims);

  runge_kutta &integrator = dynamic_cast<runge_kutta &>(in_integrator);
  SystemDynamics system;
  system.hamiltonian =
      const_cast<operator_sum<cudaq::matrix_operator> *>(&hamiltonian);
  system.collapseOps = collapse_operators;
  system.modeExtents = dims;
  integrator.set_system(system);

  integrator.set_state(initial_state, 0.0);

  cudm_helper helper(handle);
  std::vector<cudm_expectation> expectations;
  for (auto &obs : observables)
    expectations.emplace_back(cudm_expectation(
        handle, helper.convert_to_cudensitymat_operator<cudaq::matrix_operator>(
                    {}, *obs, dims)));

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