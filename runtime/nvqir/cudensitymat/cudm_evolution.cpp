/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/evolution.h"
#include "cudm_error_handling.h"
#include "cudaq/runge_kutta_integrator.h"
#include <Eigen/Dense>
#include "cudm_expectation.h"
#include "cudm_time_stepper.h"
#include <iostream>
#include <random>
#include <stdexcept>
#include "cudm_helpers.h"
#include "cudm_state.h"
namespace cudaq {
evolve_result evolve_single(
    const operator_sum<cudaq::matrix_operator> &hamiltonian,
    const std::map<int, int> &dimensions, const Schedule &schedule,
    const state &initial_state,
    BaseIntegrator& in_integrator,
    const std::vector<operator_sum<cudaq::matrix_operator> *>
        &collapse_operators,
    const std::vector<operator_sum<cudaq::matrix_operator> *> &observables,
    bool store_intermediate_results,
    std::optional<int> shots_count) {
  cudensitymatHandle_t handle;
  HANDLE_CUDM_ERROR(cudensitymatCreate(&handle));

  cudm_helper helper(handle);

  std::vector<int64_t> dims;
  for (const auto &[id, dim] : dimensions)
    dims.emplace_back(dim);

  auto cudmState = cudm_state(handle, initial_state, dims);
  auto liouvillian = helper.construct_liouvillian(
      hamiltonian, collapse_operators, dims, {}, cudmState.is_density_matrix());
  // std::cout << "Evolve Liouvillian: " << liouvillian << "\n";
  // Need to pass liouvillian here
  auto time_stepper = std::make_shared<cudm_time_stepper>(handle, liouvillian);
  runge_kutta_integrator &integrator =
      dynamic_cast<runge_kutta_integrator &>(in_integrator);
  integrator.set_stepper(time_stepper);
  integrator.set_state(std::move(cudmState));
  // auto integrator = std::make_unique<runge_kutta_integrator>(
  //     cudm_state(handle, initial_state, dims), 0.0, time_stepper, 1);
  // integrator.set_option("dt", 0.000001);

  std::vector<cudm_expectation> expectations;
  for (auto &obs : observables)
    expectations.emplace_back(cudm_expectation(
        handle, helper.convert_to_cudensitymat_operator<cudaq::matrix_operator>(
                    {}, *obs, dims)));

  std::vector<std::vector<double>> expectationVals;
  for (const auto &step : schedule) {
    integrator.integrate(step);
    auto [t, currentState] = integrator.get_cudm_state();
    if (store_intermediate_results) {
      std::vector<double> expVals;
      for (auto &expectation : expectations) {
        expectation.prepare(currentState->get_impl());
        const auto expVal = expectation.compute(currentState->get_impl(), step);
        expVals.emplace_back(expVal.real());
      }
      expectationVals.emplace_back(std::move(expVals));
    }
  }

  if (store_intermediate_results) {
    // TODO: need to convert to proper state
    return evolve_result({initial_state}, expectationVals);
  } else {
    // Only final state is needed
    auto [finalTime, finalState] = integrator.get_cudm_state();
    std::vector<double> expVals;
    for (auto &expectation : expectations) {
      expectation.prepare(finalState->get_impl());
      const auto expVal = expectation.compute(finalState->get_impl(), finalTime);
      expVals.emplace_back(expVal.real());
    }
    // TODO: need to convert to proper state
    return evolve_result(initial_state, expVals);
  }
}

} // namespace cudaq