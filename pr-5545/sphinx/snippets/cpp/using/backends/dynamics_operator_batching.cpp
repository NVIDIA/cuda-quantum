/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include <cudaq.h>

int main() {
  // [Begin Operator Batching]
  // Dimensions of sub-system
  cudaq::dimension_map dimensions = {{0, 2}};
  // Qubit resonant frequency
  const double omega_z = 10.0 * 2 * M_PI;
  // Transverse driving term
  const double omega_x = 2 * M_PI;
  // Harmonic driving frequency (sweeping in the +/- 10% range around the
  // resonant frequency).
  const auto omega_drive = cudaq::linspace(0.9 * omega_z, 1.1 * omega_z, 16);
  const auto zero_state =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  // List of Hamiltonians to be batched together
  std::vector<cudaq::spin_op> hamiltonians;

  for (const auto &omega : omega_drive) {
    auto mod_func =
        [omega](const cudaq::parameter_map &params) -> std::complex<double> {
      auto it = params.find("t");
      if (it != params.end()) {
        double t = it->second.real();
        return std::cos(omega * t);
      }
      throw std::runtime_error("Cannot find the time parameter.");
    };

    // Add the Hamiltonian for each drive frequency to the batch.
    hamiltonians.emplace_back(0.5 * omega_z * cudaq::spin_op::z(0) +
                              mod_func * cudaq::spin_op::x(0) * omega_x);
  }

  // The qubit starts in the |0> state for all operators in the batch.
  std::vector<cudaq::state> initial_states(hamiltonians.size(), zero_state);
  // Schedule of time steps
  const std::vector<double> steps = cudaq::linspace(0.0, 0.5, 5000);
  // The schedule carries the time parameter `labelled` `t`, which is used by
  // the callback.
  cudaq::schedule schedule(steps, {"t"});

  // A default Runge-`Kutta` integrator (4`th` order) with time step `dt`
  // depending on the schedule.
  cudaq::integrators::runge_kutta integrator;

  // Run the batch simulation.
  auto evolve_results = cudaq::evolve(
      hamiltonians, dimensions, schedule, initial_states, integrator, {},
      {cudaq::spin_op::x(0), cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
      cudaq::IntermediateResultSave::ExpectationValue);

  // [End Operator Batching]

  // [Begin Batch Results]
  // The results of the batched evolution is an array of evolution results, one
  // for each Hamiltonian operator in the batch.

  // For example, we can split the results into separate arrays for each
  // observable. These will be nested lists, where each inner list corresponds
  // to the results for a specific Hamiltonian operator in the batch.
  std::vector<std::vector<double>> all_exp_val_x;
  std::vector<std::vector<double>> all_exp_val_y;
  std::vector<std::vector<double>> all_exp_val_z;
  // Iterate over the evolution results in the batch:
  for (auto &evolution_result : evolve_results) {
    // Extract the expectation values for each observable at the respective
    // Hamiltonian operator in the batch.
    std::vector<double> exp_val_x, exp_val_y, exp_val_z;
    for (auto &exp_vals : evolution_result.expectation_values.value()) {
      exp_val_x.push_back(exp_vals[0].expectation());
      exp_val_y.push_back(exp_vals[1].expectation());
      exp_val_z.push_back(exp_vals[2].expectation());
    }

    // Append the results to the respective lists.
    all_exp_val_x.push_back(exp_val_x);
    all_exp_val_y.push_back(exp_val_y);
    all_exp_val_z.push_back(exp_val_z);
  }
  // [End Batch Results]

  // [Begin Batch Size]

  // Run the batch simulation with a maximum batch size of 2.
  // This means that the evolution will be performed in batches of 2 Hamiltonian
  // operators at a time, which can be useful for memory management or
  // performance tuning.
  auto results = cudaq::evolve(
      hamiltonians, dimensions, schedule, initial_states, integrator, {},
      {cudaq::spin_op::x(0), cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
      cudaq::IntermediateResultSave::ExpectationValue, /*max_batch_size=*/2);
  // [End Batch Size]

  return 0;
}
