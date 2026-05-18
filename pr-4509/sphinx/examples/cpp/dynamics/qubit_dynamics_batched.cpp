/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ --target dynamics qubit_dynamics.cpp -o a.out && ./a.out
// ```

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "export_csv_helper.h"
#include <cudaq.h>

int main() {
  // Qubit `hamiltonian`: 2 * pi * 0.1 * sigma_x
  // Physically, this represents a qubit (a two-level system) driven by a weak
  // transverse field along the x-axis.
  auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);

  // Dimensions: one subsystem of dimension 2 (a two-level system).
  const cudaq::dimension_map dimensions = {{0, 2}};

  // Initial state: ground state
  std::vector<std::complex<double>> initial_state_zero = {1.0, 0.0};
  std::vector<std::complex<double>> initial_state_one = {0.0, 1.0};

  auto psi0 = cudaq::state::from_data(initial_state_zero);
  auto psi1 = cudaq::state::from_data(initial_state_one);

  // Create a schedule of time steps from 0 to 10 with 101 points
  std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
  cudaq::schedule schedule(steps);

  // Runge-`Kutta` integrator with a time step of 0.01 and order 4
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  // Run the simulation without collapse operators (ideal evolution)
  auto evolve_results =
      cudaq::evolve(hamiltonian, dimensions, schedule, {psi0, psi1}, integrator,
                    {}, {cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
                    cudaq::IntermediateResultSave::ExpectationValue);

  // Lambda to extract expectation values for a given observable index
  auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
    std::vector<double> expectations;

    auto all_exps = result.expectation_values.value();
    for (auto exp_vals : all_exps) {
      expectations.push_back((double)exp_vals[idx]);
    }
    return expectations;
  };

  auto result_state0_y = get_expectation(0, evolve_results[0]);
  auto result_state0_z = get_expectation(1, evolve_results[0]);
  auto result_state1_y = get_expectation(0, evolve_results[1]);
  auto result_state1_z = get_expectation(1, evolve_results[1]);

  export_csv("qubit_dynamics_state_0.csv", "time", steps, "sigma_y",
             result_state0_y, "sigma_z", result_state0_z);
  export_csv("qubit_dynamics_state_1.csv", "time", steps, "sigma_y",
             result_state1_y, "sigma_z", result_state1_z);

  std::cout << "Results exported to qubit_dynamics_state_0.csv and "
               "qubit_dynamics_state_1.csv"
            << std::endl;

  return 0;
}
