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
  std::vector<std::complex<double>> initial_state_vec = {1.0, 0.0};
  auto psi0 = cudaq::state::from_data(initial_state_vec);

  // Create a schedule of time steps from 0 to 10 with 101 points
  std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
  cudaq::schedule schedule(steps);

  // Runge-`Kutta` integrator with a time step of 0.01 and order 4
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  // Run the simulation without collapse operators (ideal evolution)
  auto evolve_result =
      cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator, {},
                    {cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
                    cudaq::IntermediateResultSave::ExpectationValue);

  constexpr double decay_rate = 0.05;
  auto collapse_operator = std::sqrt(decay_rate) * cudaq::spin_op::x(0);

  // Evolve with collapse operators
  cudaq::evolve_result evolve_result_decay = cudaq::evolve(
      hamiltonian, dimensions, schedule, psi0, integrator, {collapse_operator},
      {cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
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

  auto ideal_result0 = get_expectation(0, evolve_result);
  auto ideal_result1 = get_expectation(1, evolve_result);
  auto decay_result0 = get_expectation(0, evolve_result_decay);
  auto decay_result1 = get_expectation(1, evolve_result_decay);

  export_csv("qubit_dynamics_ideal_result.csv", "time", steps, "sigma_y",
             ideal_result0, "sigma_z", ideal_result1);
  export_csv("qubit_dynamics_decay_result.csv", "time", steps, "sigma_y",
             decay_result0, "sigma_z", decay_result1);

  std::cout << "Results exported to qubit_dynamics_ideal_result.csv and "
               "qubit_dynamics_decay_result.csv"
            << std::endl;

  return 0;
}
