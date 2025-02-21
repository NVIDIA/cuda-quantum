/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/base_integrator.h"
#include "cudaq/dynamics_integrators.h"
#include "cudaq/evolution.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include "export_csv_helper.h"
#include <cudaq.h>

int main() {
  // Qubit hamiltonian: 2 * pi * 0.1 * sigma_x
  cudaq::product_operator<cudaq::matrix_operator> ham =
      2.0 * M_PI * 0.1 * cudaq::spin_operator::x(0);
  cudaq::operator_sum<cudaq::matrix_operator> hamiltonian(ham);

  // Dimensions: one subsystem of dimension 2 (a two-level system)
  const std::map<int, int> dimensions = {{0, 2}};

  // Initial state (density matrix) of the system, given as a 2x2 matrix
  // Ground state: [ [1, 0],
  //                 [0, 0] ]
  std::vector<std::complex<double>> initial_state_vec = {1.0, 0.0, 0.0, 0.0};
  auto rho0 = cudaq::state::from_data(initial_state_vec);

  // Create a schedule of time steps from 0 to 10 with 101 points
  auto steps = cudaq::linspace(0.0, 10.0, 101);
  cudaq::Schedule schedule(steps, {"time"});

  // Runge-Kutta integrator with a time step of 0.001 and order 4
  std::shared_ptr<cudaq::runge_kutta> integrator =
      std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.001;
  integrator->order = 4;

  cudaq::product_operator<cudaq::matrix_operator> spin_op_y_t =
      cudaq::spin_operator::y(0);
  cudaq::operator_sum<cudaq::matrix_operator> spin_op_y(spin_op_y_t);
  cudaq::product_operator<cudaq::matrix_operator> spin_op_z_t =
      cudaq::spin_operator::z(0);
  cudaq::operator_sum<cudaq::matrix_operator> spin_op_z(spin_op_z_t);
  auto collapse_operators = {spin_op_y, spin_op_z};

  // Run the simulation without collapse operators (ideal evolution)
  auto evolve_result = cudaq::evolve(hamiltonian, dimensions, schedule, rho0,
                                     integrator, {}, collapse_operators, true);

  std::shared_ptr<cudaq::runge_kutta> integrator_1 =
      std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.001;
  integrator->order = 4;

  constexpr double decay_rate = 0.05;
  cudaq::product_operator<cudaq::matrix_operator> collapse_operator_t =
      std::sqrt(decay_rate) * cudaq::spin_operator::x(0);
  cudaq::operator_sum<cudaq::matrix_operator> collapse_operator(
      collapse_operator_t);

  // Evolve with collapse operators
  cudaq::evolve_result evolve_result_decay =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator_1,
                    {collapse_operator}, {collapse_operator}, true);

  // Lambda to extract expectation values for a given observable index
  auto get_expectation = [](int idx,
                            const auto &result) -> std::vector<double> {
    std::vector<double> expectations;

    auto all_exps = result.get_expectation_values().value();
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