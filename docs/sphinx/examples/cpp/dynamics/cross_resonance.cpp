/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/dynamics_integrators.h"
#include "cudaq/operators.h"
#include "export_csv_helper.h"
#include <cudaq.h>

int main() {

  // Detuning between two qubits
  double delta = 100 * 2 * M_PI;
  // Static coupling between qubits
  double J = 7 * 2 * M_PI;
  // Spurious electromagnetic crosstalk
  double m_12 = 0.2;
  // Drive strength
  double Omega = 20 * 2 * M_PI;

  auto hamiltonian = (delta / 2.0) * cudaq::spin_operator::z(0) +
                     J * (cudaq::fermion_operator::annihilate(1) *
                              cudaq::fermion_operator::create(0) +
                          cudaq::fermion_operator::create(1) *
                              cudaq::fermion_operator::annihilate(0)) +
                     Omega * cudaq::spin_operator::x(0) +
                     m_12 * Omega * cudaq::spin_operator::x(1);

  std::map<int, int> dimensions{{0, 2}, {1, 2}};

  // Build the initial state
  std::vector<std::complex<double>> psi00_data = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  std::vector<std::complex<double>> psi10_data = {
      {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  // Two initial state vectors for the 2-qubit system (dimension 4)
  // psi_00 corresponds to |00> and psi_10 corresponds to |10>
  auto psi_00 = cudaq::state::from_data(psi00_data);
  auto psi_10 = cudaq::state::from_data(psi10_data);

  // Create a schedule of time steps
  const int num_steps = 1001;
  const double t0 = 0.0, t1 = 1.0;
  std::vector<double> steps(num_steps);
  double dt = (t1 - t0) / (num_steps - 1);
  for (int i = 0; i < num_steps; ++i) {
    steps[i] = t0 + i * dt;
  }
  std::vector<std::string> labels = {"time"};
  cudaq::Schedule schedule(steps, labels);

  std::shared_ptr<cudaq::runge_kutta> integrator =
      std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.01;
  integrator->order = 1;

  auto observables = {cudaq::spin_operator::x(0), cudaq::spin_operator::y(0),
                      cudaq::spin_operator::z(0), cudaq::spin_operator::x(1),
                      cudaq::spin_operator::y(1), cudaq::spin_operator::z(1)};

  // Evolution for initial state |00>
  auto evolution_result_00 =
      cudaq::evolve(hamiltonian, dimensions, schedule, psi_00, integrator, {},
                    observables, true);

  // Evolution for initial state |10>
  auto evolution_result_10 =
      cudaq::evolve(hamiltonian, dimensions, schedule, psi_10, integrator, {},
                    observables, true);

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

  // For the two evolutions, extract the six observable trajectories.
  auto result_00_0 = get_expectation(0, evolution_result_00);
  auto result_00_1 = get_expectation(1, evolution_result_00);
  auto result_00_2 = get_expectation(2, evolution_result_00);
  auto result_00_3 = get_expectation(3, evolution_result_00);
  auto result_00_4 = get_expectation(4, evolution_result_00);
  auto result_00_5 = get_expectation(5, evolution_result_00);

  auto result_10_0 = get_expectation(0, evolution_result_10);
  auto result_10_1 = get_expectation(1, evolution_result_10);
  auto result_10_2 = get_expectation(2, evolution_result_10);
  auto result_10_3 = get_expectation(3, evolution_result_10);
  auto result_10_4 = get_expectation(4, evolution_result_10);
  auto result_10_5 = get_expectation(5, evolution_result_10);

  // Export the results to a CSV file
  export_csv("cross_resonance_00.csv", "time", steps, "sigma_x_0", result_00_0,
             "sigma_y_0", result_00_1, "sigma_z_0", result_00_2, "sigma_x_1",
             result_00_3, "sigma_y_1", result_00_4, "sigma_z_1", result_00_5);

  export_csv("cross_resonance_10.csv", "time", steps, "sigma_x_0", result_10_0,
             "sigma_y_0", result_10_1, "sigma_z_0", result_10_2, "sigma_x_1",
             result_10_3, "sigma_y_1", result_10_4, "sigma_z_1", result_10_5);

  std::cout
      << "Simulation complete. The results are saved in cross_resonance_00.csv "
         "and cross_resonance_10.csv files."
      << std::endl;
  return 0;
}