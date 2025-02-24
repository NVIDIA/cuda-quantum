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

  auto spin_plus = [](int degree) {
    return 0.5 *
           (cudaq::spin_operator::x(degree) +
            std::complex<double>(0.0, 1.0) * cudaq::spin_operator::y(degree));
  };

  auto spin_minus = [](int degree) {
    return 0.5 *
           (cudaq::spin_operator::x(degree) -
            std::complex<double>(0.0, 1.0) * cudaq::spin_operator::y(degree));
  };

  auto hamiltonian =
      (delta / 2.0) * cudaq::spin_operator::z(0) +
      J * (spin_minus(1) * spin_plus(0) + spin_plus(1) * spin_minus(0)) +
      Omega * cudaq::spin_operator::x(0) +
      m_12 * Omega * cudaq::spin_operator::x(1);

  std::map<int, int> dimensions{{0, 2}, {1, 2}};

  // Build the initial state
  std::vector<std::complex<double>> psi00_data = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  std::vector<std::complex<double>> psi10_data = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  // Two initial state vectors for the 2-qubit system (dimension 4)
  // psi_00 corresponds to |00> and psi_10 corresponds to |10>
  auto psi_00 = cudaq::state::from_data(psi00_data);
  auto psi_10 = cudaq::state::from_data(psi10_data);

  // Create a schedule of time steps
  const int num_steps = 1001;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::Schedule schedule(steps);

  std::shared_ptr<cudaq::runge_kutta> integrator =
      std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.0001;
  integrator->order = 4;

  auto observables = {cudaq::spin_operator::x(0), cudaq::spin_operator::y(0),
                      cudaq::spin_operator::z(0), cudaq::spin_operator::x(1),
                      cudaq::spin_operator::y(1), cudaq::spin_operator::z(1)};

  // Evolution with 2 initial states
  const auto evolution_results =
      cudaq::evolve(hamiltonian, dimensions, schedule, {psi_00, psi_10},
                    integrator, {}, observables, true);

  auto &evolution_result_00 = evolution_results[0];
  auto &evolution_result_10 = evolution_results[1];

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
  export_csv("cross_resonance_z.csv", "time", steps, "<Z1>_00", result_00_5,
             "<Z1>_10", result_10_5);
  export_csv("cross_resonance_y.csv", "time", steps, "<Y1>_00", result_00_4,
             "<Y1>_10", result_10_4);

  std::cout
      << "Simulation complete. The results are saved in cross_resonance_z.csv "
         "and cross_resonance_y.csv files."
      << std::endl;
  return 0;
}