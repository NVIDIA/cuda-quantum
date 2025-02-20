/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/dynamics_integrators.h"
#include "cudaq/evolution.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include "matplotlibcpp.h"
#include <cudaq.h>

namespace plt = matplotlibcpp;

int main() {

  cudaq::set_target_backend("dynamics");

  // Detuning between two qubits
  double delta = 100 * 2 * M_PI;
  // Static coupling between qubits
  double J = 7 * 2 * M_PI;
  // Spurious electromagnetic crosstalk
  double m_12 = 0.2;
  // Drive strength
  double Omega = 20 * 2 * M_PI;

  auto hamiltonian =
      (delta / 2.0) * cudaq::spin_operator::z(0) +
      J * (cudaq::spin_operator::minus(1) * cudaq::spin_operator::plus(0) +
           cudaq::spin_operator::plus(1) * cudaq::spin_operator::minus(0)) +
      Omega * cudaq::spin_operator::x(0) +
      m_12 * Omega * cudaq::spin_operator::x(1);

  std::map<int, int> dimensions{{0, 2}, {1, 2}};

  // Build the initial state
  cudaq::matrix_2 rho_mat({1.0, 0.0}, {0.0, 0.0});

  // Flatten the matrix
  std::vector<std::complex<double>> flat_rho;
  for (size_t j = 0; j < rho_mat.get_columns(); ++j) {
    for (size_t i = 0; i < rho_mat.get_rows(); ++i) {
      flat_rho.push_back(rho_mat[{i, j}]);
    }
  }

  cudaq::state_data rho_data = flat_rho;
  auto rho0 = cudaq::state::from_data(rho_data);

  // Two initial state vectors for the 2-qubit system (dimension 4)
  // psi_00 corresponds to |00> and psi_10 corresponds to |10>
  auto psi_00 = cudaq::state::from_data({1.0, 0.0, 0.0, 0.0});
  auto psi_10 = cudaq::state::from_data({0.0, 0.0, 1.0, 0.0});

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

  cudaq::runge_kutta integrator;
  integrator.dt = 0.001;
  integrator.order = 1;

  std::vector<cudaq::spin_operator> observables = {
      cudaq::spin_operator::x(0), cudaq::spin_operator::y(0),
      cudaq::spin_operator::z(0), cudaq::spin_operator::x(1),
      cudaq::spin_operator::y(1), cudaq::spin_operator::z(1)};

  // Evolution for initial state |00>
  auto evolution_result_00 =
      cudaq::evolve(hamiltonian, dimensions, schedule, psi_00, observables, {},
                    true, integrator);

  // Evolution for initial state |10>
  auto evolution_result_10 =
      cudaq::evolve(hamiltonian, dimensions, schedule, psi_10, observables, {},
                    true, integrator);

  auto get_result = [](int idx, const auto &result) -> std::vector<double> {
    std::vector<double> expectations;
    auto all_exps = result.get_expectation_values().value();
    for (const auto &exp_vals : all_exps) {
      expectations.push_back((double)exp_vals[idx]);
    }
    return expectations;
  };

  // For the two evolutions, extract the six observable trajectories.
  auto result_00_0 = get_result(0, evolution_result_00);
  auto result_00_1 = get_result(1, evolution_result_00);
  auto result_00_2 = get_result(2, evolution_result_00);
  auto result_00_3 = get_result(3, evolution_result_00);
  auto result_00_4 = get_result(4, evolution_result_00);
  auto result_00_5 = get_result(5, evolution_result_00);

  auto result_10_0 = get_result(0, evolution_result_10);
  auto result_10_1 = get_result(1, evolution_result_10);
  auto result_10_2 = get_result(2, evolution_result_10);
  auto result_10_3 = get_result(3, evolution_result_10);
  auto result_10_4 = get_result(4, evolution_result_10);
  auto result_10_5 = get_result(5, evolution_result_10);

  // Plot the results
  plt::figure_size(1000, 600);

  // Subplot 1: Expectation value <Z> for qubit 1.
  plt::subplot(1, 2, 1);
  plt::plot(steps, result_00_5, {{"label", "$|\\psi_0\\rangle=|00\\rangle$"}});
  plt::plot(steps, result_10_5, {{"label", "$|\\psi_0\\rangle=|10\\rangle$"}});
  plt::xlabel("Time");
  plt::ylabel("$\\langle Z_2 \\rangle$");
  plt::legend();

  // Subplot 2: Expectation value <Y> for qubit 1.
  plt::subplot(1, 2, 2);
  plt::plot(steps, result_00_4, {{"label", "$|\\psi_0\\rangle=|00\\rangle$"}});
  plt::plot(steps, result_10_4, {{"label", "$|\\psi_0\\rangle=|10\\rangle$"}});
  plt::xlabel("Time");
  plt::ylabel("$\\langle Y_2 \\rangle$");
  plt::legend();

  plt::save("cross_resonance.png");

  std::cout << "Simulation complete. Plot saved to cross_resonance.png"
            << std::endl;
  return 0;
}