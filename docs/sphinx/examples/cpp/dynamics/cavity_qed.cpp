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

  // System dimensions:
  // subsystem 0 (atom) has 2 levels
  // subsystem 1 (cavity) has 10 levels
  std::map<int, int> dimensions{{0, 2}, {1, 10}};

  // For the cavity subsystem 1
  auto a = cudaq::matrix_operator::annihilate(1);
  auto a_dag = cudaq::matrix_operator::create(1);

  // For the atom subsystem 0
  auto sm = cudaq::matrix_operator::annihilate(0);
  auto sm_dag = cudaq::matrix_operator::create(0);

  cudaq::product_operator<cudaq::matrix_operator> atom_occ_op_t =
      cudaq::matrix_operator::number(0);
  cudaq::operator_sum<cudaq::matrix_operator> atom_occ_op(atom_occ_op_t);

  cudaq::product_operator<cudaq::matrix_operator> cavity_occ_op_t =
      cudaq::matrix_operator::number(1);
  cudaq::operator_sum<cudaq::matrix_operator> cavity_occ_op(cavity_occ_op_t);

  auto hamiltonian = 2 * M_PI * atom_occ_op + 2 * M_PI * cavity_occ_op +
                     2 * M_PI * 0.25 * (sm * a_dag + sm_dag * a);

  // Build the initial state
  // For the atom, the density matrix in the ground state
  cudaq::matrix_2 qubit_state({1.0, 0.0}, {0.0, 0.0});

  // For the cavity, 1 10x10 matrix with a single photon number state at |5>
  cudaq::matrix_2 cavity_state = cudaq::matrix_2(10, 10);
  cavity_state[{5, 5}] = 1.0;

  // Compute the tensor (kronecker) product of the atom and cavity states
  cudaq::matrix_2 rho = qubit_state.kronecker_inplace(cavity_state);

  // Flatten the matrix
  std::vector<std::complex<double>> flat_rho;
  for (size_t j = 0; j < rho.get_columns(); ++j) {
    for (size_t i = 0; i < rho.get_rows(); ++i) {
      flat_rho.push_back(rho[{i, j}]);
    }
  }

  cudaq::state_data rho_data = flat_rho;

  // Create a CUDA quantum state from a density matrix
  auto rho0 = cudaq::state::from_data(rho_data);

  // Create time steps between 0 and 10
  const int num_steps = 201;
  const double t0 = 0.0, t1 = 10.0;
  std::vector<double> steps(num_steps);
  double dt = (t1 - t0) / (num_steps - 1);
  for (int i = 0; i < num_steps; ++i) {
    steps[i] = t0 + i * dt;
  }

  // Create a schedule for the time steps and a label for the time parameter
  std::vector<std::string> labels = {"time"};
  cudaq::Schedule schedule(steps, labels);

  cudaq::runge_kutta integrator;
  integrator.dt = 0.001;
  integrator.order = 1;

  // Evolve without collapse operators
  cudaq::evolve_result evolve_result =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator, {},
                    {hamiltonian}, true);

  constexpr double decay_rate = 0.1;
  auto collapse_operator = std::sqrt(decay_rate) * a;
  // Evolve with collapse operators
  cudaq::evolve_result evolve_result_decay =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator,
                    {collapse_operator}, {hamiltonian}, true);

  // Lambda to extract expectation values for a given observable index
  auto get_expectation = [](int idx,
                            const auto &result) -> std::vector<double> {
    std::vector<double> expectations;

    auto all_exps = result.get_expectation_values().value();
    for (const auto &exp_vals : all_exps) {
      expectations.push_back((double)exp_vals[idx]);
    }
    return expectations;
  };

  auto ideal_result0 = get_expectation(0, evolve_result);
  auto ideal_result1 = get_expectation(1, evolve_result);
  auto decay_result0 = get_expectation(0, evolve_result_decay);
  auto decay_result1 = get_expectation(1, evolve_result_decay);

  // Plot the results
  plt::figure_size(1000, 600);

  // Subplot 1: No decay
  plt::subplot(1, 2, 1);
  plt::plot(steps, ideal_result0, {{"label", "Cavity Photon Number"}});
  plt::plot(steps, ideal_result1, {{"label", "Atom Excitation Probability"}});
  plt::xlabel("Time");
  plt::ylabel("Expectation value");
  plt::legend();
  plt::title("No decay");

  // Subplot 2: With decay
  plt::subplot(1, 2, 2);
  plt::plot(steps, decay_result0, {{"label", "Cavity Photon Number"}});
  plt::plot(steps, decay_result1, {{"label", "Atom Excitation Probability"}});
  plt::xlabel("Time");
  plt::ylabel("Expectation value");
  plt::legend();
  plt::title("No decay");

  plt::save("cavity_qed.png");

  std::cout << "Simulation complete. Plot saved to cavity_qed.png" << std::endl;
  return 0;
}