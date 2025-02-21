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

  const int num_spins = 9;
  std::map<int, int> dimensions;
  for (int i = 0; i < num_spins; i++) {
    dimensions[i] = 2;
  }

  // Initial state: even indices: '0' and odd indices: '1'
  std::string spin_state;
  for (int i = 0; i < num_spins; i++) {
    spin_state.push_back((i % 2 == 0) ? '0' : '1');
  }

  // Convert the binary string to an integer index
  int initial_state_index = std::stoi(spin_state, nullptr, 2);

  // Build the staggered magnetization operator
  auto staggered_magnetization_t = cudaq::matrix_operator::empty();
  for (int i = 0; i < num_spins; i++) {
    auto sz = cudaq::spin_operator::z(i);
    if (i % 2 == 0) {
      staggered_magnetization_t += sz;
    } else {
      staggered_magnetization_t -= sz;
    }
  }
  auto stagged_magnetization_op =
      (1 / static_cast<double>(num_spins)) * staggered_magnetization_t;

  // Each entry will hold a value of g and its corresponding vector of
  // expectation values
  std::vector<std::pair<double, std::vector<double>>> observe_results;

  std::vector<double> g_values = {0.0, 0.25, 4.0};

  const int num_steps = 1000;
  std::vector<double> steps(num_steps);
  double t0 = 0.0, tf = 5.0;
  for (int i = 0; i < num_steps; i++) {
    steps[i] = t0 + i * (tf - t0) / (num_steps - 1);
  }

  cudaq::Schedule schedule(steps, {"time"});

  // Initial state vector
  const int state_size = 1 << num_spins;
  std::vector<std::complex<double>> psi0_data(state_size, {0.0, 0.0});
  psi0_data[initial_state_index] = {1.0, 0.0};
  auto psi0 = cudaq::state::from_data(psi0_data);

  for (auto g : g_values) {
    double Jx = 1.0, Jy = 1.0, Jz = g;

    auto hamiltonian = cudaq::spin_operator::empty();
    for (int i = 0; i < num_spins - 1; i++) {
      hamiltonian = hamiltonian + Jx * cudaq::spin_operator::x(i) *
                                      cudaq::spin_operator::x(i + 1);
      hamiltonian = hamiltonian + Jy * cudaq::spin_operator::y(i) *
                                      cudaq::spin_operator::y(i + 1);
      hamiltonian = hamiltonian + Jz * cudaq::spin_operator::z(i) *
                                      cudaq::spin_operator::z(i + 1);
    }

    std::shared_ptr<cudaq::runge_kutta> integrator =
        std::make_shared<cudaq::runge_kutta>();
    integrator->dt = 0.01;
    integrator->order = 4;

    auto evolve_result =
        cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator, {},
                      {stagged_magnetization_op}, true);

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

    observe_results.push_back({g, get_expectation(0, evolve_result)});
  }

  if (observe_results.size() != 3) {
    std::cerr << "Unexpected number of g values" << std::endl;
    return 1;
  }

  export_csv("heisenberg_model_result.csv", "time", steps, "g_0",
             observe_results[0].second, "g_0.25", observe_results[1].second,
             "g_4", observe_results[2].second);

  std::cout << "Simulation complete. The results are saved in "
               "heisenberg_model_result.csv file."
            << std::endl;
  return 0;
}