/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ --target dynamics heisenberg_model.cpp -o a.out && ./a.out
// ```

#include "cudaq/algorithms/evolve.h"
#include "cudaq/dynamics_integrators.h"
#include "cudaq/operators.h"
#include "export_csv_helper.h"
#include <cudaq.h>

int main() {

  // Set up a 9-spin chain, where each spin is a two-level system.
  const int num_spins = 9;
  std::map<int, int> dimensions;
  for (int i = 0; i < num_spins; i++) {
    dimensions[i] = 2; // Each spin (site) has dimension 2.
  }

  // Initial state
  // Prepare an initial state where the spins are arranged in a staggered
  // configuration. Even indices get the value '0' and odd indices get '1'. For
  // example, for 9 spins: spins: 0 1 0 1 0 1 0 1 0
  std::string spin_state;
  for (int i = 0; i < num_spins; i++) {
    spin_state.push_back((i % 2 == 0) ? '0' : '1');
  }

  // Convert the binary string to an integer index
  // In the Hilbert space of 9 spins (size 2^9 = 512), this index corresponds to
  // the state |0 1 0 1 0 1 0 1 0>
  int initial_state_index = std::stoi(spin_state, nullptr, 2);

  // Build the staggered magnetization operator
  // The staggered magnetization operator is used to measure antiferromagnetic
  // order. It is defined as a sum over all spins of the Z operator, alternating
  // in sign. For even sites, we add sz; for odd sites, we subtract sz.
  auto staggered_magnetization_t = cudaq::matrix_operator::empty();
  for (int i = 0; i < num_spins; i++) {
    auto sz = cudaq::spin_operator::z(i);
    if (i % 2 == 0) {
      staggered_magnetization_t += sz;
    } else {
      staggered_magnetization_t -= sz;
    }
  }

  // Normalize the number of spins so that the observable is intensive.
  auto stagged_magnetization_op =
      (1 / static_cast<double>(num_spins)) * staggered_magnetization_t;

  // Each entry will associate a value of g (the anisotropy in the Z coupling)
  // with its corresponding time-series of expectation values of the staggered
  // magnetization.
  std::vector<std::pair<double, std::vector<double>>> observe_results;

  // Simulate the dynamics over 1000 time steps spanning from time 0 to 5.
  const int num_steps = 1000;
  std::vector<std::complex<double>> steps;
  for (double t : cudaq::linspace(0.0, 5.0, num_steps)) {
    steps.emplace_back(t, 0.0);
  }

  // For three different values of g, which sets the strength of the Z-Z
  // interaction: g = 0.0 (isotropic in the XY plane), 0.25, and 4.0 (strongly
  // anisotropy).
  std::vector<double> g_values = {0.0, 0.25, 4.0};

  for (auto g : g_values) {
    // Set the coupling strengths:
    // Jx and Jy are set to 1.0 (coupling along X and Y axes), while Jz is set
    // to the current g value (coupling along the Z axis).
    double Jx = 1.0, Jy = 1.0, Jz = g;

    // The Hamiltonian is built from the nearest-neighbor interactions:
    // H = H + Jx * Sx_i * Sx_{i+1}
    // H = H + Jy * Sy_i * Sy_{i+1}
    // H = H + Jz * Sz_i * Sz_{i+1}
    // This is a form of the anisotropic Heisenberg (or XYZ) model.
    auto hamiltonian = cudaq::spin_operator::empty();
    for (int i = 0; i < num_spins - 1; i++) {
      hamiltonian = hamiltonian + Jx * cudaq::spin_operator::x(i) *
                                      cudaq::spin_operator::x(i + 1);
      hamiltonian = hamiltonian + Jy * cudaq::spin_operator::y(i) *
                                      cudaq::spin_operator::y(i + 1);
      hamiltonian = hamiltonian + Jz * cudaq::spin_operator::z(i) *
                                      cudaq::spin_operator::z(i + 1);
    }

    // Initial state vector
    // For a 9-spin system, the Hilbert space dimension is 2^9 = 512.
    // Initialize the state as a vector with all zeros except for a 1 at the
    // index corresponding to our staggered state.
    const int state_size = 1 << num_spins;
    std::vector<std::complex<double>> psi0_data(state_size, {0.0, 0.0});
    psi0_data[initial_state_index] = {1.0, 0.0};
    auto psi0 = cudaq::state::from_data(psi0_data);

    // The schedule is built using the time steps array.
    cudaq::Schedule schedule(
        steps, {"time"},
        [](const std::string &, const std::complex<double> &val) {
          return val;
        });

    // Use a Runge-Kutta integrator (4th order) with a small time step dt =
    // 0.001.
    std::shared_ptr<cudaq::RungeKuttaIntegrator> integrator =
        std::make_shared<cudaq::RungeKuttaIntegrator>();
    integrator->dt = 0.001;
    integrator->order = 4;

    // Evolve the initial state psi0 under the Hamiltonian, using the specified
    // schedule and integrator. No collapse operators are included (closed
    // system evolution). Measure the expectation value of the staggered
    // magnetization operator at each time step.
    auto evolve_result =
        cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator, {},
                      {stagged_magnetization_op}, true);

    // Lambda to extract expectation values for a given observable index
    auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
      std::vector<double> expectations;

      auto all_exps = result.get_expectation_values();
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

  // The CSV file "heisenberg_model.csv" will contain column with:
  //    - The time steps
  //    - The expectation values of the staggered magnetization for each g value
  //    (labeled g_0, g_0.25, g_4).
  export_csv("heisenberg_model_result.csv", "time", steps, "g_0",
             observe_results[0].second, "g_0.25", observe_results[1].second,
             "g_4", observe_results[2].second);

  std::cout << "Simulation complete. The results are saved in "
               "heisenberg_model_result.csv file."
            << std::endl;
  return 0;
}