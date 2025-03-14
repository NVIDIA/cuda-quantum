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
#include <cmath>
#include <complex>
#include <cudaq.h>
#include <iostream>
#include <vector>

using namespace cudaq;

int main() {
  // [Begin Transmon]
  // Parameters
  double omega_z = 6.5;
  double omega_x = 4.0;
  double omega_d = 0.5;

  // Qubit Hamiltonian
  auto hamiltonian = sum_op<product_op>(0.5 * omega_z * spin_handler::z(0));

  // Time dependent modulation
  auto mod_func =
      [omega_d](const parameter_map &params) -> std::complex<double> {
    auto it = params.find("t");
    if (it != params.end()) {
      double t = it->second.real();
      const auto result = std::cos(omega_d * t);
      return result;
    }
    throw std::runtime_error("Cannot find the time parameter.");
  };

  hamiltonian += mod_func * spin_handler::x(0) * omega_x;
  // [End Transmon]
  // [Begin Evolve]
  double t_final = 1.0;
  int n_steps = 100;

  // Define dimensions of subsystem (single two-level system)
  std::map<int, int> dimensions = {{0, 2}};

  // Initial state (ground state density matrix)
  auto rho0 = std::vector<std::complex<double>>{1.0, 0.0};

  // Schedule of time steps
  std::vector<double> steps(n_steps);
  for (int i = 0; i < n_steps; i++)
    steps[i] = i * t_final / (n_steps - 1);

  schedule schedule(steps, {"t"});

  // Numerical integrator
  // Here we choose a Runge-`Kutta` method for time evolution.
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  // Observables to track
  auto observables = {cudaq::spin_handler::x(0), cudaq::spin_handler::y(0),
                      cudaq::spin_handler::z(0)};

  // Run simulation
  // We evolve the system under the defined Hamiltonian. No collapsed operators
  // are provided (closed system evolution). The evolution returns expectation
  // values for all defined observables at each time step.
  auto evolution_result = cudaq::evolve(hamiltonian, dimensions, schedule, rho0,
                                        integrator, {}, observables, true);
  // [End Evolve]
  // [Begin Print]
  // Extract and print results
  for (size_t i = 0; i < steps.size(); i++) {
    double ex = evolution_result.expectation_values()[0][i].expectation();
    double ey = evolution_result.expectation_values()[1][i].expectation();
    double ez = evolution_result.expectation_values()[2][i].expectation();
    std::cout << steps[i] << " " << ex << " " << ey << " " << ez << "\n";
  }
  // [End Print]

  cudaq::mpi::initialize();

  // Jaynes-Cummings Hamiltonian
  double omega_c = 6.0 * M_PI;
  double omega_a = 4.0 * M_PI;
  double Omega = 0.5;

  // [Begin Jaynes-Cummings]
  // Jaynes-Cummings Hamiltonian
  auto jc_hamiltonian =
      omega_c * boson_op::create(1) * boson_op::annihilate(1) +
      (omega_a / 2.0) * spin_handler::z(0) +
      (Omega / 2.0) * (boson_op::annihilate(1) * spin_handler::plus(0) +
                       boson_op::create(1) * spin_handler::minus(0));
  // [End Jaynes-Cummings]

  // [Begin Hamiltonian]
  // Hamiltonian with driving frequency
  double omega = M_PI;
  auto H0 = spin_handler::z(0);
  auto H1 = spin_handler::x(0);
  auto func = [omega](double t) { return std::cos(omega * t); };
  auto mod_func = [omega](double t) -> double { return std::cos(omega * t); };
  auto driven_hamiltonian = H0 + mod_func * H1;
  // [End Hamiltonian]

  // Multi-qubit example
  int N = 4;
  double g = 1.0;
  dimensions.clear();
  for (int i = 0; i < N; i++)
    dimensions[i] = 2;

  auto H_multi = cudaq::sum_op<cudaq::spin_handler>::empty();
  for (int i = 0; i < N; i++) {
    H_multi += 2.0 * M_PI * spin_handler::x(i);
    H_multi += 2.0 * M_PI * spin_handler::y(i);
  }

  for (int i = 0; i < N - 1; i++) {
    H_multi += 2.0 * M_PI * g * spin_handler::x(i) * spin_handler::x(i + 1);
    H_multi += 2.0 * M_PI * g * spin_handler::y(i) * spin_handler::z(i + 1);
  }

  std::vector<double> multi_steps(200);
  for (int i = 0; i < 200; i++)
    multi_steps[i] = i * (1.0 / 199);

  schedule multi_schedule(multi_steps, {"time"});

  // Evolve multi-qubit system
  auto psi0 = std::complex<double>(1.0, 0.0);
  auto multi_result = evolve(H_multi, dimensions, multi_schedule, psi0,
                             integrator, {}, {}, true);

  return 0;
}