/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// $nvq++ --target dynamics dynamics.cpp -o dynamics && ./dynamics
// ```

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
  auto hamiltonian = spin_op(0.5 * omega_z * spin_op::z(0));

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

  hamiltonian += mod_func * spin_op::x(0) * omega_x;
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
  auto observables = {cudaq::spin_op::x(0), cudaq::spin_op::y(0),
                      cudaq::spin_op::z(0)};

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
      (omega_a / 2.0) * spin_op::z(0) +
      (Omega / 2.0) * (boson_op::annihilate(1) * spin_op::plus(0) +
                       boson_op::create(1) * spin_op::minus(0));
  // [End Jaynes-Cummings]

  // [Begin Hamiltonian]
  // Hamiltonian with driving frequency
  double omega = M_PI;
  auto H0 = spin_op::z(0);
  auto H1 = spin_op::x(0);
  auto func = [omega](double t) { return std::cos(omega * t); };
  auto mod_func = [omega](double t) -> double { return std::cos(omega * t); };
  auto driven_hamiltonian = H0 + mod_func * H1;
  // [End Hamiltonian]

  // [Begin DefineOp]

  auto displacement_matrix =
      [](const std::vector<int> &dimensions,
         const cudaq::parameter_map &parameters) -> cudaq::complex_matrix {
    // Returns the displacement operator matrix.
    //  Args:
    //   - displacement: Amplitude of the displacement operator.
    // See also https://en.wikipedia.org/wiki/Displacement_operator.
    std::size_t dimension = dimensions[0];
    auto entry = parameters.find("displacement");
    if (entry == parameters.end())
      throw std::runtime_error("missing value for parameter 'displacement'");
    auto displacement_amplitude = entry->second;
    auto create = cudaq::complex_matrix(dimension, dimension);
    auto annihilate = cudaq::complex_matrix(dimension, dimension);
    for (std::size_t i = 0; i + 1 < dimension; i++) {
      create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1));
      annihilate[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1));
    }
    auto term1 = displacement_amplitude * create;
    auto term2 = std::conj(displacement_amplitude) * annihilate;
    return (term1 - term2).exponential();
  };

  cudaq::matrix_handler::define("displace_op", {-1}, displacement_matrix);

  // Instantiate a displacement operator acting on the given degree of freedom.
  auto displacement = [](int degree) {
    return cudaq::matrix_handler::instantiate("displace_op", {degree});
  };
  // [End DefineOp]

  {
    // [Begin Schedule1]

    // Define a system consisting of a single degree of freedom (0) with
    // dimension 3.
    cudaq::dimension_map system_dimensions{{0, 3}};
    auto system_operator = displacement(0);

    // Define the time dependency of the system operator as a schedule that
    // linearly increases the displacement parameter from 0 to 1.
    cudaq::schedule time_dependence(cudaq::linspace(0, 1, 100),
                                    {"displacement"});
    const std::vector<std::complex<double>> state_vec(3, 1.0 / std::sqrt(3.0));
    auto initial_state = cudaq::state::from_data(state_vec);
    cudaq::integrators::runge_kutta integrator(4, 0.01);
    // Simulate the evolution of the system under this time dependent operator.
    cudaq::evolve(system_operator, system_dimensions, time_dependence,
                  initial_state, integrator);
    // [End Schedule1]
  }
  {
    cudaq::dimension_map system_dimensions{{0, 3}};
    cudaq::integrators::runge_kutta integrator(4, 0.1);
    const std::vector<std::complex<double>> state_vec(3, 1.0 / std::sqrt(3.0));
    auto initial_state = cudaq::state::from_data(state_vec);
    // [Begin Schedule2]
    auto system_operator = displacement(0) + cudaq::matrix_op::squeeze(0);

    // Define a schedule such that displacement amplitude increases linearly in
    // time but the squeezing amplitude decreases, that is follows the inverse
    // schedule. def parameter_values(time_steps):
    auto parameter_values = [](const std::vector<double> &time_steps) {
      auto compute_value = [time_steps](const std::string &param_name,
                                        const std::complex<double> &step) {
        int step_idx = (int)step.real();
        if (param_name == "displacement")
          return time_steps[step_idx];
        if (param_name == "squeezing")
          return time_steps[time_steps.size() - (step_idx + 1)];

        throw std::runtime_error("value for parameter " + param_name +
                                 " undefined");
      };

      std::vector<std::complex<double>> steps;
      for (int i = 0; i < time_steps.size(); ++i)
        steps.emplace_back(i);
      return cudaq::schedule(steps, {"displacement", "squeezing"},
                             compute_value);
    };

    auto time_dependence = parameter_values(cudaq::linspace(0, 1, 100));
    cudaq::evolve(system_operator, system_dimensions, time_dependence,
                  initial_state, integrator);
    // [End Schedule2]
  }
  // Multi-qubit example
  int N = 4;
  double g = 1.0;
  dimensions.clear();
  for (int i = 0; i < N; i++)
    dimensions[i] = 2;

  auto H_multi = cudaq::sum_op<cudaq::spin_op>::empty();
  for (int i = 0; i < N; i++) {
    H_multi += 2.0 * M_PI * spin_op::x(i);
    H_multi += 2.0 * M_PI * spin_op::y(i);
  }

  for (int i = 0; i < N - 1; i++) {
    H_multi += 2.0 * M_PI * g * spin_op::x(i) * spin_op::x(i + 1);
    H_multi += 2.0 * M_PI * g * spin_op::y(i) * spin_op::z(i + 1);
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
