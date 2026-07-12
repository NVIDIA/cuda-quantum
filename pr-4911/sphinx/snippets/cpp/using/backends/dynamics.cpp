/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include <cmath>
#include <complex>
#include <cudaq.h>
#include <iostream>
#include <vector>

using namespace cudaq;

int main() {
  {
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
    cudaq::dimension_map dimensions = {{0, 2}};

    // Initial state (ground state)
    auto psi0 =
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});

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
    // We evolve the system under the defined Hamiltonian. No collapsed
    // operators are provided (closed system evolution). The evolution returns
    // expectation values for all defined observables at each time step.
    auto evolution_result =
        cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator, {},
                      observables, cudaq::IntermediateResultSave::All);
    // [End Evolve]
    // [Begin Print]
    // Extract and print results
    for (size_t i = 0; i < steps.size(); i++) {
      double ex =
          evolution_result.expectation_values.value()[0][i].expectation();
      double ey =
          evolution_result.expectation_values.value()[1][i].expectation();
      double ez =
          evolution_result.expectation_values.value()[2][i].expectation();
      std::cout << steps[i] << " " << ex << " " << ey << " " << ez << "\n";
    }
    // [End Print]
  }

  {
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
    auto mod_func =
        [omega](const std::unordered_map<std::string, std::complex<double>>
                    &parameters) {
          auto entry = parameters.find("t");
          if (entry == parameters.end())
            throw std::runtime_error("Cannot find value of expected parameter");
          const auto t = entry->second.real();
          return std::cos(omega * t);
        };
    auto driven_hamiltonian = H0 + mod_func * H1;
    // [End Hamiltonian]
  }
  {
    // [Begin DefineOp]
    auto displacement_matrix =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>>
               &parameters) -> cudaq::complex_matrix {
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

    // Instantiate a displacement operator acting on the given degree of
    // freedom.
    auto displacement = [](std::size_t degree) {
      return cudaq::matrix_handler::instantiate("displace_op", {degree});
    };
    // [End DefineOp]

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

    // [Begin Schedule2]
    auto hamiltonian = displacement(0) + cudaq::matrix_op::squeeze(0);

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

    auto time_dependence_param = parameter_values(cudaq::linspace(0, 1, 100));
    cudaq::evolve(hamiltonian, system_dimensions, time_dependence_param,
                  initial_state, integrator);
    // [End Schedule2]
  }
  {
    // [Begin SuperOperator]
    const cudaq::dimension_map dims = {{0, 2}};
    cudaq::product_op<cudaq::matrix_handler> ham_ =
        2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
    cudaq::sum_op<cudaq::matrix_handler> ham(ham_);
    constexpr int numSteps = 10;
    cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, numSteps), {"t"});
    auto initialState =
        cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
    cudaq::integrators::runge_kutta integrator(1, 0.001);
    // Create a super-operator to evolve the system under the Schrödinger
    // equation `-iH * |psi>`, where `H` is the Hamiltonian.
    cudaq::super_op sup;
    // Apply `-iH * |psi>` superop
    sup +=
        cudaq::super_op::left_multiply(std::complex<double>(0.0, -1.0) * ham);
    auto result = cudaq::evolve(sup, dims, schedule, initialState, integrator,
                                {cudaq::spin_op::z(0)},
                                cudaq::IntermediateResultSave::All);
    // [End SuperOperator]
  }

  {
    auto H = spin_op::z(0) + spin_op::z(1);
    const cudaq::dimension_map dimensions = {{0, 2}, {1, 2}};
    cudaq::schedule schedule(cudaq::linspace(0.0, 1.0, 100), {"t"});
    cudaq::integrators::runge_kutta integrator(1, 0.001);
    // [Begin MPI]
    cudaq::mpi::initialize();
    // Initial state (expressed as an enum)
    auto psi0 = cudaq::InitialState::ZERO;

    // Run the simulation
    auto evolution_result =
        cudaq::evolve(H, dimensions, schedule, psi0, integrator);

    cudaq::mpi::finalize();
    // [End MPI]
  }
  return 0;
}
