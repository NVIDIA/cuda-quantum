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
  // Qubit resonant frequency
  double omega_z = 10.0 * 2 * M_PI;
  // Transverse driving term
  double omega_x = 2 * M_PI;
  // Driving frequency
  double omega_drive = 0.99 * omega_z;

  auto hamiltonian_t = 0.5 * omega_z * cudaq::spin_operator::z(0);
  auto mod_func =
      [omega_drive](
          const std::unordered_map<std::string, std::complex<double>> &params)
      -> std::complex<double> {
    auto it = params.find("t");
    if (it != params.end()) {
      double t = it->second.real();
      return std::cos(omega_drive * t);
    }
    return 0.0;
  };
  cudaq::operator_sum<cudaq::spin_operator> hamiltonian =
      hamiltonian_t +
      omega_x * cudaq::scalar_operator(mod_func) * cudaq::spin_operator::x(0);

  std::map<int, int> dimensions = {{0, 2}};

  std::vector<std::complex<double>> initial_state_vec = {1.0, 0.0, 0.0, 0.0};
  auto rho0 = cudaq::state::from_data(initial_state_vec);

  double t_final = M_PI / omega_x;
  double dt = 2.0 * M_PI / omega_drive / 100.0;
  int num_steps = static_cast<int>(std::ceil(t_final / dt)) + 1;
  auto steps = cudaq::linspace(0, t_final, num_steps);
  cudaq::Schedule schedule(steps, {"t"});

  std::shared_ptr<cudaq::runge_kutta> integrator =
      std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.001;
  integrator->order = 4;

  cudaq::product_operator<cudaq::matrix_operator> spin_op_x_t =
      cudaq::spin_operator::x(0);
  cudaq::operator_sum<cudaq::matrix_operator> spin_op_x(spin_op_x_t);
  cudaq::product_operator<cudaq::matrix_operator> spin_op_y_t =
      cudaq::spin_operator::y(0);
  cudaq::operator_sum<cudaq::matrix_operator> spin_op_y(spin_op_y_t);
  cudaq::product_operator<cudaq::matrix_operator> spin_op_z_t =
      cudaq::spin_operator::z(0);
  cudaq::operator_sum<cudaq::matrix_operator> spin_op_z(spin_op_z_t);
  auto observables = {spin_op_x, spin_op_y, spin_op_z};

  // Simulation without decoherence
  auto evolve_result = cudaq::evolve(hamiltonian, dimensions, schedule, rho0,
                                     integrator, {}, observables, true);

  // Simulation with decoherence
  double gamma_sm = 4.0;
  double gamma_sz = 1.0;

  auto evolve_result_decay = cudaq::evolve(
      hamiltonian, dimensions, schedule, rho0, integrator,
      {std::sqrt(gamma_sm) * spin_op_x, std::sqrt(gamma_sz) * spin_op_z},
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

  // For the ideal evolution
  auto ideal_result_x = get_expectation(0, evolve_result);
  auto ideal_result_y = get_expectation(1, evolve_result);
  auto ideal_result_z = get_expectation(2, evolve_result);

  // For the decoherence evolution
  auto decoherence_result_x = get_expectation(0, evolve_result_decay);
  auto decoherence_result_y = get_expectation(1, evolve_result_decay);
  auto decoherence_result_z = get_expectation(2, evolve_result_decay);

  // Export the results to a CSV file
  export_csv("qubit_control_ideal_result.csv", "t", steps, "sigma_x",
             ideal_result_x, "sigma_y", ideal_result_y, "sigma_z",
             ideal_result_z);
  export_csv("qubit_control_decoherence_result.csv", "t", steps, "sigma_x",
             decoherence_result_x, "sigma_y", decoherence_result_y, "sigma_z",
             decoherence_result_z);

  std::cout << "Results exported to qubit_control_ideal_result.csv and "
               "qubit_control_decoherence_result.csv"
            << std::endl;

  return 0;
}
