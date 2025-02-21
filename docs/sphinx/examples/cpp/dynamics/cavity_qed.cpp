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

  // System dimensions:
  // subsystem 0 (atom) has 2 levels
  // subsystem 1 (cavity) has 10 levels
  std::map<int, int> dimensions{{0, 2}, {1, 10}};

  // For the cavity subsystem 1
  auto a = cudaq::boson_operator::annihilate(1);
  auto a_dag = cudaq::boson_operator::create(1);

  // For the atom subsystem 0
  auto sm = cudaq::boson_operator::annihilate(0);
  auto sm_dag = cudaq::boson_operator::create(0);

  auto atom_occ_op = cudaq::matrix_operator::number(0);
  auto cavity_occ_op = cudaq::matrix_operator::number(1);

  auto hamiltonian = (2 * M_PI * cavity_occ_op) + (2 * M_PI * atom_occ_op) +
                     (2 * M_PI * 0.25 * (sm * a_dag + sm_dag * a));

  // Build the initial state
  // Atom (sub-system 0) in ground state
  // Cavity (sub-system 1) has 5 photons (Fock space)
  const int num_photons = 5;
  std::vector<std::complex<double>> initial_state_vec(20, 0.0);
  initial_state_vec[dimensions[0] * num_photons] = 1;

  // Define a time evolution schedule
  const int num_steps = 201;
  auto steps = cudaq::linspace(0.0, 10.0, num_steps);
  cudaq::Schedule schedule(steps);

  // Create a CUDA quantum state
  auto rho0 = cudaq::state::from_data(initial_state_vec);

  std::shared_ptr<cudaq::runge_kutta> integrator =
      std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.01;
  integrator->order = 4;

  // Evolve without collapse operators
  cudaq::evolve_result evolve_result =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator, {},
                    {cavity_occ_op, atom_occ_op}, true);

  constexpr double decay_rate = 0.1;
  auto collapse_operator = std::sqrt(decay_rate) * a;
  // Evolve with collapse operators
  cudaq::evolve_result evolve_result_decay =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator,
                    {collapse_operator}, {cavity_occ_op, atom_occ_op}, true);

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

  auto ideal_result0 = get_expectation(0, evolve_result);
  auto ideal_result1 = get_expectation(1, evolve_result);
  auto decay_result0 = get_expectation(0, evolve_result_decay);
  auto decay_result1 = get_expectation(1, evolve_result_decay);

  export_csv("cavity_qed_ideal_result.csv", "time", steps,
             "cavity_photon_number", ideal_result0,
             "atom_excitation_probability", ideal_result1);
  export_csv("cavity_qed_decay_result.csv", "time", steps,
             "cavity_photon_number", decay_result0,
             "atom_excitation_probability", decay_result1);

  std::cout << "Simulation complete. The results are saved in ideal_result.csv "
               "and decay_result.csv files."
            << std::endl;
  return 0;
}