/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/dynamics_integrators.h"
#include "cudaq/base_integrator.h"
#include "cudaq/evolution.h"
#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <cudaq.h>

int main() {

  // System dimensions:
  // subsystem 0 (atom) has 2 levels
  // subsystem 1 (cavity) has 10 levels
  std::map<int, int> dimensions{{0, 2}, {1, 10}};

  // For the cavity subsystem 1
  cudaq::product_operator<cudaq::matrix_operator> a = cudaq::boson_operator::annihilate(1);
  cudaq::product_operator<cudaq::matrix_operator> a_dag = cudaq::boson_operator::create(1);

  // For the atom subsystem 0
  cudaq::product_operator<cudaq::matrix_operator> sm = cudaq::boson_operator::annihilate(0);
  cudaq::product_operator<cudaq::matrix_operator> sm_dag = cudaq::boson_operator::create(0);

  cudaq::product_operator<cudaq::matrix_operator> atom_occ_op_t =
      cudaq::matrix_operator::number(0);
  cudaq::operator_sum<cudaq::matrix_operator> atom_occ_op(atom_occ_op_t);

  cudaq::product_operator<cudaq::matrix_operator> cavity_occ_op_t =
      cudaq::matrix_operator::number(1);
  cudaq::operator_sum<cudaq::matrix_operator> cavity_occ_op(cavity_occ_op_t);

  auto hamiltonian = (2 * M_PI * cavity_occ_op) + (2 * M_PI * atom_occ_op) + 
                     (2 * M_PI * 0.25 * (sm * a_dag + sm_dag * a));

  // Build the initial state
  // For the atom, the density matrix in the ground state
  std::vector<std::complex<double>> initial_state_vec(20, 0.0);
  initial_state_vec[10] = 1;

  // Define a time evolution schedule
  const int num_steps = 21;
  cudaq::Schedule schedule(cudaq::linspace(0.0, 10.0, num_steps));

  // Create a CUDA quantum state from a density matrix
  auto rho0 = cudaq::state::from_data(initial_state_vec);

  std::shared_ptr<cudaq::runge_kutta> integrator = std::make_shared<cudaq::runge_kutta>();
  integrator->dt = 0.001;
  integrator->order = 4;

  // Evolve without collapse operators
  cudaq::evolve_result evolve_result =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator, {},
        {cavity_occ_op, atom_occ_op}, true);

  std::shared_ptr<cudaq::runge_kutta> integrator_1 = std::make_shared<cudaq::runge_kutta>();
  integrator_1->dt = 0.001;
  integrator_1->order = 4;

  constexpr double decay_rate = 0.1;
  auto collapse_operator = std::sqrt(decay_rate) * a;
  // Evolve with collapse operators
  cudaq::evolve_result evolve_result_decay =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator_1,
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

  std::cout << "Ideal result 0: ";
  for (auto val : ideal_result0) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  std::cout << "Ideal result 1: ";
  for (auto val : ideal_result1) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  std::cout << "Decay result 0: ";
  for (auto val : decay_result0) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  std::cout << "Decay result 1: ";
  for (auto val : decay_result1) {
    std::cout << val << " ";
  }
  std::cout << std::endl;

  std::cout << "Simulation complete. Plot saved to cavity_qed.png" << std::endl;
  return 0;
}