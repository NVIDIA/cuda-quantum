/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include <cudaq.h>

int main() {
  // [Begin State Batching]

  // Qubit Hamiltonian
  auto hamiltonian = 2 * M_PI * 0.1 * cudaq::spin_op::x(0);

  // A single qubit with dimension 2.
  cudaq::dimension_map dimensions = {{0, 2}};

  // Initial states in the `SIC-POVM` set:
  // https://en.wikipedia.org/wiki/SIC-POVM
  auto psi_1 =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto psi_2 = cudaq::state::from_data(std::vector<std::complex<double>>{
      1.0 / std::sqrt(3.0), std::sqrt(2.0 / 3.0)});
  auto psi_3 = cudaq::state::from_data(std::vector<std::complex<double>>{
      1.0 / std::sqrt(3.0),
      std::sqrt(2.0 / 3.0) *
          std::exp(std::complex<double>{0.0, 1.0} * 2.0 * M_PI / 3.0)});
  auto psi_4 = cudaq::state::from_data(std::vector<std::complex<double>>{
      1.0 / std::sqrt(3.0),
      std::sqrt(2.0 / 3.0) *
          std::exp(std::complex<double>{0.0, 1.0} * 4.0 * M_PI / 3.0)});
  // We run the evolution for all the SIC state to determine the process
  // tomography.
  std::vector<cudaq::state> sic_states = {psi_1, psi_2, psi_3, psi_4};

  // Schedule of time steps.
  std::vector<double> steps = cudaq::linspace(0.0, 10.0, 101);
  cudaq::schedule schedule(steps);

  // A default Runge-`Kutta` integrator
  cudaq::integrators::runge_kutta integrator;

  // Run the batch simulation.
  auto evolve_results = cudaq::evolve(
      hamiltonian, dimensions, schedule, sic_states, integrator, {},
      {cudaq::spin_op::x(0), cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
      cudaq::IntermediateResultSave::ExpectationValue);

  // [End State Batching]

  return 0;
}
