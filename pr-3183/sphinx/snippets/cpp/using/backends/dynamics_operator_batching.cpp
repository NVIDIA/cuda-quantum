/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
  // [Begin Operator Batching]
  // Dimensions of sub-system
  cudaq::dimension_map dimensions = {{0, 2}};
  // Qubit resonant frequency
  const double omega_z = 10.0 * 2 * M_PI;
  // Transverse driving term
  const double omega_x = 2 * M_PI;
  // Harmonic driving frequency (sweeping in the +/- 10% range around the
  // resonant frequency).
  const auto omega_drive = cudaq::linspace(0.9 * omega_z, 1.1 * omega_z, 16);
  const auto zero_state =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  // List of Hamiltonians to be batched together
  std::vector<cudaq::spin_op> hamiltonians;
  std::vector<cudaq::state> initial_states;

  for (const auto &omega : omega_drive) {
    auto mod_func =
        [omega](const cudaq::parameter_map &params) -> std::complex<double> {
      auto it = params.find("t");
      if (it != params.end()) {
        double t = it->second.real();
        return std::cos(omega * t);
      }
      throw std::runtime_error("Cannot find the time parameter.");
    };

    // Add the Hamiltonian for each drive frequency to the batch.
    hamiltonians.emplace_back(0.5 * omega_z * cudaq::spin_op::z(0) +
                              mod_func * cudaq::spin_op::x(0) * omega_x);

    // The qubit starts in the |0> state.
    initial_states.emplace_back(zero_state);
  }

  // Schedule of time steps
  const std::vector<double> steps = cudaq::linspace(0.0, 0.5, 5000);
  // The schedule carries the time parameter `labelled` `t`, which is used by
  // the callback.
  cudaq::schedule schedule(steps, {"t"});

  // A default Runge-`Kutta` integrator (4`th` order) with time step `dt`
  // depending on the schedule.
  cudaq::integrators::runge_kutta integrator;

  // Run the batch simulation.
  auto evolve_result = cudaq::evolve(
      hamiltonians, dimensions, schedule, initial_states, integrator, {},
      {cudaq::spin_op::x(0), cudaq::spin_op::y(0), cudaq::spin_op::z(0)},
      cudaq::IntermediateResultSave::ExpectationValue);

  // [End Operator Batching]

  return 0;
}
