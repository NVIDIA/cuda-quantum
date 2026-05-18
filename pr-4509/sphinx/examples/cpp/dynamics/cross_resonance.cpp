/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ --target dynamics cavity_qed.cpp -o a.out && ./a.out
// ```

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "export_csv_helper.h"
#include <cudaq.h>

int main() {

  // `delta` represents the detuning between the two qubits.
  // In physical terms, detuning is the energy difference (or frequency offset)
  // between qubit levels. Detuning term (in angular frequency units).
  double delta = 100 * 2 * M_PI;
  // `J` is the static coupling strength between the two qubits.
  // This terms facilitates energy exchange between the qubits, effectively
  // coupling their dynamics.
  double J = 7 * 2 * M_PI;
  // `m_12` models spurious electromagnetic `crosstalk`.
  // `Crosstalk` is an unwanted interaction , here represented as a fraction of
  // the drive strength applied to the second qubit.
  double m_12 = 0.2;
  // `Omega` is the drive strength applied to the qubits.
  // A driving field can induce transitions between qubit states.
  double Omega = 20 * 2 * M_PI;

  // For a spin-1/2 system, the raising operator S^+ and lowering operator S^-
  // are defined as: S^+ = 0.5 * (X + `iY`) and S^- = 0.5 * (X - `iY`) These
  // operators allow transitions between the spin states (|0> and |1>).
  auto spin_plus = [](int degree) {
    return 0.5 * (cudaq::spin_op::x(degree) +
                  std::complex<double>(0.0, 1.0) * cudaq::spin_op::y(degree));
  };

  auto spin_minus = [](int degree) {
    return 0.5 * (cudaq::spin_op::x(degree) -
                  std::complex<double>(0.0, 1.0) * cudaq::spin_op::y(degree));
  };

  // The Hamiltonian describes the energy and dynamics of our 2-qubit system.
  // It consist of several parts:
  // 1. Detuning term for qubit 0: (delta / 2) * Z. This sets the energy
  // splitting for qubit 0.
  // 2. Exchange interaction: J * (S^-_1 * S^+_0 + S^+_1 * S^-_0). This couples
  // the two qubits, enabling excitation transfer.
  // 3. Drive on qubit 0: Omega * X. A control field that drives transition in
  // qubit 0.
  // 4. `Crosstalk` drive on qubit 1: m_12 * Omega * X. A reduces drive on qubit
  // 1 due to electromagnetic `crosstalk`.
  auto hamiltonian =
      (delta / 2.0) * cudaq::spin_op::z(0) +
      J * (spin_minus(1) * spin_plus(0) + spin_plus(1) * spin_minus(0)) +
      Omega * cudaq::spin_op::x(0) + m_12 * Omega * cudaq::spin_op::x(1);

  // Each qubit is a 2-level system (dimension 2).
  // The composite system (two qubits) has a total Hilbert space dimension of 2
  // * 2 = 4.
  cudaq::dimension_map dimensions{{0, 2}, {1, 2}};

  // Build the initial state
  // psi_00 represents the state |00> (both qubits in the ground state).
  // psi_10 represents the state |10> (first qubit excited, second qubit in the
  // ground state).
  std::vector<std::complex<double>> psi00_data = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  std::vector<std::complex<double>> psi10_data = {
      {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};

  // Two initial state vectors for the 2-qubit system (dimension 4)
  auto psi_00 = cudaq::state::from_data(psi00_data);
  auto psi_10 = cudaq::state::from_data(psi10_data);

  // Create a list of time steps for the simulation.
  // Here we use 1001 points linearly spaced between time 0 and 1.
  // This schedule will be used to integrate the time evolution of the system.
  const int num_steps = 1001;
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, num_steps);
  cudaq::schedule schedule(steps);

  // Use Runge-`Kutta` integrator (4`th` order) to solve the time-dependent
  // evolution. `dt` is the integration time step, and `order` sets the accuracy
  // of the integrator method.
  cudaq::integrators::runge_kutta integrator(4, 0.0001);

  // The observables are the spin components along the x, y, and z directions
  // for both qubits. These observables will be measured during the evolution.
  auto observables = {cudaq::spin_op::x(0), cudaq::spin_op::y(0),
                      cudaq::spin_op::z(0), cudaq::spin_op::x(1),
                      cudaq::spin_op::y(1), cudaq::spin_op::z(1)};

  // Evolution with 2 initial states
  // We evolve the system under the defined Hamiltonian for both initial states
  // simultaneously. No collapsed operators are provided (closed system
  // evolution). The evolution returns expectation values for all defined
  // observables at each time step.
  auto evolution_results = cudaq::evolve(
      hamiltonian, dimensions, schedule, {psi_00, psi_10}, integrator, {},
      observables, cudaq::IntermediateResultSave::ExpectationValue);

  // Retrieve the evolution result corresponding to each initial state.
  auto &evolution_result_00 = evolution_results[0];
  auto &evolution_result_10 = evolution_results[1];

  // Lambda to extract expectation values for a given observable index
  auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
    std::vector<double> expectations;

    auto all_exps = result.expectation_values.value();
    for (auto exp_vals : all_exps) {
      expectations.push_back((double)exp_vals[idx]);
    }
    return expectations;
  };

  // For the two `evolutions`, extract the six observable trajectories.
  // For the |00> initial state, we extract the expectation trajectories for
  // each observable.
  auto result_00_0 = get_expectation(0, evolution_result_00);
  auto result_00_1 = get_expectation(1, evolution_result_00);
  auto result_00_2 = get_expectation(2, evolution_result_00);
  auto result_00_3 = get_expectation(3, evolution_result_00);
  auto result_00_4 = get_expectation(4, evolution_result_00);
  auto result_00_5 = get_expectation(5, evolution_result_00);

  // Similarly, for the |10> initial state:
  auto result_10_0 = get_expectation(0, evolution_result_10);
  auto result_10_1 = get_expectation(1, evolution_result_10);
  auto result_10_2 = get_expectation(2, evolution_result_10);
  auto result_10_3 = get_expectation(3, evolution_result_10);
  auto result_10_4 = get_expectation(4, evolution_result_10);
  auto result_10_5 = get_expectation(5, evolution_result_10);

  // Export the results to a `CSV` file
  // Export the Z-component of qubit 1's expectation values for both initial
  // states. The `CSV` file "cross_resonance_z.`csv`" will have time versus (Z1)
  // data for both |00> and |10> initial conditions.
  export_csv("cross_resonance_z.csv", "time", steps, "<Z1>_00", result_00_5,
             "<Z1>_10", result_10_5);
  // Export the Y-component of qubit 1's expectation values for both initial
  // states. The `CSV` file "cross_resonance_y.`csv`" will have time versus (Y1)
  // data.
  export_csv("cross_resonance_y.csv", "time", steps, "<Y1>_00", result_00_4,
             "<Y1>_10", result_10_4);

  std::cout
      << "Simulation complete. The results are saved in cross_resonance_z.csv "
         "and cross_resonance_y.csv files."
      << std::endl;
  return 0;
}
