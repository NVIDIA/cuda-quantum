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

  // Dimension of our composite quantum system:
  // subsystem 0 (atom) has 2 levels (ground and excited states).
  // subsystem 1 (cavity) has 10 levels (Fock states, representing photon number
  // states).
  cudaq::dimension_map dimensions{{0, 2}, {1, 10}};

  // For the cavity subsystem 1
  // We create the annihilation (a) and creation (a+) operators.
  // These operators lower and raise the photon number, respectively.
  auto a = cudaq::boson_op::annihilate(1);
  auto a_dag = cudaq::boson_op::create(1);

  // For the atom subsystem 0
  // We create the annihilation (`sm`) and creation (`sm_dag`) operators.
  // These operators lower and raise the excitation number, respectively.
  auto sm = cudaq::boson_op::annihilate(0);
  auto sm_dag = cudaq::boson_op::create(0);

  // Number operators
  // These operators count the number of excitations.
  // For the atom (`subsytem` 0) and the cavity (`subsystem` 1) they give the
  // population in each subsystem.
  auto atom_occ_op = cudaq::matrix_op::number(0);
  auto cavity_occ_op = cudaq::matrix_op::number(1);

  // Hamiltonian
  // The `hamiltonian` models the dynamics of the atom-cavity (cavity QED)
  // system It has 3 parts:
  // 1. Cavity energy: 2 * pi * photon_number_operator -> energy proportional to
  // the number of photons.
  // 2. Atomic energy: 2 * pi * atom_number_operator -> energy proportional to
  // the atomic excitation.
  // 3. Atomic-cavity interaction: 2 * pi * 0.25 * (`sm` * a_dag + `sm_dag` * a)
  // -> represents the exchange of energy between the atom and the cavity. It is
  // analogous to the Jaynes-Cummings model in cavity QED.
  auto hamiltonian = (2 * M_PI * cavity_occ_op) + (2 * M_PI * atom_occ_op) +
                     (2 * M_PI * 0.25 * (sm * a_dag + sm_dag * a));

  // Build the initial state
  // Atom (sub-system 0) in ground state.
  // Cavity (sub-system 1) has 5 photons (Fock space).
  // The overall Hilbert space is 2 * 10 = 20.
  const int num_photons = 5;
  std::vector<std::complex<double>> initial_state_vec(20, 0.0);
  // The index is chosen such that the atom is in the ground state while the
  // cavity is in the Fock state with 5 photons.
  initial_state_vec[dimensions[0] * num_photons] = 1;

  // Define a time evolution schedule
  // We define a time grid from 0 to 10 (in arbitrary time units) with 201 time
  // steps. This schedule is used by the integrator to simulate the dynamics.
  const int num_steps = 201;
  std::vector<double> steps = cudaq::linspace(0.0, 10.0, num_steps);
  cudaq::schedule schedule(steps);

  // Create a CUDA quantum state
  // The initial state is converted into a quantum state object for evolution.
  auto rho0 = cudaq::state::from_data(initial_state_vec);

  // Numerical integrator
  // Here we choose a Runge-`Kutta` method for time evolution.
  // `dt` defines the time step for the numerical integration, and order 4
  // indicates a 4`th` order method.
  cudaq::integrators::runge_kutta integrator(4, 0.01);

  // Evolve without collapse operators
  // This evolution is ideal (closed system) dynamics governed solely by the
  // Hamiltonian. The expectation values of the observables (cavity photon
  // number and atom excitation probability) are recorded.
  cudaq::evolve_result evolve_result =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator, {},
                    {cavity_occ_op, atom_occ_op},
                    cudaq::IntermediateResultSave::ExpectationValue);

  // Adding dissipation
  // To simulate a realistic scenario, we introduce decay (dissipation).
  // Here, the collapse operator represents photon loss from the cavity.
  constexpr double decay_rate = 0.1;
  auto collapse_operator = std::sqrt(decay_rate) * a;
  // Evolve with the collapse operator to incorporate the effect of decay.
  cudaq::evolve_result evolve_result_decay =
      cudaq::evolve(hamiltonian, dimensions, schedule, rho0, integrator,
                    {collapse_operator}, {cavity_occ_op, atom_occ_op},
                    cudaq::IntermediateResultSave::ExpectationValue);

  // Lambda to extract expectation values for a given observable index
  // Here, index 0 corresponds to the cavity photon number and index 1
  // corresponds to the atom excitation probability.
  auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
    std::vector<double> expectations;

    auto all_exps = result.expectation_values.value();
    for (auto exp_vals : all_exps) {
      expectations.push_back((double)exp_vals[idx]);
    }
    return expectations;
  };

  // Retrieve expectation values from both the ideal and decaying `evolutions`.
  auto ideal_result0 = get_expectation(0, evolve_result);
  auto ideal_result1 = get_expectation(1, evolve_result);
  auto decay_result0 = get_expectation(0, evolve_result_decay);
  auto decay_result1 = get_expectation(1, evolve_result_decay);

  // Export the results to `CSV` files
  // "cavity_`qed`_ideal_result.`csv`" contains the ideal evolution results.
  // "cavity_`qed`_decay_result.`csv`" contains the evolution results with
  // cavity decay.
  export_csv("cavity_qed_ideal_result.csv", "time", steps,
             "cavity_photon_number", ideal_result0,
             "atom_excitation_probability", ideal_result1);
  export_csv("cavity_qed_decay_result.csv", "time", steps,
             "cavity_photon_number", decay_result0,
             "atom_excitation_probability", decay_result1);

  std::cout << "Simulation complete. The results are saved in "
               "cavity_qed_ideal_result.csv "
               "and cavity_qed_decay_result.csv files."
            << std::endl;
  return 0;
}
