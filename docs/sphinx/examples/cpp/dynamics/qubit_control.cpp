/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ --target dynamics qubit_control.cpp -o a.out && ./a.out
// ```

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include "export_csv_helper.h"
#include <cudaq.h>

int main() {
  // Qubit resonant frequency (energy splitting along Z).
  double omega_z = 10.0 * 2 * M_PI;
  // Transverse driving term (amplitude of the drive along the X-axis).
  double omega_x = 2 * M_PI;
  // Driving frequency, chosen to be slightly off-resonance (0.99 of omega_z).
  double omega_drive = 0.99 * omega_z;

  // The lambda function acts as a callback that returns a modulation factor for
  // the drive. It extracts the time `t` from the provided parameters and
  // computes cos(omega_drive * t).
  auto mod_func =
      [omega_drive](
          const cudaq::parameter_map &params) -> std::complex<double> {
    auto it = params.find("t");
    if (it != params.end()) {
      double t = it->second.real();
      const auto result = std::cos(omega_drive * t);
      return result;
    }
    throw std::runtime_error("Cannot find the time parameter.");
  };

  // The Hamiltonian consists of two terms:
  // 1. A static term: 0.5 * omega_z * `Sz`_0, representing the `qubit's`
  // intrinsic energy splitting.
  // 2. A time-dependent driving term: omega_x * cos(omega_drive * t) * `Sx`_0,
  // which induces rotations about the X-axis. The scalar_operator(mod_`func`)
  // allows the drive term to vary in time according to mod_`func`.
  auto hamiltonian = 0.5 * omega_z * cudaq::spin_op::z(0) +
                     mod_func * cudaq::spin_op::x(0) * omega_x;

  // A single qubit with dimension 2.
  cudaq::dimension_map dimensions = {{0, 2}};

  // The qubit starts in the |0> state, represented by the vector [1, 0].
  std::vector<std::complex<double>> initial_state_vec = {1.0, 0.0};
  auto psi0 = cudaq::state::from_data(initial_state_vec);

  // Set the final simulation time such that t_final = pi / omega_x, which
  // relates to a specific qubit rotation.
  double t_final = M_PI / omega_x;
  // Define the integration time step `dt` as a small fraction of the drive
  // period.
  double dt = 2.0 * M_PI / omega_drive / 100;
  // Compute the number of steps required for the simulation
  int num_steps = static_cast<int>(std::ceil(t_final / dt)) + 1;
  // Create a schedule with time steps from 0 to t_final.
  std::vector<double> steps = cudaq::linspace(0.0, t_final, num_steps);
  // The schedule carries the time parameter `labelled` `t`, which is used by
  // mod_`func`.
  cudaq::schedule schedule(steps, {"t"});

  // A default Runge-`Kutta` integrator (4`th` order) with time step `dt`
  // depending on the schedule.
  cudaq::integrators::runge_kutta integrator;

  // Measure the expectation values of the `qubit's` spin components along the
  // X, Y, and Z directions.
  auto observables = {cudaq::spin_op::x(0), cudaq::spin_op::y(0),
                      cudaq::spin_op::z(0)};

  // Simulation without decoherence
  // Evolve the system under the Hamiltonian, using the specified schedule and
  // integrator. No collapse operators are included (closed system evolution).
  auto evolve_result = cudaq::evolve(
      hamiltonian, dimensions, schedule, psi0, integrator, {}, observables,
      cudaq::IntermediateResultSave::ExpectationValue);

  // Simulation with decoherence
  // Introduce `dephasing` (decoherence) through a collapse operator.
  // Here, gamma_`sz` = 1.0 is the `dephasing` rate, and the collapse operator
  // is `sqrt`(gamma_`sz`) * `Sz`_0 which simulates decoherence in the energy
  // basis (Z-basis `dephasing`).
  double gamma_sz = 1.0;
  auto evolve_result_decay =
      cudaq::evolve(hamiltonian, dimensions, schedule, psi0, integrator,
                    {std::sqrt(gamma_sz) * cudaq::spin_op::z(0)}, observables,
                    cudaq::IntermediateResultSave::ExpectationValue);

  // Lambda to extract expectation values for a given observable index
  auto get_expectation = [](int idx, auto &result) -> std::vector<double> {
    std::vector<double> expectations;

    auto all_exps = result.expectation_values.value();
    for (auto exp_vals : all_exps) {
      expectations.push_back(exp_vals[idx].expectation());
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

  // Export the results to a `CSV` file
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
