/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ --target dynamics dynamics_integrators.cpp -o a.out && ./a.out
// ```

#include "cudaq/algorithms/evolve.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/operators.h"
#include <cudaq.h>

int main() {
  // Common setup: single qubit driven by a transverse field.
  auto hamiltonian = 2.0 * M_PI * 0.1 * cudaq::spin_op::x(0);
  const cudaq::dimension_map dimensions = {{0, 2}};
  std::vector<std::complex<double>> state_vec = {1.0, 0.0};
  auto psi0 = cudaq::state::from_data(state_vec);
  std::vector<double> steps = cudaq::linspace(0.0, 1.0, 101);
  cudaq::schedule schedule(steps);
  auto observables = {cudaq::spin_op::z(0)};

  // [Begin RungeKutta]
  // Explicit 4th-order Runge-Kutta method (the default integrator).
  // Arguments: order (1, 2, or 4) and optional max sub-step size.
  cudaq::integrators::runge_kutta rk_integrator(/*order=*/4,
                                                /*max_step_size=*/0.01);
  auto rk_result = cudaq::evolve(
      hamiltonian, dimensions, schedule, psi0, rk_integrator, {}, observables,
      cudaq::IntermediateResultSave::ExpectationValue);
  // [End RungeKutta]

  // [Begin CrankNicolson]
  // Implicit Crank-Nicolson predictor-corrector method.
  // Well-suited for stiff systems or when energy conservation is important.
  // Arguments: number of corrector iterations (default: 2) and optional max
  // sub-step size.
  cudaq::integrators::crank_nicolson cn_integrator(/*num_corrector_steps=*/2,
                                                   /*max_step_size=*/0.01);
  auto cn_result = cudaq::evolve(
      hamiltonian, dimensions, schedule, psi0, cn_integrator, {}, observables,
      cudaq::IntermediateResultSave::ExpectationValue);
  // [End CrankNicolson]

  // [Begin MagnusExpansion]
  // Magnus expansion integrator.
  // Uses a finite Taylor series truncation to approximate the matrix
  // exponential, approximating unitary evolution. Suitable for smooth,
  // oscillatory
  // Hamiltonians. Arguments: maximum number of Taylor terms (default: 10) and
  // optional max sub-step size.
  cudaq::integrators::magnus_expansion magnus_integrator(
      /*num_taylor_terms=*/10, /*max_step_size=*/0.01);
  auto magnus_result = cudaq::evolve(
      hamiltonian, dimensions, schedule, psi0, magnus_integrator, {},
      observables, cudaq::IntermediateResultSave::ExpectationValue);
  // [End MagnusExpansion]

  return 0;
}
