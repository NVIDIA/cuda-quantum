/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ qaoa_maxcut_builder.cpp -o builder.x && ./builder.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>
#include <cudaq/spin_op.h>

// This example demonstrates the same code as in `qaoa_maxcut.cpp`,
// but with the use of dynamic kernels. Here we have QAOA with `p` layers,
// with each layer containing the alternating set of unitaries corresponding
// to the problem and the mixer Hamiltonians. The algorithm leverages the
// CUDA Quantum VQE support to compute the Max-Cut of a rectangular graph
// illustrated below.
//
//        v0  0---------------------0 v1
//            |                     |
//            |                     |
//            |                     |
//            |                     |
//        v3  0---------------------0 v2
// The Max-Cut for this problem is 0101 or 1010.

int main() {

  using namespace cudaq::spin;

  cudaq::set_random_seed(13); // set for repeatability

  // Problem Hamiltonian
  const cudaq::spin_op Hp = 0.5 * z(0) * z(1) + 0.5 * z(1) * z(2) +
                            0.5 * z(0) * z(3) + 0.5 * z(2) * z(3);

  // Specify problem parameters
  const int n_qubits = 4;
  const int n_layers = 2;
  const int n_params = 2 * n_layers;

  auto [kernel, theta] = cudaq::make_kernel<std::vector<double>>();

  auto q = kernel.qalloc(n_qubits);

  // Prepare the initial state by superposition
  kernel.h(q);

  // Loop over all the layers
  for (int i = 0; i < n_layers; ++i) {
    // Problem Hamiltonian
    for (int j = 0; j < n_qubits; ++j) {

      kernel.x<cudaq::ctrl>(q[j], q[(j + 1) % n_qubits]);
      kernel.rz(2.0 * theta[i], q[(j + 1) % n_qubits]);
      kernel.x<cudaq::ctrl>(q[j], q[(j + 1) % n_qubits]);
    }

    for (int j = 0; j < n_qubits; ++j) {
      // Mixer Hamiltonian
      kernel.rx(2.0 * theta[i + n_layers], q[j]);
    }
  }

  // Instantiate the optimizer
  cudaq::optimizers::lbfgs optimizer;                    // gradient-based
  cudaq::gradients::central_difference gradient(kernel); // grad vector

  // Set initial values for the optimization parameters
  optimizer.initial_parameters = cudaq::random_vector(
      -M_PI / 8.0, M_PI / 8.0, n_params, std::mt19937::default_seed);

  // Optimization using gradient-based L-BFGS
  auto [opt_val, opt_params] =
      cudaq::vqe(kernel, gradient, Hp, optimizer, n_params);

  // Print the optimized value and the parameters
  printf("Optimal value = %.16lf\n", opt_val);
  printf("Optimal params = %.16lf %.16lf %.16lf %.16lf\n", opt_params[0],
         opt_params[1], opt_params[2], opt_params[3]);

  // Sample the circuit using optimized parameters
  auto counts = cudaq::sample(kernel, opt_params);

  // Dump the states and the counts
  counts.dump();

  return 0;
}
