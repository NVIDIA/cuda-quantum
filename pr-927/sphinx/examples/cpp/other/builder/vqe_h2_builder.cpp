/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ vqe_h2_builder.cpp -o builder.x && ./builder.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>

// This example demonstrates the utility of the builder pattern
// for a common chemistry example. Here we build up a CUDA Quantum kernel
// with N layers and each layer containing an arrangement of
// random SO(4) rotations. The algorithm leverages the CUDA Quantum
// VQE support to compute the ground state of the Hydrogen atom.

namespace cudaq {

// Define a function that applies a general SO(4) rotation to
// the builder on the provided qubits with the provided parameters.
// Note we keep this qubit and parameter arguments as auto as these
// will default to taking the qubits and variational parameters (`QuakeValue`s)
void so4(cudaq::kernel_builder<std::vector<double>> &builder, QuakeValue &&q,
         QuakeValue &&r, QuakeValue &parameters) {
  builder.ry(parameters[0], q);
  builder.ry(parameters[1], r);

  builder.h(r);
  builder.x<cudaq::ctrl>(q, r);
  builder.h(r);

  builder.ry(parameters[2], q);
  builder.ry(parameters[3], r);

  builder.h(r);
  builder.x<cudaq::ctrl>(q, r);
  builder.h(r);

  builder.ry(parameters[4], q);
  builder.ry(parameters[5], r);

  builder.h(r);
  builder.x<cudaq::ctrl>(q, r);
  builder.h(r);
}

} // namespace cudaq

int main() {

  // Read in the spin op from file
  std::vector<double> h2_data{0, 0, 0, 0, -0.10647701149499994, 0.0,
                              1, 1, 1, 1, 0.0454063328691,      0.0,
                              1, 1, 3, 3, 0.0454063328691,      0.0,
                              3, 3, 1, 1, 0.0454063328691,      0.0,
                              3, 3, 3, 3, 0.0454063328691,      0.0,
                              2, 0, 0, 0, 0.170280101353,       0.0,
                              2, 2, 0, 0, 0.120200490713,       0.0,
                              2, 0, 2, 0, 0.168335986252,       0.0,
                              2, 0, 0, 2, 0.165606823582,       0.0,
                              0, 2, 0, 0, -0.22004130022499996, 0.0,
                              0, 2, 2, 0, 0.165606823582,       0.0,
                              0, 2, 0, 2, 0.174072892497,       0.0,
                              0, 0, 2, 0, 0.17028010135300004,  0.0,
                              0, 0, 2, 2, 0.120200490713,       0.0,
                              0, 0, 0, 2, -0.22004130022499999, 0.0,
                              15};
  cudaq::spin_op H(h2_data, /*nQubits*/ 4);

  int layers = 2, n_qubits = H.num_qubits(), block_size = 2, p_counter = 0;
  int n_blocks_per_layer = 2 * (n_qubits / block_size) - 1;
  int n_params = layers * 6 * n_blocks_per_layer;
  printf("%d qubit hamiltonian -> %d parameters\n", n_qubits, n_params);

  // Create the builder with signature void(std::vector<double>)
  auto [kernel, params] = cudaq::make_kernel<std::vector<double>>();

  // Allocate the qubits, initialize the HF product state
  auto q = kernel.qalloc(n_qubits);
  kernel.x(q[0]);
  kernel.x(q[2]);

  // Loop over adding layers of SO4 entanglers
  int counter = 0;
  for (int i = 0; i < layers; i++) {
    // first layer of so4 blocks (even)
    for (int k = 0; k < n_qubits; k += block_size) {
      auto subq = q.slice(k, block_size);
      auto sub_p = params.slice(p_counter, 6);
      cudaq::so4(kernel, subq[0], subq[1], sub_p);
      p_counter += 6;
    }

    // second layer of so4 blocks (odd)
    for (int k = 1; k + block_size < n_qubits; k += block_size) {
      auto subq = q.slice(k, block_size);
      auto sub_p = params.slice(p_counter, 6);
      cudaq::so4(kernel, subq[0], subq[1], sub_p);
      p_counter += 6;
    }
  }

  // Run the VQE algorithm from specific initial parameters.
  auto init_params =
      cudaq::random_vector(-1, 1, n_params, std::mt19937::default_seed);

  // Run VQE
  cudaq::optimizers::lbfgs optimizer;
  cudaq::gradients::central_difference gradient(kernel);
  optimizer.initial_parameters = init_params;
  optimizer.max_eval = 20;
  auto [opt_val, opt_params] =
      cudaq::vqe(kernel, gradient, H, optimizer, n_params);
  printf("Optimal value = %.16lf\n", opt_val);
}
