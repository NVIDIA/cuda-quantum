/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <iostream>

// [Begin Documentation]
#include <cudaq.h>

struct xOp {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
  }
};

int main() {
  // Add a simple bit-flip noise channel to X gate
  const double error_probability = 0.1;

  cudaq::bit_flip_channel bit_flip(error_probability);
  // Add noise channels to our noise model.
  cudaq::noise_model noise_model;
  // Apply the bitflip channel to any X-gate on any qubits
  noise_model.add_all_qubit_channel<cudaq::types::x>(bit_flip);

  double noisy_exp_val =
      cudaq::observe({.noise = noise_model, .num_trajectories = 1024}, xOp{},
                     cudaq::spin::z(0));

  // True expectation: 0.1 - 0.9 = -0.8 (|1> has <Z> of -1 and |1> has <Z> of
  // +1)
  std::cout << "Noisy <Z> with 1024 trajectories = " << noisy_exp_val << "\n";

  // Rerun with a higher number of trajectories
  noisy_exp_val =
      cudaq::observe({.noise = noise_model, .num_trajectories = 8192}, xOp{},
                     cudaq::spin::z(0));
  std::cout << "Noisy <Z> with 8192 trajectories = " << noisy_exp_val << "\n";
  return 0;
}
