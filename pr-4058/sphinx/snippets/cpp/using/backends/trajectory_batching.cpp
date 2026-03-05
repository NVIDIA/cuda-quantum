/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Documentation]
#include <chrono>
#include <cudaq.h>
#include <iostream>

struct xOp {
  void operator()(int qubit_count) __qpu__ {
    cudaq::qvector q(qubit_count);
    x(q);
    mz(q);
  }
};

int main() {
  // Add a simple bit-flip noise channel to X gate
  const double error_probability = 0.01;

  cudaq::bit_flip_channel bit_flip(error_probability);
  // Add noise channels to our noise model.
  cudaq::noise_model noise_model;
  // Apply the bitflip channel to any X-gate on any qubits
  noise_model.add_all_qubit_channel<cudaq::types::x>(bit_flip);

  const int qubit_count = 10;
  const auto start_time = std::chrono::high_resolution_clock::now();
  // Due to the impact of noise, our measurements will no longer be uniformly in
  // the |1...1> state.
  auto counts =
      cudaq::sample({.shots = 1000, .noise = noise_model}, xOp{}, qubit_count);
  const auto end_time = std::chrono::high_resolution_clock::now();
  counts.dump();
  const std::chrono::duration<double, std::milli> elapsed_time =
      end_time - start_time;
  std::cout << "Simulation elapsed time: " << elapsed_time.count() << "ms\n";
  return 0;
}
