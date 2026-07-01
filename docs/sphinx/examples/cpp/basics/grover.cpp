/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cmath>
#include <cstdio>
#include <cudaq.h>
#include <iostream>

/*
 * Grover's algorithm is a quantum algorithm that finds with high probability
 * the unique input to a black box function that produces a particular output
 * value, using just O(sqrt(N)) evaluations of the function, where N is the
 * size of the function's domain.
 *
 * This example demonstrates:
 * 1. Multi-controlled Z gates.
 * 2. The `cudaq::compute_action` primitive for automatic un-computation.
 * 3. Kernel composition and template kernels.
 */

// Reflection about the uniform superposition state.
__qpu__ void reflect_about_uniform(cudaq::qvector<> &qs) {
  auto ctrlQubits = qs.front(qs.size() - 1);
  auto &lastQubit = qs.back();

  // Compute (U) Action (V) produces: U V U::Adjoint
  cudaq::compute_action(
      [&]() {
        h(qs);
        x(qs);
      },
      [&]() { z<cudaq::ctrl>(ctrlQubits, lastQubit); });
}

// The oracle marks the target state by flipping its phase.
struct oracle {
  void operator()(const long target_state, cudaq::qvector<> &qs) __qpu__ {
    cudaq::compute_action(
        [&]() {
          for (int i = 0; i < qs.size(); ++i) {
            // If the bit in target_state is 0, apply X to that qubit
            // so that the multi-controlled Z applies to the |0> state.
            bool target_bit_set = (target_state >> (qs.size() - i - 1)) & 1;
            if (!target_bit_set)
              x(qs[i]);
          }
        },
        [&]() {
          auto ctrlQubits = qs.front(qs.size() - 1);
          z<cudaq::ctrl>(ctrlQubits, qs.back());
        });
  }
};

// The main Grover algorithm kernel.
struct run_grover {
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, const long target_state,
                          CallableKernel &&oracle_kernel) {
    // Calculate the optimal number of iterations: floor(pi/4 * sqrt(N))
    int n_iterations = std::round(0.25 * M_PI * std::sqrt(1 << n_qubits));

    cudaq::qvector qs(n_qubits);

    // Start in uniform superposition.
    h(qs);

    // Iteratively apply oracle and diffusion operator.
    for (int i = 0; i < n_iterations; i++) {
      oracle_kernel(target_state, qs);
      reflect_about_uniform(qs);
    }

    // Measure to find the target state.
    mz(qs);
  }
};

int main() {
  const int n_qubits = 4;
  const long target_state = 0b1011; // We are searching for 11

  std::cout << "Searching for target state: 1011\n";

  // Sample the kernel.
  auto counts = cudaq::sample(run_grover{}, n_qubits, target_state, oracle{});

  // Output results.
  std::cout << "Found string: " << counts.most_probable() << "\n";

  // Validation.
  if (counts.most_probable() == "1011") {
    std::cout << "Success!\n";
  } else {
    std::cout << "Failure.\n";
  }

  return 0;
}
