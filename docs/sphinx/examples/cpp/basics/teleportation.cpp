# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. #
# All rights reserved.                                                        #
#                                                                             #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.    #
# ============================================================================ #

#include <cudaq.h>
#include <iostream>

/*
 * Quantum Teleportation allows for the transfer of a quantum state from one 
 * qubit to another, using a shared entangled pair and classical communication.
 * 
 * This example demonstrates:
 * 1. Mid-circuit measurements (`mz`).
 * 2. Conditional operations based on classical measurement results.
 * 3. Use of `cudaq::run` to execute kernels and process results.
 */

struct teleportation {
  auto operator()() __qpu__ {
    // Allocate 3 qubits:
    // q[0]: The qubit to be teleported (Alice's data)
    // q[1]: Alice's half of the Bell pair
    // q[2]: Bob's half of the Bell pair
    cudaq::qarray<3> q;

    // 1. Prepare the state to be teleported on q[0].
    // For this example, let's prepare the |1> state by applying X.
    x(q[0]);

    // 2. Create a Bell pair between q[1] and q[2].
    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);

    // 3. Alice performs a Bell measurement on q[0] and q[1].
    x<cudaq::ctrl>(q[0], q[1]);
    h(q[0]);

    // Mid-circuit measurement
    auto b0 = mz(q[0]);
    auto b1 = mz(q[1]);

    // 4. Bob applies conditional gates based on Alice's measurements.
    if (b1)
      x(q[2]);
    if (b0)
      z(q[2]);

    // 5. Measure Bob's qubit to verify the state was teleported.
    // It should be in the |1> state.
    return mz(q[2]);
  }
};

int main() {
  std::cout << "Executing Quantum Teleportation...\n";

  // Since teleportation is a probabilistic process that requires 
  // classical feedback, we use `cudaq::run` instead of `cudaq::sample`.
  // `cudaq::sample` does not support kernels with conditional feedback.
  int n_shots = 100;
  auto results = cudaq::run(n_shots, teleportation{});

  // Extract counts for the last qubit (Bob's qubit)
  std::size_t n_ones = 0;
  for (auto r : results) {
    if (r) {
        n_ones++;
    }
  }

  std::cout << "Measured '1' on target qubit " << n_ones << " times out of " << n_shots << " shots.\n";

  // Validation
  if (n_ones == n_shots) {
      std::cout << "Success! The |1> state was teleported perfectly.\n";
  } else {
      std::cout << "Failure.\n";
  }

  return 0;
}
