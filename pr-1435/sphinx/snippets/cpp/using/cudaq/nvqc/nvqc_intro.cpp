/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Documentation]
#include <cudaq.h>

// Define a simple quantum kernel to execute on NVQC.
struct ghz {
  // Maximally entangled state between 25 qubits.
  auto operator()() __qpu__ {
    constexpr int NUM_QUBITS = 25;
    cudaq::qvector q(NUM_QUBITS);
    h(q[0]);
    for (int i = 0; i < NUM_QUBITS - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    auto result = mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(ghz{});
  counts.dump();
}
