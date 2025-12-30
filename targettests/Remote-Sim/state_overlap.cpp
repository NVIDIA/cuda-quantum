/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>

struct bellCircuit {
  void operator()(int N) __qpu__ {
    cudaq::qvector qubits(N);
    h(qubits[0]);
    for (int i = 0; i < N - 1; ++i)
      cx(qubits[i], qubits[i + 1]);
  }
};

struct noOpCircuit {
  void operator()(int N) __qpu__ {
    cudaq::qvector qubits(N);
    h(qubits);
    h(qubits);
  }
};

int main() {
  constexpr int N = 2;
  auto state1 = cudaq::get_state(bellCircuit{}, 2);
  auto state2 = cudaq::get_state(noOpCircuit{}, 2);
  const auto overlap = state1.overlap(state2);
  REMOTE_TEST_ASSERT(std::abs(M_SQRT1_2 - overlap) < 1e-3);
  return 0;
}
