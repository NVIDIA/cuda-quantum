/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>

struct bellCircuit {
  void operator()() __qpu__ {
    cudaq::qvector qubits(2);
    h(qubits[0]);
    cx(qubits[0], qubits[1]);
  }
};

struct noOpCircuit {
  void operator()() __qpu__ {
    cudaq::qvector qubits(2);
    h(qubits[0]);
    cx(qubits[0], qubits[1]);
    cx(qubits[0], qubits[1]);
    h(qubits[0]);
  }
};

int main() {
  auto state1 = cudaq::get_state(bellCircuit{});
  auto state2 = cudaq::get_state(noOpCircuit{});
  const auto overlap = state1.overlap(state2);
  REMOTE_TEST_ASSERT(std::abs(M_SQRT1_2 - overlap) < 1e-3);
  return 0;
}
