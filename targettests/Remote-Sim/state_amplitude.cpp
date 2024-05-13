/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-backend tensornet %s -o %t && %t
// clang-format on

#include <cudaq.h>

struct cat_state {
  void operator()(int N) __qpu__ {
    cudaq::qvector qubits(N);
    h(qubits[0]);
    for (int i = 0; i < N - 1; ++i)
      cx(qubits[i], qubits[i + 1]);
  }
};

int main() {
  constexpr int numQubits = 100;
  const std::vector<int> basisStateAll0(numQubits, 0);
  const std::vector<int> basisStateAll1(numQubits, 1);
  auto state = cudaq::get_state(cat_state{}, numQubits);
  assert(std::abs(M_SQRT1_2 - state.amplitude(basisStateAll0)) < 1e-3);
  assert(std::abs(M_SQRT1_2 - state.amplitude(basisStateAll1)) < 1e-3);

  return 0;
}
