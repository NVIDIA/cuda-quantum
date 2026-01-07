/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include <cudaq/algorithm.h>
#include <iostream>

struct kernelSingle {
  auto operator()(cudaq::pauli_word pauli, double theta) __qpu__ {
    cudaq::qvector q(4);
    x(q[0]);
    x(q[1]);
    exp_pauli(theta, q, pauli);
  }
};

struct kernelVector {
  void operator()(std::vector<cudaq::pauli_word> &words, double theta) __qpu__ {
    cudaq::qvector q(4);
    x(q[0]);
    x(q[1]);
    exp_pauli(theta / 3.0, q, words[0]);
    exp_pauli(theta / 3.0, q, words[1]);
    exp_pauli(theta / 3.0, q, words[2]);
  }
};

int main() {
  const std::vector<double> h2_data{
      3, 1, 1, 3, 0.0454063,  0,  2, 0, 0, 0, 0.17028,    0,
      0, 0, 2, 0, -0.220041,  -0, 1, 3, 3, 1, 0.0454063,  0,
      0, 0, 0, 0, -0.106477,  0,  0, 2, 0, 0, 0.17028,    0,
      0, 0, 0, 2, -0.220041,  -0, 3, 3, 1, 1, -0.0454063, -0,
      2, 2, 0, 0, 0.168336,   0,  2, 0, 2, 0, 0.1202,     0,
      0, 2, 0, 2, 0.1202,     0,  2, 0, 0, 2, 0.165607,   0,
      0, 2, 2, 0, 0.165607,   0,  0, 0, 2, 2, 0.174073,   0,
      1, 1, 3, 3, -0.0454063, -0, 15};
  cudaq::spin_op h(h2_data, 4);
  {
    const double e =
        cudaq::observe(kernelSingle{}, h, cudaq::pauli_word{"XXXY"}, 0.11);
    std::cout << "e = " << e << "\n";
    constexpr double expectedVal = -1.13;
    REMOTE_TEST_ASSERT(std::abs(e - expectedVal) < 0.1);
  }
  {
    // Vector of three terms (same values)
    std::vector<cudaq::pauli_word> paulis(3, cudaq::pauli_word{"XXXY"});
    const double e = cudaq::observe(kernelVector{}, h, paulis, 0.11);
    std::cout << "e = " << e << "\n";
    constexpr double expectedVal = -1.13;
    REMOTE_TEST_ASSERT(std::abs(e - expectedVal) < 0.1);
  }
  return 0;
}
