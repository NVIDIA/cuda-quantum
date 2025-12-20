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

struct hGateTest {
  auto operator()(const std::vector<double> &angles) __qpu__ {
    cudaq::qubit q;
    ry(angles[0], q);
    rx(angles[1], q);
    h(q);
  }
};

int main() {
  // H == Rx(pi)Ry(pi/2) (up to a global phase)
  auto counts = cudaq::sample(hGateTest{}, std::vector<double>{M_PI_2, M_PI});
  counts.dump();
  REMOTE_TEST_ASSERT(counts.size() == 1);
  REMOTE_TEST_ASSERT(counts.begin()->first == "0");
  return 0;
}
