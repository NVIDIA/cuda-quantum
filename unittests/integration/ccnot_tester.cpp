/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

// Demonstrate we can perform multi-controlled operations
struct ccnot_test {
  void operator()() __qpu__ {
    cudaq::qvector q(3);

    x(q);
    x(q[1]);

    auto apply_x = [](cudaq::qubit &q) { x(q); };
    auto test_inner_adjoint = [&](cudaq::qubit &q) {
      cudaq::adjoint(apply_x, q);
    };

    auto controls = q.front(2);
    cudaq::control(test_inner_adjoint, controls, q[2]);

    mz(q);
  }
};

struct nested_ctrl {
  void operator()() __qpu__ {
    auto apply_x = [](cudaq::qubit &r) { x(r); };

    cudaq::qvector q(3);
    // Create 101
    x(q);
    x(q[1]);

    // Fancy nested CCX
    // Walking inner nest to outer
    // 1. Queue X(q[2])
    // 2. Queue Ctrl (q[1]) X (q[2])
    // 3. Queue Ctrl (q[0], q[1]) X(q[2]);
    // 4. Apply
    cudaq::control(
        [&](cudaq::qubit &r) {
          cudaq::control([&](cudaq::qubit &r) { apply_x(r); }, q[1], r);
        },
        q[0], q[2]);

    mz(q);
  }
};

CUDAQ_TEST(CCNOTTester, checkSimple) {
  auto ccnot = []() {
    cudaq::qvector q(3);

    // Apply X to the following qubits
    x(q[0], q[2]);

    // Apply control X with q0 q1 as controls
    x<cudaq::ctrl>(q[0], q[1], q[2]);

    mz(q);
  };

  auto counts = cudaq::sample(ccnot);
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "101");

  auto counts2 = cudaq::sample(ccnot_test{});
  EXPECT_EQ(1, counts2.size());
  EXPECT_TRUE(counts2.begin()->first == "101");

  auto counts3 = cudaq::sample(nested_ctrl{});
  EXPECT_EQ(1, counts3.size());
  EXPECT_TRUE(counts3.begin()->first == "101");
}

CUDAQ_TEST(FredkinTester, checkTruth) {

  auto test = []() __qpu__ {
    cudaq::qubit q, r, s;
    x(q, s);
    swap<cudaq::ctrl>(q, r, s);
    mz(q, r, s);
  };

  auto counts = cudaq::sample(test);
  counts.dump();
  EXPECT_EQ(counts.size(), 1);
  EXPECT_EQ(counts.begin()->first, "110");
}