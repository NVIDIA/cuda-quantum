/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

#ifndef CUDAQ_BACKEND_STIM
namespace ccnot_tester {
// Local lambdas passed to cudaq::control / cudaq::adjoint are lifted to
// separate functions; apply-op-specialization then fails on those calls
// because inlining runs later in the nvq++ pipeline. Use named noinline
// __qpu__ helpers instead (same pattern as adjoint_tester / grover_test).
// See https://github.com/NVIDIA/cuda-quantum/issues/3762.
__attribute__((noinline)) __qpu__ void apply_x(cudaq::qubit &q) { x(q); }

__attribute__((noinline)) __qpu__ void test_inner_adjoint(cudaq::qubit &q) {
  cudaq::adjoint(apply_x, q);
}

__attribute__((noinline)) __qpu__ void ctrl_x(cudaq::qubit &ctrl,
                                              cudaq::qubit &target) {
  cudaq::control(apply_x, ctrl, target);
}

// Demonstrate we can perform multi-controlled operations
struct ccnot_test {
  void operator()() __qpu__ {
    cudaq::qvector q(3);

    x(q);
    x(q[1]);

    auto controls = q.front(2);
    cudaq::control(test_inner_adjoint, controls, q[2]);

    mz(q);
  }
};

struct nested_ctrl {
  void operator()() __qpu__ {
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
    cudaq::control(ctrl_x, q[0], q[1], q[2]);

    mz(q);
  }
};
} // namespace ccnot_tester

using namespace ccnot_tester;

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

#endif
