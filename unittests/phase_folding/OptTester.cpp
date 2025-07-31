/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <gtest/gtest.h>
#include <random>

TEST(OptTester, checkSimple) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q, p, r;
    h(q);
    h(p);
    h(r);
    rz(1.0, p);
    rz(2.0, r);
    x<cudaq::ctrl>(p, q);
    rz(3.0, q);
    x<cudaq::ctrl>(p, r);
    x<cudaq::ctrl>(q, p);
    h(r);
    x<cudaq::ctrl>(p, r);
    x<cudaq::ctrl>(q, p);
    rz(4.0, p);
    h(q);
    h(p);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  setenv(PHASE_SWITCH, "0", true);
  auto state1 = cudaq::get_state(kernel);
  setenv(PHASE_SWITCH, "1", true);
  auto state2 = cudaq::get_state(kernel);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  EXPECT_NEAR(result.real(), 1, 0.0000001);
  EXPECT_NEAR(result.imag(), 0, 0.0000001);
}
