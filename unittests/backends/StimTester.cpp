/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "StimCircuitSimulator.cpp"

#include <gtest/gtest.h>

#include "CUDAQTestUtils.h"

/// Wrapper to expose protected methods for testing.
class StimCircuitSimulatorTester : public nvqir::StimCircuitSimulator {
public:
  using GateApplicationTask =
      nvqir::CircuitSimulatorBase<double>::GateApplicationTask;

  void applyNamedGate(const std::string &name,
                      const std::vector<std::size_t> &controls,
                      const std::vector<std::size_t> &targets) {
    flushGateQueue();
    GateApplicationTask task(name, {}, controls, targets, {});
    applyGate(task);
  }

  void resetToZero() { setToZeroState(); }
};

CUDAQ_TEST(StimTester, TwoQubitPauliProducts) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();
  auto q1 = sim.allocateQubit();

  sim.applyNamedGate("IX", {}, {q0, q1});
  EXPECT_EQ(false, sim.mz(q0));
  EXPECT_EQ(true, sim.mz(q1));
}

CUDAQ_TEST(StimTester, IDGateMapping) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();

  sim.applyNamedGate("ID", {}, {q0});
  EXPECT_EQ(false, sim.mz(q0));
}

CUDAQ_TEST(StimTester, SetToZeroStatePreservesSimulators) {
  StimCircuitSimulatorTester sim;
  sim.setRandomSeed(42);
  auto q0 = sim.allocateQubit();

  for (int i = 0; i < 3; i++) {
    sim.x(q0);
    EXPECT_EQ(true, sim.mz(q0));
    sim.resetToZero();
  }
  EXPECT_EQ(false, sim.mz(q0));
}
