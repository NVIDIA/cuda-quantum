/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "common/ExecutionContext.h"
#include "nvqir/MockSimulator.h"

namespace {} // anonymous namespace

/// Allocating qubits with a pre-constructed SimulationState, in a pre-allocated
/// state.
TEST(CircuitSimulatorTester, BatchModeAllocateWithStateBug) {
  MockCircuitSimulator simulator;

  cudaq::ExecutionContext ctx("sample", 1);
  ctx.totalIterations = 2; // batch mode triggers memory re-use
  ctx.batchIteration = 0;

  {
    // First iteration: allocate 2 qubits normally
    simulator.setExecutionContext(&ctx);

    auto qubits = simulator.allocateQubits(2);
    ASSERT_EQ(qubits.size(), 2u);
    EXPECT_EQ(simulator.getNumQubitsAllocated(), 2u);
    EXPECT_EQ(simulator.getMockStateNumQubits(), 2u);

    // Computation done, deallocate the qubits (will be deferred in batch mode)
    simulator.deallocateQubits(qubits);
    simulator.resetExecutionContext();
  }

  // Verify state was NOT deallocated (batch mode behavior)
  // The nQubitsAllocated should still be 2
  EXPECT_EQ(simulator.getNumQubitsAllocated(), 2u);

  // Start second iteration
  ctx.batchIteration = 1;
  // Create a mock simulation state with 2 qubits
  MockSimulationState mockState(2);

  {
    // Second iteration: allocate 2 qubits again, should re-use the memory
    // allocated earlier
    simulator.setExecutionContext(&ctx);
    auto qubits2 = simulator.allocateQubits(2, &mockState);
    ASSERT_EQ(qubits2.size(), 2u);
    ASSERT_EQ(simulator.getMockStateNumQubits(), 2u);

    // Computation done, deallocate the qubits (will be deferred in batch mode)
    simulator.deallocateQubits(qubits2);
    simulator.resetExecutionContext();
  }
}
