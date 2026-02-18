/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include "nvqir/Gates.h"
#include <cmath>

using namespace cudaq;
using namespace cudaq::ptsbe;

/// Verify basic conversion: gate name, matrix populated, qubit IDs extracted
CUDAQ_TEST(TraceConversionTest, BasicConversion) {
  TraceInstruction inst(TraceInstructionType::Gate, "h", {5}, {}, {});
  auto task = convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.operationName, "h");
  EXPECT_EQ(task.matrix.size(), 4u);
  EXPECT_EQ(task.targets.size(), 1u);
  EXPECT_EQ(task.targets[0], 5u);
  EXPECT_TRUE(task.controls.empty());
  EXPECT_TRUE(task.parameters.empty());
}

/// Verify parameterized gate: parameters passed through and cast to ScalarType
CUDAQ_TEST(TraceConversionTest, ParameterizedGate) {
  double angle = M_PI / 3;
  TraceInstruction inst(TraceInstructionType::Gate, "rx", {0}, {}, {angle});
  auto task = convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.operationName, "rx");
  EXPECT_EQ(task.parameters.size(), 1u);
  EXPECT_NEAR(task.parameters[0], angle, 1e-12);
}

/// Verify controlled gate: controls and targets extracted correctly
CUDAQ_TEST(TraceConversionTest, ControlledGate) {
  TraceInstruction inst(TraceInstructionType::Gate, "x", {2}, {0, 1}, {});
  auto task = convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.controls.size(), 2u);
  EXPECT_EQ(task.controls[0], 0u);
  EXPECT_EQ(task.controls[1], 1u);
  EXPECT_EQ(task.targets.size(), 1u);
  EXPECT_EQ(task.targets[0], 2u);
}

/// Verify unknown gate throws with descriptive error
CUDAQ_TEST(TraceConversionTest, UnknownGateThrows) {
  TraceInstruction inst(TraceInstructionType::Gate, "invalid_gate_xyz", {0}, {},
                        {});
  try {
    convertToSimulatorTask<double>(inst);
    FAIL() << "Expected an exception for unknown gate";
  } catch (...) {
  }
}

/// Verify float precision: parameters cast to float
CUDAQ_TEST(TraceConversionTest, FloatPrecision) {
  TraceInstruction inst(TraceInstructionType::Gate, "rx", {0}, {}, {M_PI / 4});
  auto task = convertToSimulatorTask<float>(inst);

  EXPECT_EQ(task.parameters.size(), 1u);
  EXPECT_NEAR(task.parameters[0], static_cast<float>(M_PI / 4), 1e-6f);
}

/// Verify multi-target gate (swap)
CUDAQ_TEST(TraceConversionTest, MultiTargetGate) {
  TraceInstruction inst(TraceInstructionType::Gate, "swap", {3, 7}, {}, {});
  auto task = convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.targets.size(), 2u);
  EXPECT_EQ(task.targets[0], 3u);
  EXPECT_EQ(task.targets[1], 7u);
  EXPECT_EQ(task.matrix.size(), 16u);
}
