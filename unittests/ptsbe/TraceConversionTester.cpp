/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBEInterface.h"
#include "nvqir/Gates.h"
#include <cmath>

using namespace cudaq;
using namespace cudaq::ptsbe;

/// Verify basic conversion: gate name, matrix populated, qubit IDs extracted
CUDAQ_TEST(TraceConversionTest, BasicConversion) {
  Trace::Instruction inst("h", {}, {}, {QuditInfo(2, 5)});
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
  Trace::Instruction inst("rx", {angle}, {}, {QuditInfo(2, 0)});
  auto task = convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.operationName, "rx");
  EXPECT_EQ(task.parameters.size(), 1u);
  EXPECT_NEAR(task.parameters[0], angle, 1e-12);
}

/// Verify controlled gate: controls and targets extracted correctly
CUDAQ_TEST(TraceConversionTest, ControlledGate) {
  Trace::Instruction inst("x", {}, {QuditInfo(2, 0), QuditInfo(2, 1)},
                          {QuditInfo(2, 2)});
  auto task = convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.controls.size(), 2u);
  EXPECT_EQ(task.controls[0], 0u);
  EXPECT_EQ(task.controls[1], 1u);
  EXPECT_EQ(task.targets.size(), 1u);
  EXPECT_EQ(task.targets[0], 2u);
}

/// Verify unknown gate throws with descriptive error
CUDAQ_TEST(TraceConversionTest, UnknownGateThrows) {
  Trace::Instruction inst("invalid_gate_xyz", {}, {}, {QuditInfo(2, 0)});
  EXPECT_THROW(convertToSimulatorTask<double>(inst), std::runtime_error);
}

/// Verify float precision: parameters cast to float
CUDAQ_TEST(TraceConversionTest, FloatPrecision) {
  Trace::Instruction inst("rx", {M_PI / 4}, {}, {QuditInfo(2, 0)});
  auto task = convertToSimulatorTask<float>(inst);

  EXPECT_EQ(task.parameters.size(), 1u);
  EXPECT_NEAR(task.parameters[0], static_cast<float>(M_PI / 4), 1e-6f);
}

/// Verify multi-target gate (swap)
CUDAQ_TEST(TraceConversionTest, MultiTargetGate) {
  Trace::Instruction inst("swap", {}, {}, {QuditInfo(2, 3), QuditInfo(2, 7)});
  auto task = convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.targets.size(), 2u);
  EXPECT_EQ(task.targets[0], 3u);
  EXPECT_EQ(task.targets[1], 7u);
  EXPECT_EQ(task.matrix.size(), 16u);
}
