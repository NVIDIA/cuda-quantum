/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "nvqir/Gates.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cmath>

using namespace cudaq;

/// Verify basic conversion: gate name, matrix populated, qubit IDs extracted
CUDAQ_TEST(TraceConversionTest, BasicConversion) {
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate, "h", {5}, {},
                               {});
  auto task = cudaq::ptsbe::detail::convertToSimulatorTask<double>(inst);

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
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate, "rx", {0}, {},
                               {angle});
  auto task = cudaq::ptsbe::detail::convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.operationName, "rx");
  EXPECT_EQ(task.parameters.size(), 1u);
  EXPECT_NEAR(task.parameters[0], angle, 1e-12);
}

/// Verify controlled gate: controls and targets extracted correctly
CUDAQ_TEST(TraceConversionTest, ControlledGate) {
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate, "x", {2},
                               {0, 1}, {});
  auto task = cudaq::ptsbe::detail::convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.controls.size(), 2u);
  EXPECT_EQ(task.controls[0], 0u);
  EXPECT_EQ(task.controls[1], 1u);
  EXPECT_EQ(task.targets.size(), 1u);
  EXPECT_EQ(task.targets[0], 2u);
}

/// Verify unknown gate throws with descriptive error
CUDAQ_TEST(TraceConversionTest, UnknownGateThrows) {
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate,
                               "invalid_gate_xyz", {0}, {}, {});
  try {
    cudaq::ptsbe::detail::convertToSimulatorTask<double>(inst);
    FAIL() << "Expected an exception for unknown gate";
  } catch (...) {
  }
}

/// Verify float precision: parameters cast to float
CUDAQ_TEST(TraceConversionTest, FloatPrecision) {
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate, "rx", {0}, {},
                               {M_PI / 4});
  auto task = cudaq::ptsbe::detail::convertToSimulatorTask<float>(inst);

  EXPECT_EQ(task.parameters.size(), 1u);
  EXPECT_NEAR(task.parameters[0], static_cast<float>(M_PI / 4), 1e-6f);
}

/// Verify multi-target gate (swap)
CUDAQ_TEST(TraceConversionTest, MultiTargetGate) {
  ptsbe::TraceInstruction inst(ptsbe::TraceInstructionType::Gate, "swap",
                               {3, 7}, {}, {});
  auto task = cudaq::ptsbe::detail::convertToSimulatorTask<double>(inst);

  EXPECT_EQ(task.targets.size(), 2u);
  EXPECT_EQ(task.targets[0], 3u);
  EXPECT_EQ(task.targets[1], 7u);
  EXPECT_EQ(task.matrix.size(), 16u);
}

CUDAQ_TEST(TraceConversionTest, OpenControlsRestoreBeforeNoise) {
  cudaq::Trace trace;
  trace.appendInstruction("ry", {0.37}, {{2, 1}}, {{2, 0}}, {0});
  trace.appendMeasurement("mz", {{2, 0}, {2, 1}});

  cudaq::noise_model noise;
  // Register X noise to detect accidental noise on the synthetic conjugations.
  noise.add_channel("x", {1}, cudaq::bit_flip_channel(1.));
  noise.add_channel("ry", {1, 0}, cudaq::depolarization2(0.1));
  const auto converted = cudaq::ptsbe::detail::buildPTSBETrace(trace, noise);

  // One modeled noise event follows the complete ideal X / controlled-Ry / X.
  ASSERT_EQ(converted.size(), 5u);
  for (const auto i : {0, 2}) {
    EXPECT_EQ(converted[i].type, ptsbe::TraceInstructionType::Gate);
    EXPECT_EQ(converted[i].name, "x");
    EXPECT_EQ(converted[i].targets, (std::vector<std::size_t>{1}));
    EXPECT_TRUE(converted[i].controls.empty());
  }
  EXPECT_EQ(converted[1].name, "ry");
  EXPECT_EQ(converted[1].controls, (std::vector<std::size_t>{1}));
  EXPECT_EQ(converted[1].targets, (std::vector<std::size_t>{0}));
  EXPECT_EQ(converted[1].params, (std::vector<double>{0.37}));
  EXPECT_EQ(converted[3].type, ptsbe::TraceInstructionType::Noise);
  EXPECT_EQ(converted[4].type, ptsbe::TraceInstructionType::Measurement);
}

CUDAQ_TEST(TraceConversionTest, MixedControlsKeepOrder) {
  cudaq::Trace trace;
  trace.appendInstruction("swap", {}, {{2, 3}, {2, 0}, {2, 2}},
                          {{2, 1}, {2, 4}}, {0, 1, 0});
  trace.appendMeasurement("mz", {{2, 4}});
  const auto converted =
      cudaq::ptsbe::detail::buildPTSBETrace(trace, cudaq::noise_model{});

  // Only q3 and q2 are open: X(q3), X(q2), SWAP, X(q2), X(q3).
  // The central gate retains the original control order, including closed q0.
  ASSERT_EQ(converted.size(), 6u);
  EXPECT_EQ(converted[0].targets, (std::vector<std::size_t>{3}));
  EXPECT_EQ(converted[1].targets, (std::vector<std::size_t>{2}));
  EXPECT_EQ(converted[2].name, "swap");
  EXPECT_EQ(converted[2].controls, (std::vector<std::size_t>{3, 0, 2}));
  EXPECT_EQ(converted[2].targets, (std::vector<std::size_t>{1, 4}));
  EXPECT_EQ(converted[3].targets, (std::vector<std::size_t>{2}));
  EXPECT_EQ(converted[4].targets, (std::vector<std::size_t>{3}));
}
