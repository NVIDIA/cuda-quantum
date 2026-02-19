/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/KrausTrajectory.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cmath>

using namespace cudaq;
using namespace cudaq::ptsbe;

/// Verify convertTrace handles multi-gate PTSBE trace correctly
CUDAQ_TEST(MergeTasksWithTrajectoryTest, ConvertTraceMultiGate) {
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {0}, {}},
  };

  auto tasks = convertTrace<double>(ptsbeTrace);

  ASSERT_EQ(tasks.size(), 3u);

  EXPECT_EQ(tasks[0].operationName, "h");
  EXPECT_EQ(tasks[0].targets.size(), 1u);
  EXPECT_EQ(tasks[0].targets[0], 0u);
  EXPECT_TRUE(tasks[0].controls.empty());

  EXPECT_EQ(tasks[1].operationName, "x");
  EXPECT_EQ(tasks[1].targets.size(), 1u);
  EXPECT_EQ(tasks[1].targets[0], 1u);
  EXPECT_TRUE(tasks[1].controls.empty());

  EXPECT_EQ(tasks[2].operationName, "x");
  EXPECT_EQ(tasks[2].controls.size(), 1u);
  EXPECT_EQ(tasks[2].controls[0], 0u);
  EXPECT_EQ(tasks[2].targets.size(), 1u);
  EXPECT_EQ(tasks[2].targets[0], 1u);
}

/// Verify convertTrace preserves gate parameters
CUDAQ_TEST(MergeTasksWithTrajectoryTest, ConvertTracePreservesParameters) {
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "rx", {0}, {}, {M_PI / 2}},
      {ptsbe::TraceInstructionType::Gate, "rz", {1}, {}, {M_PI / 4}},
  };

  auto tasks = convertTrace<double>(ptsbeTrace);

  ASSERT_EQ(tasks.size(), 2u);
  EXPECT_EQ(tasks[0].parameters.size(), 1u);
  EXPECT_NEAR(tasks[0].parameters[0], M_PI / 2, 1e-12);
  EXPECT_EQ(tasks[1].parameters.size(), 1u);
  EXPECT_NEAR(tasks[1].parameters[0], M_PI / 4, 1e-12);
}

/// Verify convertTrace skips Noise and Measurement entries
CUDAQ_TEST(MergeTasksWithTrajectoryTest, ConvertTraceSkipsNoiseAndMeasurement) {
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
      {ptsbe::TraceInstructionType::Measurement, "mz", {0, 1}, {}, {}},
  };

  auto tasks = convertTrace<double>(ptsbeTrace);

  ASSERT_EQ(tasks.size(), 2u);
  EXPECT_EQ(tasks[0].operationName, "h");
  EXPECT_EQ(tasks[1].operationName, "x");
}

/// Verify mergeTasksWithTrajectory returns gate tasks unchanged when no noise
CUDAQ_TEST(MergeTasksWithTrajectoryTest, NoNoiseInsertions) {
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
  };

  KrausTrajectory trajectory(0, {}, 1.0, 100);

  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  ASSERT_EQ(merged.size(), 2u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "x");
}

/// Verify single noise insertion at its trace position
CUDAQ_TEST(MergeTasksWithTrajectoryTest, SingleNoiseInsertion) {
  // Trace: [0] H on q0, [1] Noise(depol) on q0, [2] X on q1
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
  };

  // Z error (index 3) at trace position 1 (the Noise entry)
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", static_cast<KrausOperatorType>(3))};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  // Should be: H, noise(Z), X
  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "z");
  EXPECT_EQ(merged[1].targets[0], 0u);
  EXPECT_EQ(merged[2].operationName, "x");
}

/// Verify two consecutive noise entries at the same gate
CUDAQ_TEST(MergeTasksWithTrajectoryTest, MultipleNoiseEntriesAfterGate) {
  // Trace: [0] H on q0, [1] Noise on q0, [2] Noise on q1
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       depolarization_channel(0.1)},
  };

  // X on qubit 0 at trace pos 1, Z on qubit 1 at trace pos 2
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", static_cast<KrausOperatorType>(1)),
      KrausSelection(2, {1}, "h", static_cast<KrausOperatorType>(3))};
  KrausTrajectory trajectory(0, selections, 0.05, 5);

  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "x");
  EXPECT_EQ(merged[1].targets[0], 0u);
  EXPECT_EQ(merged[2].operationName, "z");
  EXPECT_EQ(merged[2].targets[0], 1u);
}

/// Verify invalid circuit_location throws error
CUDAQ_TEST(MergeTasksWithTrajectoryTest, InvalidCircuitLocationThrows) {
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
  };

  // circuit_location = 5 is beyond the trace
  std::vector<KrausSelection> selections = {
      KrausSelection(5, {0}, "h", static_cast<KrausOperatorType>(2))};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  try {
    mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);
    FAIL() << "Expected an exception for invalid circuit_location";
  } catch (...) {
  }
}

/// Verify noise at the last trace position works
CUDAQ_TEST(MergeTasksWithTrajectoryTest, NoiseAtLastPosition) {
  // Trace: [0] H on q0, [1] X on q1, [2] Noise on q1
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       depolarization_channel(0.1)},
  };

  // Z error at trace position 2 (the Noise entry)
  std::vector<KrausSelection> selections = {
      KrausSelection(2, {1}, "x", static_cast<KrausOperatorType>(3))};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  // Should be: H, X, noise(Z)
  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "x");
  EXPECT_EQ(merged[2].operationName, "z");
}

/// Verify identity noise inserts the channel's identity unitary
CUDAQ_TEST(MergeTasksWithTrajectoryTest, IdentityNoiseInsertion) {
  // Trace: [0] H on q0, [1] Noise on q0, [2] X on q1
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
  };

  // IDENTITY noise (index 0) at trace position 1
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", KrausOperatorType::IDENTITY)};
  KrausTrajectory trajectory(0, selections, 0.9, 90);

  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "id");
  EXPECT_EQ(merged[1].targets[0], 0u);
  ASSERT_EQ(merged[1].matrix.size(), 4u);
  EXPECT_NEAR(merged[1].matrix[0].real(), 1.0, 1e-6);
  EXPECT_NEAR(merged[1].matrix[3].real(), 1.0, 1e-6);
  EXPECT_NEAR(std::abs(merged[1].matrix[1]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(merged[1].matrix[2]), 0.0, 1e-6);
  EXPECT_EQ(merged[2].operationName, "x");
}

/// Verify mixed identity and error noise insertions
CUDAQ_TEST(MergeTasksWithTrajectoryTest, MixedIdentityAndErrorNoise) {
  // Trace: [0] H q0, [1] Noise q0, [2] X q1, [3] Noise q1, [4] Z q0
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "z", {0}, {}, {}},
  };

  // IDENTITY at trace pos 1, Y error (index 2) at trace pos 3
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", KrausOperatorType::IDENTITY),
      KrausSelection(3, {1}, "x", static_cast<KrausOperatorType>(2))};
  KrausTrajectory trajectory(0, selections, 0.2, 20);

  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  // H, noise(I), X, noise(Y), Z
  ASSERT_EQ(merged.size(), 5u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "id");
  EXPECT_EQ(merged[2].operationName, "x");
  EXPECT_EQ(merged[3].operationName, "y");
  EXPECT_EQ(merged[3].targets[0], 1u);
  EXPECT_EQ(merged[4].operationName, "z");
}

/// Verify empty trace produces empty task list
CUDAQ_TEST(MergeTasksWithTrajectoryTest, EmptyTrace) {
  std::vector<TraceInstruction> ptsbeTrace;

  auto tasks = convertTrace<double>(ptsbeTrace);
  EXPECT_TRUE(tasks.empty());

  KrausTrajectory trajectory(0, {}, 1.0, 100);
  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);
  EXPECT_TRUE(merged.empty());
}

/// Verify noise after every gate in the circuit
CUDAQ_TEST(MergeTasksWithTrajectoryTest, NoiseOnEveryGate) {
  // Trace: [0] H q0, [1] Noise q0, [2] X q1, [3] Noise q1
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       depolarization_channel(0.1)},
      {ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
      {ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {1},
       {},
       {},
       depolarization_channel(0.1)},
  };

  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", static_cast<KrausOperatorType>(3)),
      KrausSelection(3, {1}, "x", static_cast<KrausOperatorType>(1))};
  KrausTrajectory trajectory(0, selections, 0.01, 1);

  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  // H, noise(Z), X, noise(X)
  ASSERT_EQ(merged.size(), 4u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "z");
  EXPECT_EQ(merged[2].operationName, "x");
  EXPECT_EQ(merged[3].operationName, "x");
}

/// Verify measurement entries are skipped during merge
CUDAQ_TEST(MergeTasksWithTrajectoryTest, MeasurementsSkipped) {
  // Trace: [0] H q0, [1] Measurement mz q0
  std::vector<TraceInstruction> ptsbeTrace = {
      {ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {ptsbe::TraceInstructionType::Measurement, "mz", {0}, {}, {}},
  };

  KrausTrajectory trajectory(0, {}, 1.0, 100);
  auto merged = mergeTasksWithTrajectory<double>(ptsbeTrace, trajectory);

  ASSERT_EQ(merged.size(), 1u);
  EXPECT_EQ(merged[0].operationName, "h");
}
