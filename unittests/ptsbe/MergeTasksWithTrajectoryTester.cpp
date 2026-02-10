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

namespace {

/// Build a NoisePoint with a real depolarization channel at the given location.
NoisePoint makeDepolarizingNoisePoint(std::size_t circuit_location,
                                      std::vector<std::size_t> qubits,
                                      const std::string &op_name,
                                      double probability = 0.1) {
  depolarization_channel channel(probability);
  return NoisePoint{circuit_location, std::move(qubits), op_name,
                    std::move(channel)};
}

} // namespace

/// Verify convertTrace handles multi-gate kernels correctly
CUDAQ_TEST(MergeTasksWithTrajectoryTest, ConvertTraceMultiGate) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});
  trace.appendInstruction("x", {}, {QuditInfo(2, 0)}, {QuditInfo(2, 1)});

  auto tasks = convertTrace<double>(trace);

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
  Trace trace;
  trace.appendInstruction("rx", {M_PI / 2}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("rz", {M_PI / 4}, {}, {QuditInfo(2, 1)});

  auto tasks = convertTrace<double>(trace);

  ASSERT_EQ(tasks.size(), 2u);
  EXPECT_EQ(tasks[0].parameters.size(), 1u);
  EXPECT_NEAR(tasks[0].parameters[0], M_PI / 2, 1e-12);
  EXPECT_EQ(tasks[1].parameters.size(), 1u);
  EXPECT_NEAR(tasks[1].parameters[0], M_PI / 4, 1e-12);
}

/// Verify mergeTasksWithTrajectory returns base tasks unchanged when no noise
CUDAQ_TEST(MergeTasksWithTrajectoryTest, NoNoiseInsertions) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});

  auto baseTasks = convertTrace<double>(trace);
  KrausTrajectory trajectory(0, {}, 1.0, 100);

  auto merged = mergeTasksWithTrajectory<double>(baseTasks, trajectory, {});

  ASSERT_EQ(merged.size(), baseTasks.size());
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "x");
}

/// Verify single noise insertion is placed after the indicated gate
CUDAQ_TEST(MergeTasksWithTrajectoryTest, SingleNoiseInsertion) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});

  auto baseTasks = convertTrace<double>(trace);

  // Z error (index 3) after gate 0 (H) on qubit 0
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", static_cast<KrausOperatorType>(3))};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  std::vector<NoisePoint> noiseSites = {
      makeDepolarizingNoisePoint(0, {0}, "h")};

  auto merged =
      mergeTasksWithTrajectory<double>(baseTasks, trajectory, noiseSites);

  // Should be: H, noise(Z), X
  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "depolarization_channel[3]");
  EXPECT_EQ(merged[1].targets[0], 0u);
  EXPECT_EQ(merged[2].operationName, "x");
}

/// Verify multiple noise insertions at the same circuit location
CUDAQ_TEST(MergeTasksWithTrajectoryTest, MultipleInsertionsSameIndex) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});

  auto baseTasks = convertTrace<double>(trace);

  // Two noise operations after gate 0: X on qubit 0, then Z on qubit 1
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", static_cast<KrausOperatorType>(1)),
      KrausSelection(0, {1}, "h", static_cast<KrausOperatorType>(3))};
  KrausTrajectory trajectory(0, selections, 0.05, 5);

  std::vector<NoisePoint> noiseSites = {
      makeDepolarizingNoisePoint(0, {0}, "h"),
      makeDepolarizingNoisePoint(0, {1}, "h")};

  auto merged =
      mergeTasksWithTrajectory<double>(baseTasks, trajectory, noiseSites);

  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "depolarization_channel[1]");
  EXPECT_EQ(merged[1].targets[0], 0u);
  EXPECT_EQ(merged[2].operationName, "depolarization_channel[3]");
  EXPECT_EQ(merged[2].targets[0], 1u);
}

/// Verify invalid circuit_location throws error
CUDAQ_TEST(MergeTasksWithTrajectoryTest, InvalidCircuitLocationThrows) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});

  auto baseTasks = convertTrace<double>(trace);

  // circuit_location = 1 is invalid (only gate 0 exists)
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {0}, "h", static_cast<KrausOperatorType>(2))};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  std::vector<NoisePoint> noiseSites = {
      makeDepolarizingNoisePoint(1, {0}, "h")};

  try {
    mergeTasksWithTrajectory<double>(baseTasks, trajectory, noiseSites);
    FAIL() << "Expected an exception for invalid circuit_location";
  } catch (...) {
    // Expected: any exception type
  }
}

/// Verify noise at last valid gate index works
CUDAQ_TEST(MergeTasksWithTrajectoryTest, NoiseAtLastGate) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});

  auto baseTasks = convertTrace<double>(trace);

  // Noise after gate 1 (the last gate, index 1)
  std::vector<KrausSelection> selections = {
      KrausSelection(1, {1}, "x", static_cast<KrausOperatorType>(3))};
  KrausTrajectory trajectory(0, selections, 0.1, 10);

  std::vector<NoisePoint> noiseSites = {
      makeDepolarizingNoisePoint(1, {1}, "x")};

  auto merged =
      mergeTasksWithTrajectory<double>(baseTasks, trajectory, noiseSites);

  // Should be: H, X, noise(Z)
  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "x");
  EXPECT_EQ(merged[2].operationName, "depolarization_channel[3]");
}

/// Verify identity noise inserts the channel's identity unitary
CUDAQ_TEST(MergeTasksWithTrajectoryTest, IdentityNoiseInsertion) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});

  auto baseTasks = convertTrace<double>(trace);

  // IDENTITY noise (index 0) after gate 0
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", KrausOperatorType::IDENTITY)};
  KrausTrajectory trajectory(0, selections, 0.9, 90);

  std::vector<NoisePoint> noiseSites = {
      makeDepolarizingNoisePoint(0, {0}, "h")};

  auto merged =
      mergeTasksWithTrajectory<double>(baseTasks, trajectory, noiseSites);

  ASSERT_EQ(merged.size(), 3u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "depolarization_channel[0]");
  EXPECT_EQ(merged[1].targets[0], 0u);
  // The identity unitary from depolarization_channel is the 2x2 identity
  ASSERT_EQ(merged[1].matrix.size(), 4u);
  EXPECT_NEAR(merged[1].matrix[0].real(), 1.0, 1e-6);
  EXPECT_NEAR(merged[1].matrix[3].real(), 1.0, 1e-6);
  EXPECT_NEAR(std::abs(merged[1].matrix[1]), 0.0, 1e-6);
  EXPECT_NEAR(std::abs(merged[1].matrix[2]), 0.0, 1e-6);
  EXPECT_EQ(merged[2].operationName, "x");
}

/// Verify mixed identity and error noise insertions
CUDAQ_TEST(MergeTasksWithTrajectoryTest, MixedIdentityAndErrorNoise) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});
  trace.appendInstruction("z", {}, {}, {QuditInfo(2, 0)});

  auto baseTasks = convertTrace<double>(trace);

  // IDENTITY at gate 0, Y error (index 2) at gate 1
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", KrausOperatorType::IDENTITY),
      KrausSelection(1, {1}, "x", static_cast<KrausOperatorType>(2))};
  KrausTrajectory trajectory(0, selections, 0.2, 20);

  std::vector<NoisePoint> noiseSites = {
      makeDepolarizingNoisePoint(0, {0}, "h"),
      makeDepolarizingNoisePoint(1, {1}, "x")};

  auto merged =
      mergeTasksWithTrajectory<double>(baseTasks, trajectory, noiseSites);

  // H, noise(I), X, noise(Y), Z
  ASSERT_EQ(merged.size(), 5u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "depolarization_channel[0]");
  EXPECT_EQ(merged[2].operationName, "x");
  EXPECT_EQ(merged[3].operationName, "depolarization_channel[2]");
  EXPECT_EQ(merged[3].targets[0], 1u);
  EXPECT_EQ(merged[4].operationName, "z");
}

/// Verify empty trace produces empty task list
CUDAQ_TEST(MergeTasksWithTrajectoryTest, EmptyTrace) {
  Trace trace;
  auto baseTasks = convertTrace<double>(trace);

  EXPECT_TRUE(baseTasks.empty());

  KrausTrajectory trajectory(0, {}, 1.0, 100);
  auto merged = mergeTasksWithTrajectory<double>(baseTasks, trajectory, {});
  EXPECT_TRUE(merged.empty());
}

/// Verify noise after every gate in the circuit
CUDAQ_TEST(MergeTasksWithTrajectoryTest, NoiseOnEveryGate) {
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});

  auto baseTasks = convertTrace<double>(trace);

  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", static_cast<KrausOperatorType>(3)),
      KrausSelection(1, {1}, "x", static_cast<KrausOperatorType>(1))};
  KrausTrajectory trajectory(0, selections, 0.01, 1);

  std::vector<NoisePoint> noiseSites = {
      makeDepolarizingNoisePoint(0, {0}, "h"),
      makeDepolarizingNoisePoint(1, {1}, "x")};

  auto merged =
      mergeTasksWithTrajectory<double>(baseTasks, trajectory, noiseSites);

  // H, noise(Z), X, noise(X)
  ASSERT_EQ(merged.size(), 4u);
  EXPECT_EQ(merged[0].operationName, "h");
  EXPECT_EQ(merged[1].operationName, "depolarization_channel[3]");
  EXPECT_EQ(merged[2].operationName, "x");
  EXPECT_EQ(merged[3].operationName, "depolarization_channel[1]");
}
