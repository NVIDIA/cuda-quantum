/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "QppCircuitSimulator.cpp"
#include "backends/QPPTester.h"
#include "cudaq/ptsbe/KrausTrajectory.h"
#include "cudaq/ptsbe/PTSBEInterface.h"
#include <cmath>

using namespace cudaq;
using namespace cudaq::ptsbe;

// Use QPP simulator for testing executePTSBE
using QppSimulator =
    QppCircuitSimulatorTester<nvqir::QppCircuitSimulator<qpp::ket>>;

/// Single trajectory Hadamard circuit: execute H|0> and expect 50/50
CUDAQ_TEST(ExecutePTSBETest, SingleTrajectoryHadamard) {
  QppSimulator sim;

  // Create trace: H gate on qubit 0
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});

  // Create PTSBatch with single trajectory (no noise)
  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};

  KrausTrajectory traj(0, {}, 1.0, 1000);
  batch.trajectories.push_back(traj);

  auto result = executePTSBE(sim, batch);

  // Hadamard creates superposition, expect ~50/50 with 10% tolerance
  std::size_t count0 = result.count("0");
  std::size_t count1 = result.count("1");
  EXPECT_GT(count0, 400u);
  EXPECT_LT(count0, 600u);
  EXPECT_GT(count1, 400u);
  EXPECT_LT(count1, 600u);
  EXPECT_EQ(count0 + count1, 1000u);
}

/// Multiple trajectories: verify counts from all trajectories are aggregated
CUDAQ_TEST(ExecutePTSBETest, MultipleTrajectoryAggregation) {
  QppSimulator sim;

  // Create trace: X gate (flips to |1>)
  Trace trace;
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 0)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};

  // Two trajectories with different shot counts
  KrausTrajectory traj1(0, {}, 0.7, 700);
  KrausTrajectory traj2(1, {}, 0.3, 300);
  batch.trajectories.push_back(traj1);
  batch.trajectories.push_back(traj2);

  auto result = executePTSBE(sim, batch);

  // X|0> = |1>, so all shots should measure "1"
  EXPECT_EQ(result.count("1"), 1000u);
  EXPECT_EQ(result.count("0"), 0u);
}

/// Zero-shot trajectories should be skipped during execution
CUDAQ_TEST(ExecutePTSBETest, SkipsZeroShotTrajectories) {
  QppSimulator sim;

  // Create trace: Y gate
  Trace trace;
  trace.appendInstruction("y", {}, {}, {QuditInfo(2, 0)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};

  // One trajectory with 0 shots, one with 500
  KrausTrajectory zeroShot(0, {}, 0.5, 0);
  KrausTrajectory normalShot(1, {}, 0.5, 500);
  batch.trajectories.push_back(zeroShot);
  batch.trajectories.push_back(normalShot);

  auto result = executePTSBE(sim, batch);

  // Y|0> = i|1>, measuring gives "1"
  EXPECT_EQ(result.count("1"), 500u);
}

/// Empty trajectories vector should return empty result
CUDAQ_TEST(ExecutePTSBETest, EmptyTrajectoriesReturnsEmpty) {
  QppSimulator sim;

  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};
  // No trajectories added

  auto result = executePTSBE(sim, batch);

  // Empty trajectories should produce empty result (no registers)
  EXPECT_TRUE(result.register_names().empty());
}

/// Bell state: verify (|00> + |11>)/sqrt(2) distribution
CUDAQ_TEST(ExecutePTSBETest, BellStateDistribution) {
  QppSimulator sim;

  // Create trace: H on q0, then CNOT (X with q0 as control, q1 as target)
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {QuditInfo(2, 0)}, {QuditInfo(2, 1)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0, 1};

  KrausTrajectory traj(0, {}, 1.0, 2000);
  batch.trajectories.push_back(traj);

  auto result = executePTSBE(sim, batch);

  // Bell state |00> + |11> should give ~50% each, with 10% tolerance
  std::size_t count00 = result.count("00");
  std::size_t count11 = result.count("11");
  EXPECT_GT(count00, 800u);
  EXPECT_LT(count00, 1200u);
  EXPECT_GT(count11, 800u);
  EXPECT_LT(count11, 1200u);
  EXPECT_EQ(count00 + count11, 2000u);

  // Should NOT see "01" or "10" (anti-correlated)
  EXPECT_EQ(result.count("01"), 0u);
  EXPECT_EQ(result.count("10"), 0u);
}

/// Trajectory with noise insertion: X error should flip the result
CUDAQ_TEST(ExecutePTSBETest, TrajectoryWithNoiseInsertion) {
  QppSimulator sim;

  // Create trace: identity gate (start in |0>)
  Trace trace;
  trace.appendInstruction("id", {}, {}, {QuditInfo(2, 0)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};

  // Add trajectory with X error after gate 0
  // TODO: Update to use named error enum (e.g. X_ERROR) when KrausOperatorType
  // is expanded beyond IDENTITY
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "x", static_cast<KrausOperatorType>(1))};
  KrausTrajectory traj(0, selections, 1.0, 100);
  batch.trajectories.push_back(traj);

  auto result = executePTSBE(sim, batch);

  // I|0> with X error = X|0> = |1>
  EXPECT_EQ(result.count("1"), 100u);
}

/// Multi-qubit circuit with noise on specific qubit
CUDAQ_TEST(ExecutePTSBETest, MultiQubitWithSelectiveNoise) {
  QppSimulator sim;

  // Create trace: X on both qubits
  Trace trace;
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 1)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0, 1};

  // Trajectory 1: no noise, should give "11"
  KrausTrajectory traj1(0, {}, 0.5, 100);

  // Trajectory 2: X error on qubit 0 after first gate
  // TODO: Update to use named error enum (e.g. X_ERROR) when KrausOperatorType
  // is expanded beyond IDENTITY
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "x", static_cast<KrausOperatorType>(1))};
  KrausTrajectory traj2(1, selections, 0.5, 100);

  batch.trajectories.push_back(traj1);
  batch.trajectories.push_back(traj2);

  auto result = executePTSBE(sim, batch);

  EXPECT_EQ(result.count("11"), 100u);
  EXPECT_EQ(result.count("01"), 100u);
}

/// Partial measurement: measure only one qubit of a two-qubit system
CUDAQ_TEST(ExecutePTSBETest, PartialMeasurement) {
  QppSimulator sim;

  // Create Bell state
  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("x", {}, {QuditInfo(2, 0)}, {QuditInfo(2, 1)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  // Only measure qubit 0
  batch.measureQubits = {0};

  KrausTrajectory traj(0, {}, 1.0, 1000);
  batch.trajectories.push_back(traj);

  auto result = executePTSBE(sim, batch);

  std::size_t count0 = result.count("0");
  std::size_t count1 = result.count("1");
  EXPECT_GT(count0, 400u);
  EXPECT_LT(count0, 600u);
  EXPECT_GT(count1, 400u);
  EXPECT_LT(count1, 600u);
  EXPECT_EQ(count0 + count1, 1000u);
}

/// Measurement order: verify that measureQubits order affects bitstring order
CUDAQ_TEST(ExecutePTSBETest, MeasurementOrderAffectsBitstring) {
  QppSimulator sim;

  // Create state where q0=1, q1=0 (X on q0 only)
  Trace trace;
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("id", {}, {}, {QuditInfo(2, 1)});

  // First test: measure in order {0, 1}
  {
    PTSBatch batch;
    batch.kernelTrace = trace;
    batch.measureQubits = {0, 1};

    KrausTrajectory traj(0, {}, 1.0, 100);
    batch.trajectories.push_back(traj);

    auto result = executePTSBE(sim, batch);
    // q0=1, q1=0, order {0,1} -> bitstring "10"
    EXPECT_EQ(result.count("10"), 100u);
  }

  // Second test: measure in order {1, 0}
  {
    PTSBatch batch;
    batch.kernelTrace = trace;
    batch.measureQubits = {1, 0};

    KrausTrajectory traj(0, {}, 1.0, 100);
    batch.trajectories.push_back(traj);

    auto result = executePTSBE(sim, batch);
    // q0=1, q1=0, order {1,0} -> bitstring "01"
    EXPECT_EQ(result.count("01"), 100u);
  }
}

/// Empty measureQubits: returns empty result
/// NOTE: Callers should use extractMeasureQubits() to populate batch.measureQubits
/// before calling executePTSBE. This test verifies the low-level behavior.
CUDAQ_TEST(ExecutePTSBETest, EmptyMeasureQubitsReturnsEmpty) {
  QppSimulator sim;

  // Create state where q0=1, q1=0
  Trace trace;
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 0)});
  trace.appendInstruction("id", {}, {}, {QuditInfo(2, 1)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  // Empty measureQubits - caller should have populated via extractMeasureQubits()
  batch.measureQubits = {};

  KrausTrajectory traj(0, {}, 1.0, 100);
  batch.trajectories.push_back(traj);

  auto result = executePTSBE(sim, batch);

  // Empty measureQubits returns empty result (no registers)
  EXPECT_TRUE(result.register_names().empty());
}