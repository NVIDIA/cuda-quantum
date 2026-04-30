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
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <cmath>
#include <numeric>

// Use QPP simulator for testing samplePTSBE
using QppSimulator =
    QppCircuitSimulatorTester<nvqir::QppCircuitSimulator<qpp::ket>>;

const cudaq::ptsbe::PTSBETrace kHadamardTrace = {
    {cudaq::ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}}};
const cudaq::ptsbe::PTSBETrace kXTrace = {
    {cudaq::ptsbe::TraceInstructionType::Gate, "x", {0}, {}, {}}};

/// samplePTSBEGeneric throws without ExecutionContext
CUDAQ_TEST(ExecutePTSBETest, ThrowsWithoutExecutionContext) {
  QppSimulator sim;

  cudaq::ptsbe::PTSBatch batch;
  batch.trace = kHadamardTrace;
  batch.measureQubits = {0};

  cudaq::KrausTrajectory traj(0, {}, 1.0, 10);
  batch.trajectories.push_back(traj);

  try {
    cudaq::ptsbe::detail::samplePTSBEGeneric(sim, batch);
    FAIL() << "Expected an exception without ExecutionContext";
  } catch (...) {
  }
}

CUDAQ_TEST(ExecutePTSBETest, AggregateResultsTester) {
  // Default constructed results should be empty
  std::vector<cudaq::sample_result> results(10);
  auto resultEmpty = cudaq::ptsbe::detail::aggregateResults(results);
  EXPECT_EQ(resultEmpty.get_total_shots(), 0);
  EXPECT_EQ(resultEmpty.to_map().size(), 0);

  // Add some counts to the results
  results[0] = cudaq::ExecutionResult{{{"00", 10}, {"01", 5}}};
  auto resultWithCounts = cudaq::ptsbe::detail::aggregateResults(results);
  EXPECT_EQ(resultWithCounts.get_total_shots(), 15);
  EXPECT_EQ(resultWithCounts.count("00"), 10);
  EXPECT_EQ(resultWithCounts.count("01"), 5);

  // Add more counts to other results
  results[1] = cudaq::ExecutionResult{{{"00", 20}, {"10", 15}}};
  auto resultWithMoreCounts = cudaq::ptsbe::detail::aggregateResults(results);
  EXPECT_EQ(resultWithMoreCounts.get_total_shots(), 50);
  EXPECT_EQ(resultWithMoreCounts.count("00"), 30);
  EXPECT_EQ(resultWithMoreCounts.count("01"), 5);
  EXPECT_EQ(resultWithMoreCounts.count("10"), 15);
}

/// Single trajectory Hadamard circuit: execute H|0> and expect 50/50
CUDAQ_TEST(ExecutePTSBETest, SingleTrajectoryHadamard) {
  cudaq::set_random_seed(42);

  cudaq::ptsbe::PTSBatch batch;
  batch.trace = kHadamardTrace;
  batch.measureQubits = {0};

  cudaq::KrausTrajectory traj(0, {}, 1.0, 100);
  batch.trajectories.push_back(traj);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
  auto result = cudaq::ptsbe::detail::aggregateResults(results);

  std::size_t count0 = result.count("0");
  std::size_t count1 = result.count("1");
  EXPECT_GT(count0, 0u);
  EXPECT_GT(count1, 0u);
  EXPECT_EQ(count0 + count1, 100u);
}

/// Multiple trajectories: verify counts from all trajectories are aggregated
CUDAQ_TEST(ExecutePTSBETest, MultipleTrajectoryAggregation) {

  cudaq::ptsbe::PTSBatch batch;
  batch.trace = kXTrace;
  batch.measureQubits = {0};

  cudaq::KrausTrajectory traj1(0, {}, 0.7, 7);
  cudaq::KrausTrajectory traj2(1, {}, 0.3, 3);
  batch.trajectories.push_back(traj1);
  batch.trajectories.push_back(traj2);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);

  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].count("1"), 7u);
  EXPECT_EQ(results[1].count("1"), 3u);

  auto result = cudaq::ptsbe::detail::aggregateResults(results);
  EXPECT_EQ(result.count("1"), 10u);
  EXPECT_EQ(result.count("0"), 0u);
}

/// Zero-shot trajectories return empty result to maintain index correspondence
CUDAQ_TEST(ExecutePTSBETest, ZeroShotTrajectoryReturnsEmptyResult) {

  cudaq::ptsbe::PTSBatch batch;
  batch.trace = {{cudaq::ptsbe::TraceInstructionType::Gate, "y", {0}, {}, {}}};
  batch.measureQubits = {0};

  cudaq::KrausTrajectory zeroShot(0, {}, 0.5, 0);
  cudaq::KrausTrajectory normalShot(1, {}, 0.5, 10);
  batch.trajectories.push_back(zeroShot);
  batch.trajectories.push_back(normalShot);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);

  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].count("0"), 0u);
  EXPECT_EQ(results[0].count("1"), 0u);
  EXPECT_EQ(results[1].count("1"), 10u);

  auto result = cudaq::ptsbe::detail::aggregateResults(results);
  EXPECT_EQ(result.count("1"), 10u);
}

/// Empty inputs (trajectories or measureQubits) should return empty result
CUDAQ_TEST(ExecutePTSBETest, EmptyInputsReturnEmpty) {

  // Test 1: Empty trajectories vector
  {
    cudaq::ptsbe::PTSBatch batch;
    batch.trace = kHadamardTrace;
    batch.measureQubits = {0};

    auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
    EXPECT_TRUE(results.empty());
  }

  // Test 2: Empty measureQubits
  {
    cudaq::ptsbe::PTSBatch batch;
    batch.trace = kHadamardTrace;
    batch.measureQubits = {};

    cudaq::KrausTrajectory traj(0, {}, 1.0, 10);
    batch.trajectories.push_back(traj);

    auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
    EXPECT_TRUE(results.empty());
  }
}

/// Bell state: verify (|00> + |11>)/sqrt(2) distribution
CUDAQ_TEST(ExecutePTSBETest, BellStateDistribution) {
  cudaq::set_random_seed(42);

  cudaq::ptsbe::PTSBatch batch;
  batch.trace = {
      {cudaq::ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Gate, "x", {1}, {0}, {}},
  };
  batch.measureQubits = {0, 1};

  cudaq::KrausTrajectory traj(0, {}, 1.0, 100);
  batch.trajectories.push_back(traj);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
  auto result = cudaq::ptsbe::detail::aggregateResults(results);

  std::size_t count00 = result.count("00");
  std::size_t count11 = result.count("11");
  EXPECT_GT(count00, 0u);
  EXPECT_GT(count11, 0u);
  EXPECT_EQ(count00 + count11, 100u);

  EXPECT_EQ(result.count("01"), 0u);
  EXPECT_EQ(result.count("10"), 0u);
}

/// Trajectory with noise insertion: X error should flip the result
CUDAQ_TEST(ExecutePTSBETest, TrajectoryWithNoiseInsertion) {

  // Trace: [0] id gate on q0, [1] Noise(depol) on q0
  cudaq::ptsbe::PTSBatch batch;
  batch.trace = {
      {cudaq::ptsbe::TraceInstructionType::Gate, "id", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       cudaq::depolarization_channel(0.1)},
  };
  batch.measureQubits = {0};

  // Trajectory with X error (index 1) at trace position 1
  std::vector<cudaq::KrausSelection> selections = {
      cudaq::KrausSelection(1, {0}, "id", 1, true)};
  cudaq::KrausTrajectory traj(0, selections, 1.0, 10);
  batch.trajectories.push_back(traj);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
  auto result = cudaq::ptsbe::detail::aggregateResults(results);

  // I|0> with X error = X|0> = |1>
  EXPECT_EQ(result.count("1"), 10u);
}

/// Multi-qubit circuit with noise on specific qubit
CUDAQ_TEST(ExecutePTSBETest, MultiQubitWithSelectiveNoise) {

  // Trace: [0] X q0, [1] Noise q0, [2] X q1
  cudaq::ptsbe::PTSBatch batch;
  batch.trace = {
      {cudaq::ptsbe::TraceInstructionType::Gate, "x", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       cudaq::depolarization_channel(0.1)},
      {cudaq::ptsbe::TraceInstructionType::Gate, "x", {1}, {}, {}},
  };
  batch.measureQubits = {0, 1};

  // Trajectory 1: identity noise (no error), should give "11"
  std::vector<cudaq::KrausSelection> selectionsId = {
      cudaq::KrausSelection(1, {0}, "x", 0)};
  cudaq::KrausTrajectory traj1(0, selectionsId, 0.5, 10);

  // Trajectory 2: X error (index 1) on qubit 0 at trace position 1
  std::vector<cudaq::KrausSelection> selectionsX = {
      cudaq::KrausSelection(1, {0}, "x", 1, true)};
  cudaq::KrausTrajectory traj2(1, selectionsX, 0.5, 10);

  batch.trajectories.push_back(traj1);
  batch.trajectories.push_back(traj2);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
  auto result = cudaq::ptsbe::detail::aggregateResults(results);

  EXPECT_EQ(result.count("11"), 10u);
  EXPECT_EQ(result.count("01"), 10u);
}

/// Partial measurement: measure only one qubit of a two-qubit system
CUDAQ_TEST(ExecutePTSBETest, PartialMeasurement) {
  cudaq::set_random_seed(42);

  // Bell state
  cudaq::ptsbe::PTSBatch batch;
  batch.trace = {
      {cudaq::ptsbe::TraceInstructionType::Gate, "h", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Gate, "x", {1}, {0}, {}},
  };
  batch.measureQubits = {0};

  cudaq::KrausTrajectory traj(0, {}, 1.0, 100);
  batch.trajectories.push_back(traj);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
  auto result = cudaq::ptsbe::detail::aggregateResults(results);

  std::size_t count0 = result.count("0");
  std::size_t count1 = result.count("1");
  EXPECT_GT(count0, 0u);
  EXPECT_GT(count1, 0u);
  EXPECT_EQ(count0 + count1, 100u);
}

/// Measurement order: verify that measureQubits order affects bitstring order
CUDAQ_TEST(ExecutePTSBETest, MeasurementOrderAffectsBitstring) {

  // q0=1, q1=0
  std::vector<cudaq::ptsbe::TraceInstruction> trace = {
      {cudaq::ptsbe::TraceInstructionType::Gate, "x", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Gate, "id", {1}, {}, {}},
  };

  // First test: measure in order {0, 1}
  {
    cudaq::ptsbe::PTSBatch batch;
    batch.trace = trace;
    batch.measureQubits = {0, 1};

    cudaq::KrausTrajectory traj(0, {}, 1.0, 10);
    batch.trajectories.push_back(traj);

    auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
    auto result = cudaq::ptsbe::detail::aggregateResults(results);
    EXPECT_EQ(result.count("10"), 10u);
  }

  // Second test: measure in order {1, 0}
  {
    cudaq::ptsbe::PTSBatch batch;
    batch.trace = trace;
    batch.measureQubits = {1, 0};

    cudaq::KrausTrajectory traj(0, {}, 1.0, 10);
    batch.trajectories.push_back(traj);

    auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
    auto result = cudaq::ptsbe::detail::aggregateResults(results);
    EXPECT_EQ(result.count("01"), 10u);
  }
}

/// Verify state is properly reset between trajectories via setToZeroState()
CUDAQ_TEST(ExecutePTSBETest, MultipleTrajectoryStateReset) {

  // Trace: [0] id gate q0, [1] Noise q0
  cudaq::ptsbe::PTSBatch batch;
  batch.trace = {
      {cudaq::ptsbe::TraceInstructionType::Gate, "id", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Noise,
       "depolarization",
       {0},
       {},
       {},
       cudaq::depolarization_channel(0.1)},
  };
  batch.measureQubits = {0};

  // Trajectory 1: X error (index 1) flips to |1>
  std::vector<cudaq::KrausSelection> selectionsWithX = {
      cudaq::KrausSelection(1, {0}, "id", 1, true)};
  cudaq::KrausTrajectory trajWithError(0, selectionsWithX, 0.5, 10);

  // Trajectory 2: identity noise (no error), stays |0>
  std::vector<cudaq::KrausSelection> selectionsId = {
      cudaq::KrausSelection(1, {0}, "id", 0)};
  cudaq::KrausTrajectory trajNoError(1, selectionsId, 0.5, 10);

  batch.trajectories.push_back(trajWithError);
  batch.trajectories.push_back(trajNoError);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);

  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].count("1"), 10u);
  EXPECT_EQ(results[1].count("0"), 10u);

  auto result = cudaq::ptsbe::detail::aggregateResults(results);
  EXPECT_EQ(result.count("1"), 10u);
  EXPECT_EQ(result.count("0"), 10u);
}

/// Readout noise: bit flip applied after measurement flips X|0>=|1> to |0>
CUDAQ_TEST(ExecutePTSBETest, ReadoutNoiseBitFlipFlipsOutcome) {

  cudaq::ptsbe::PTSBatch batch;
  batch.trace = {
      {cudaq::ptsbe::TraceInstructionType::Gate, "x", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Measurement, "mz", {0}, {}, {}},
      {cudaq::ptsbe::TraceInstructionType::Noise,
       "bit_flip",
       {0},
       {},
       {},
       cudaq::bit_flip_channel(1.0)},
  };
  batch.trace[2].channel->generateUnitaryParameters();
  batch.measureQubits = {0};

  // X operator (index 1) at trace position 2 (the readout noise entry)
  std::vector<cudaq::KrausSelection> selections = {
      cudaq::KrausSelection(2, {0}, "mz", 1, true)};
  cudaq::KrausTrajectory traj(0, selections, 1.0, 10);
  batch.trajectories.push_back(traj);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);
  auto result = cudaq::ptsbe::detail::aggregateResults(results);

  EXPECT_EQ(result.count("0"), 10u);
  EXPECT_EQ(result.count("1"), 0u);
}

/// Verify samplePTSBEWithLifecycle produces correct results via production
/// dispatch path (runtime precision dispatch + concept check).
CUDAQ_TEST(ExecutePTSBETest, LifecycleDispatchProducesCorrectResults) {
  cudaq::ptsbe::PTSBatch batch;
  batch.trace = kXTrace;
  batch.measureQubits = {0};

  cudaq::KrausTrajectory traj(0, {}, 1.0, 10);
  batch.trajectories.push_back(traj);

  auto results = cudaq::ptsbe::detail::samplePTSBEWithLifecycle(batch);

  EXPECT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].count("1"), 10u);
}
