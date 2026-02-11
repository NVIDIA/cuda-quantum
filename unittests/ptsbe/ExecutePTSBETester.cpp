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

using namespace cudaq;
using namespace cudaq::ptsbe;

// Use QPP simulator for testing samplePTSBE
using QppSimulator =
    QppCircuitSimulatorTester<nvqir::QppCircuitSimulator<qpp::ket>>;

/// Test helper: execute PTSBE with lifecycle management on a direct simulator.
/// This encapsulates the context setup, qubit allocation, execution, and
/// cleanup that would otherwise need to be repeated in each test.
template <typename SimulatorType>
std::vector<cudaq::sample_result> runPTSBETest(SimulatorType &sim,
                                               const PTSBatch &batch) {
  cudaq::ExecutionContext ctx("sample", batch.totalShots());
  cudaq::detail::setExecutionContext(&ctx);
  sim.configureExecutionContext(ctx);
  sim.allocateQubits(batch.kernelTrace.getNumQudits());

  std::vector<cudaq::sample_result> results;
  if constexpr (PTSBECapable<SimulatorType>) {
    results = sim.sampleWithPTSBE(batch);
  } else {
    results = samplePTSBEGeneric(sim, batch);
  }

  std::vector<std::size_t> qubitIds(batch.kernelTrace.getNumQudits());
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  sim.deallocateQubits(qubitIds);
  sim.finalizeExecutionContext(ctx);
  cudaq::detail::resetExecutionContext();
  return results;
}

/// samplePTSBEGeneric throws without ExecutionContext
CUDAQ_TEST(ExecutePTSBETest, ThrowsWithoutExecutionContext) {
  QppSimulator sim;

  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};

  KrausTrajectory traj(0, {}, 1.0, 100);
  batch.trajectories.push_back(traj);

  // No setExecutionContext() - should throw
  try {
    samplePTSBEGeneric(sim, batch);
    FAIL() << "Expected an exception without ExecutionContext";
  } catch (...) {
  }
}

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

  auto results = runPTSBETest(sim, batch);
  auto result = aggregateResults(results);

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

  auto results = runPTSBETest(sim, batch);

  // Verify per-trajectory results
  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].count("1"), 700u);
  EXPECT_EQ(results[1].count("1"), 300u);

  // Verify aggregation
  auto result = aggregateResults(results);
  EXPECT_EQ(result.count("1"), 1000u);
  EXPECT_EQ(result.count("0"), 0u);
}

/// Zero-shot trajectories return empty result to maintain index correspondence
CUDAQ_TEST(ExecutePTSBETest, ZeroShotTrajectoryReturnsEmptyResult) {
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

  auto results = runPTSBETest(sim, batch);

  // Results maintain index correspondence with trajectories
  EXPECT_EQ(results.size(), 2u);
  // results[0] is empty (zero shots) - count returns 0 for any bitstring
  EXPECT_EQ(results[0].count("0"), 0u);
  EXPECT_EQ(results[0].count("1"), 0u);
  // results[1] has actual data: Y|0> = i|1>
  EXPECT_EQ(results[1].count("1"), 500u);

  auto result = aggregateResults(results);
  EXPECT_EQ(result.count("1"), 500u);
}

/// Empty inputs (trajectories or measureQubits) should return empty result
CUDAQ_TEST(ExecutePTSBETest, EmptyInputsReturnEmpty) {
  QppSimulator sim;

  Trace trace;
  trace.appendInstruction("h", {}, {}, {QuditInfo(2, 0)});

  // Test 1: Empty trajectories vector
  {
    PTSBatch batch;
    batch.kernelTrace = trace;
    batch.measureQubits = {0};
    // No trajectories added

    auto results = runPTSBETest(sim, batch);
    EXPECT_TRUE(results.empty());
  }

  // Test 2: Empty measureQubits
  {
    PTSBatch batch;
    batch.kernelTrace = trace;
    batch.measureQubits = {};

    KrausTrajectory traj(0, {}, 1.0, 100);
    batch.trajectories.push_back(traj);

    auto results = runPTSBETest(sim, batch);
    EXPECT_TRUE(results.empty());
  }
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

  auto results = runPTSBETest(sim, batch);
  auto result = aggregateResults(results);

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

  // Noise site: depolarization on the id gate at circuit location 0
  batch.noise_sites.push_back(
      NoisePoint{0, {0}, "id", depolarization_channel(0.1)});

  // Trajectory with X error (index 1 in depolarization) after gate 0
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "id", static_cast<KrausOperatorType>(1))};
  KrausTrajectory traj(0, selections, 1.0, 100);
  batch.trajectories.push_back(traj);

  auto results = runPTSBETest(sim, batch);
  auto result = aggregateResults(results);

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

  // Noise site: depolarization on the first x gate at circuit location 0
  batch.noise_sites.push_back(
      NoisePoint{0, {0}, "x", depolarization_channel(0.1)});

  // Trajectory 1: identity noise (no error), should give "11"
  std::vector<KrausSelection> selectionsId = {
      KrausSelection(0, {0}, "x", KrausOperatorType::IDENTITY)};
  KrausTrajectory traj1(0, selectionsId, 0.5, 100);

  // Trajectory 2: X error (index 1) on qubit 0 after first gate
  std::vector<KrausSelection> selectionsX = {
      KrausSelection(0, {0}, "x", static_cast<KrausOperatorType>(1))};
  KrausTrajectory traj2(1, selectionsX, 0.5, 100);

  batch.trajectories.push_back(traj1);
  batch.trajectories.push_back(traj2);

  auto results = runPTSBETest(sim, batch);
  auto result = aggregateResults(results);

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

  auto results = runPTSBETest(sim, batch);
  auto result = aggregateResults(results);

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

    auto results = runPTSBETest(sim, batch);
    auto result = aggregateResults(results);
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

    auto results = runPTSBETest(sim, batch);
    auto result = aggregateResults(results);
    // q0=1, q1=0, order {1,0} -> bitstring "01"
    EXPECT_EQ(result.count("01"), 100u);
  }
}

/// Verify state is properly reset between trajectories via setToZeroState()
/// This ensures state doesn't leak from one trajectory to the next.
CUDAQ_TEST(ExecutePTSBETest, MultipleTrajectoryStateReset) {
  QppSimulator sim;

  // Create trace: identity gate only (no state change from |0>)
  Trace trace;
  trace.appendInstruction("id", {}, {}, {QuditInfo(2, 0)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};

  // Noise site: depolarization on the id gate at circuit location 0
  batch.noise_sites.push_back(
      NoisePoint{0, {0}, "id", depolarization_channel(0.1)});

  // Trajectory 1: X error (index 1) flips to |1>
  std::vector<KrausSelection> selectionsWithX = {
      KrausSelection(0, {0}, "id", static_cast<KrausOperatorType>(1))};
  KrausTrajectory trajWithError(0, selectionsWithX, 0.5, 100);

  // Trajectory 2: identity noise (no error), stays |0>
  std::vector<KrausSelection> selectionsId = {
      KrausSelection(0, {0}, "id", KrausOperatorType::IDENTITY)};
  KrausTrajectory trajNoError(1, selectionsId, 0.5, 100);

  batch.trajectories.push_back(trajWithError);
  batch.trajectories.push_back(trajNoError);

  auto results = runPTSBETest(sim, batch);

  // Verify per-trajectory results (confirms state reset between trajectories)
  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0].count("1"), 100u); // Trajectory 1: I|0> + X = |1>
  EXPECT_EQ(results[1].count("0"), 100u); // Trajectory 2: I|0> = |0>

  // Verify aggregation
  auto result = aggregateResults(results);
  EXPECT_EQ(result.count("1"), 100u);
  EXPECT_EQ(result.count("0"), 100u);
}

/// Mock simulator that implements sampleWithPTSBE for testing concept dispatch.
/// Delegates to the generic samplePTSBEGeneric fallback via the base class.
class MockPTSBESimulator : public QppSimulator {
public:
  // Counter to verify sampleWithPTSBE was called
  mutable std::size_t sampleWithPTSBECallCount = 0;

  /// Custom PTSBE implementation that delegates to generic fallback.
  /// In a real optimized simulator, this would do batched execution.
  std::vector<cudaq::sample_result> sampleWithPTSBE(const PTSBatch &batch) {
    ++sampleWithPTSBECallCount;

    // Use generic implementation via base class
    return samplePTSBEGeneric(*this, batch);
  }
};

/// Verify concept dispatch routes to custom implementation and generic fallback
/// produces equivalent results
CUDAQ_TEST(ExecutePTSBETest, ConceptDispatchAndGenericEquivalence) {
  MockPTSBESimulator mock;
  QppSimulator generic; // Does NOT satisfy PTSBECapable concept

  // Create trace: X gate (deterministic |1>)
  Trace trace;
  trace.appendInstruction("x", {}, {}, {QuditInfo(2, 0)});

  PTSBatch batch;
  batch.kernelTrace = trace;
  batch.measureQubits = {0};

  KrausTrajectory traj(0, {}, 1.0, 100);
  batch.trajectories.push_back(traj);

  // Test 1: Concept dispatch routes to mock.sampleWithPTSBE
  EXPECT_EQ(mock.sampleWithPTSBECallCount, 0u);
  auto mockResults = runPTSBETest(mock, batch);
  EXPECT_EQ(mock.sampleWithPTSBECallCount, 1u);

  // Test 2: Generic fallback for non-capable simulator
  auto genericResults = runPTSBETest(generic, batch);

  // Both should produce equivalent results (X|0> = |1>)
  EXPECT_EQ(mockResults.size(), genericResults.size());
  EXPECT_EQ(mockResults[0].count("1"), 100u);
  EXPECT_EQ(genericResults[0].count("1"), 100u);
}
