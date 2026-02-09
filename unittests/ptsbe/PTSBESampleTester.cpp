/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// PTSBE sample tests use the tracer context which behaves identically across
// all backends. Run only on qpp to avoid redundant test execution.
#if !defined(CUDAQ_BACKEND_DM) && !defined(CUDAQ_BACKEND_STIM) &&              \
    !defined(CUDAQ_BACKEND_TENSORNET) &&                                       \
    !defined(CUDAQ_BACKEND_CUSTATEVEC_FP32)

#include "CUDAQTestUtils.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/ptsbe/PTSBEOptions.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"
#include "cudaq/ptsbe/PTSBETrace.h"

using namespace cudaq::ptsbe;

namespace {

auto bellKernel = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  mz(q);
};

auto rotationKernel = [](double theta) __qpu__ {
  cudaq::qvector q(2);
  rx(theta, q[0]);
  ry(theta * 2, q[1]);
  mz(q);
};

auto ghzKernel = []() __qpu__ {
  cudaq::qvector q(3);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
  x<cudaq::ctrl>(q[1], q[2]);
  mz(q);
};

auto emptyKernel = []() __qpu__ {};

auto separatedMeasureKernel = []() __qpu__ {
  cudaq::qvector q(4);
  h(q[0]);
  x(q[2]);
  // Measure non-contiguous qubits: 0 and 2 (skip 1 and 3)
  mz(q[0]);
  mz(q[2]);
};

} // namespace

// ============================================================================
// TRACE CAPTURE TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchCapturesBellCircuit) {
  auto batch = capturePTSBatch(bellKernel);
  auto count =
      std::distance(batch.kernelTrace.begin(), batch.kernelTrace.end());
  EXPECT_EQ(count, 2);
}

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchPreservesGateNames) {
  auto batch = capturePTSBatch(bellKernel);

  auto it = batch.kernelTrace.begin();
  EXPECT_EQ(it->name, "h");
  ++it;
  EXPECT_EQ(it->name, "x");
}

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchHandlesKernelArgs) {
  auto batch = capturePTSBatch(rotationKernel, 1.57);

  auto it = batch.kernelTrace.begin();
  EXPECT_EQ(it->name, "rx");
  EXPECT_NEAR(it->params[0], 1.57, 0.01);
}

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchHandlesEmptyKernel) {
  auto batch = capturePTSBatch(emptyKernel);
  auto count =
      std::distance(batch.kernelTrace.begin(), batch.kernelTrace.end());
  EXPECT_EQ(count, 0);
}

// ============================================================================
// MCM DETECTION TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, NoMCMWithEmptyRegisterNames) {
  cudaq::ExecutionContext ctx("tracer");
  EXPECT_FALSE(hasMidCircuitMeasurements(ctx));
}

CUDAQ_TEST(PTSBESampleTest, DetectsMCMWithRegisterNames) {
  cudaq::ExecutionContext ctx("tracer");
  ctx.registerNames.push_back("mcm_0");
  EXPECT_TRUE(hasMidCircuitMeasurements(ctx));
}

CUDAQ_TEST(PTSBESampleTest, ThrowsForMCMContext) {
  cudaq::ExecutionContext ctx("tracer");
  ctx.registerNames.push_back("mcm_result");
  try {
    throwIfMidCircuitMeasurements(ctx);
    FAIL() << "Expected an exception for MCM context";
  } catch (...) {
  }
}

CUDAQ_TEST(PTSBESampleTest, NoThrowForValidContext) {
  cudaq::ExecutionContext ctx("tracer");
  EXPECT_NO_THROW(throwIfMidCircuitMeasurements(ctx));
}

// ============================================================================
// PTSBATCH CONSTRUCTION TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, PTSBatchHasCorrectMeasureQubits) {
  auto batch = capturePTSBatch(bellKernel);
  EXPECT_EQ(batch.measureQubits.size(), 2);
  EXPECT_EQ(batch.measureQubits[0], 0);
  EXPECT_EQ(batch.measureQubits[1], 1);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchFromGHZHas3Qubits) {
  auto batch = capturePTSBatch(ghzKernel);
  auto count =
      std::distance(batch.kernelTrace.begin(), batch.kernelTrace.end());
  EXPECT_EQ(count, 3);
  EXPECT_EQ(batch.measureQubits.size(), 3);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchTrajectoriesEmptyForPOC) {
  auto batch = capturePTSBatch(bellKernel);
  EXPECT_TRUE(batch.trajectories.empty());
}

// NOTE: Current POC implementation measures ALL qubits referenced in circuit.
// The kernel has 4 qubits but only 3 (0, 1, 2) are referenced in gate ops.
// Qubit 3 is allocated but never used in any gate, so trace sees only 0-2.
CUDAQ_TEST(PTSBESampleTest, PTSBatchSeparatedMeasureQubits) {
  auto batch = capturePTSBatch(separatedMeasureKernel);
  // POC: extractMeasureQubits returns all qubits seen in trace [0..maxQubitId]
  // Kernel uses h(q[0]) and x(q[2]), so max qubit ID is 2, giving 3 qubits
  EXPECT_EQ(batch.measureQubits.size(), 3);
  EXPECT_EQ(batch.measureQubits[0], 0);
  EXPECT_EQ(batch.measureQubits[1], 1);
  EXPECT_EQ(batch.measureQubits[2], 2);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchQubitInfoPreserved) {
  auto batch = capturePTSBatch(bellKernel);

  auto it = batch.kernelTrace.begin();
  EXPECT_EQ(it->targets.size(), 1);
  EXPECT_EQ(it->targets[0].id, 0);

  ++it;
  EXPECT_EQ(it->controls.size(), 1);
  EXPECT_EQ(it->controls[0].id, 0);
  EXPECT_EQ(it->targets.size(), 1);
  EXPECT_EQ(it->targets[0].id, 1);
}

// ============================================================================
// DISPATCH TESTS
// ============================================================================

// With no trajectories generated (noise model has no channels matching the
// circuit gates at the trajectory level), ptsbe::sample returns an empty
// result.
CUDAQ_TEST(PTSBESampleTest, SampleWithPTSBEReturnsEmptyWithNoTrajectories) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  auto result = sample(noise, 1000, bellKernel);
  // No trajectories = empty result
  EXPECT_EQ(result.size(), 0);
}

// Test that a batch with valid trace but no trajectories returns empty results
CUDAQ_TEST(PTSBESampleTest, ExecuteWithEmptyTrajectoriesReturnsEmpty) {
  // Capture a valid trace first
  auto batch = capturePTSBatch(bellKernel);
  // Clear trajectories (they're already empty from capturePTSBatch)
  EXPECT_TRUE(batch.trajectories.empty());

  // No trajectories - should return empty results
  auto results = samplePTSBEWithLifecycle(batch);
  EXPECT_TRUE(results.empty());
}

CUDAQ_TEST(PTSBESampleTest, FullInterceptFlowCapturesTrace) {
  auto batch = capturePTSBatch(bellKernel);
  auto count =
      std::distance(batch.kernelTrace.begin(), batch.kernelTrace.end());
  EXPECT_GT(count, 0);
  EXPECT_FALSE(batch.measureQubits.empty());

  // With no trajectories, should return empty (not throw)
  auto results = samplePTSBEWithLifecycle(batch);
  EXPECT_TRUE(results.empty());
}

// ============================================================================
// PTSBE SAMPLE API TESTS
// ============================================================================

// Test that sample() with an empty noise model throws
CUDAQ_TEST(PTSBESampleTest, PTSBESampleRequiresNoiseModel) {
  cudaq::noise_model emptyNoise;
  // No channels added - noise model is empty

  try {
    sample(emptyNoise, 1000, bellKernel);
    FAIL() << "Expected exception not thrown";
  } catch (...) {
  }
}

// Test that capturePTSBatch correctly captures GHZ circuit structure
CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchCapturesGHZCircuit) {
  auto batch = capturePTSBatch(ghzKernel);
  auto count =
      std::distance(batch.kernelTrace.begin(), batch.kernelTrace.end());
  // GHZ has 3 gates (h, cx, cx)
  EXPECT_EQ(count, 3);
  // 3 qubits
  EXPECT_EQ(batch.measureQubits.size(), 3);
}

// Test that capturePTSBatch correctly handles parameterized kernels
CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchHandlesParameterizedKernel) {
  auto batch = capturePTSBatch(rotationKernel, 1.57);
  auto count =
      std::distance(batch.kernelTrace.begin(), batch.kernelTrace.end());
  // rotationKernel has 2 gates (rx, ry)
  EXPECT_EQ(count, 2);
  // 2 qubits
  EXPECT_EQ(batch.measureQubits.size(), 2);
}

// ============================================================================
// TRACE OUTPUT INTEGRATION TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, SampleWithTraceOutputPopulatesTrace) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.return_trace = true;

  auto result = sample(options, bellKernel);

  ASSERT_TRUE(result.has_trace());
  const auto &trace = result.trace();

  EXPECT_GT(trace.instructions.size(), 0);
  // Bell circuit: h, x -> 2 gates
  EXPECT_EQ(trace.count_instructions(TraceInstructionType::Gate), 2);
  // Noise on h gate -> at least 1 noise instruction
  EXPECT_GE(trace.count_instructions(TraceInstructionType::Noise), 1);
  // 2 qubits measured
  EXPECT_EQ(trace.count_instructions(TraceInstructionType::Measurement), 2);

  // First instruction should be the h gate
  EXPECT_EQ(trace.instructions[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(trace.instructions[0].name, "h");

  // Noise instructions should have a channel attached
  for (const auto &inst : trace.instructions) {
    if (inst.type == TraceInstructionType::Noise) {
      EXPECT_TRUE(inst.channel.has_value());
      EXPECT_TRUE(inst.channel->is_unitary_mixture());
    }
  }
}

CUDAQ_TEST(PTSBESampleTest, SampleWithoutTraceOutputHasNoTrace) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  // return_trace defaults to false

  auto result = sample(options, bellKernel);

  EXPECT_FALSE(result.has_trace());
}

// ============================================================================
// PTSBE SAMPLE RESULT TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, SampleResultDefaultConstruction) {
  sample_result r;
  EXPECT_FALSE(r.has_trace());
}

CUDAQ_TEST(PTSBESampleTest, SampleResultMoveFromBaseResult) {
  cudaq::CountsDictionary counts;
  counts["00"] = 50;
  counts["11"] = 50;
  cudaq::sample_result baseResult{cudaq::ExecutionResult(counts)};

  sample_result r(std::move(baseResult));
  EXPECT_FALSE(r.has_trace());
  // Inherited data should be accessible
  EXPECT_GT(r.size(), 0);
}

CUDAQ_TEST(PTSBESampleTest, SampleResultConstructionWithTrace) {
  cudaq::CountsDictionary counts;
  counts["00"] = 40;
  counts["11"] = 60;
  cudaq::sample_result baseResult{cudaq::ExecutionResult(counts)};

  PTSBETrace trace;
  trace.instructions.push_back(
      TraceInstruction{TraceInstructionType::Gate, "h", {0}, {}, {}});
  trace.instructions.push_back(
      TraceInstruction{TraceInstructionType::Measurement, "mz", {0}, {}, {}});

  sample_result r(std::move(baseResult), std::move(trace));

  ASSERT_TRUE(r.has_trace());
  EXPECT_EQ(r.trace().instructions.size(), 2);
  EXPECT_EQ(r.trace().instructions[0].name, "h");
  EXPECT_EQ(r.trace().instructions[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(r.trace().instructions[1].type, TraceInstructionType::Measurement);
}

CUDAQ_TEST(PTSBESampleTest, SampleResultInheritedAccess) {
  cudaq::CountsDictionary counts;
  counts["00"] = 50;
  counts["11"] = 50;
  cudaq::sample_result baseResult{cudaq::ExecutionResult(counts)};

  sample_result r(std::move(baseResult));

  EXPECT_GT(r.size(), 0);
  auto mp = r.most_probable();
  EXPECT_TRUE(mp == "00" || mp == "11");
}

CUDAQ_TEST(PTSBESampleTest, SampleResultTraceThrowsWhenNotPresent) {
  sample_result r;
  EXPECT_THROW(r.trace(), std::runtime_error);
}

CUDAQ_TEST(PTSBESampleTest, SampleResultSetTrace) {
  sample_result r;
  EXPECT_FALSE(r.has_trace());

  PTSBETrace trace;
  trace.instructions.push_back(
      TraceInstruction{TraceInstructionType::Gate, "h", {0}, {}, {}});
  trace.instructions.push_back(TraceInstruction{
      TraceInstructionType::Noise, "depolarizing", {0}, {}, {0.01}});
  trace.instructions.push_back(
      TraceInstruction{TraceInstructionType::Measurement, "mz", {0}, {}, {}});

  r.set_trace(std::move(trace));

  ASSERT_TRUE(r.has_trace());
  EXPECT_EQ(r.trace().instructions.size(), 3);
  EXPECT_EQ(r.trace().instructions[0].name, "h");
  EXPECT_EQ(r.trace().instructions[1].name, "depolarizing");
  EXPECT_EQ(r.trace().instructions[2].name, "mz");
}

#endif // !CUDAQ_BACKEND_DM && !CUDAQ_BACKEND_STIM && !CUDAQ_BACKEND_TENSORNET
       // && !CUDAQ_BACKEND_CUSTATEVEC_FP32
