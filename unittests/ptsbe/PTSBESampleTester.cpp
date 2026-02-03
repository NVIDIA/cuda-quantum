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
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/ptsbe/PTSBESampleIntegration.h"

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
  EXPECT_THROW(throwIfMidCircuitMeasurements(ctx), std::runtime_error);
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

// PTSBE is now fully implemented. With no trajectories generated (POC doesn't
// have noise model integration yet), it returns an empty result.
CUDAQ_TEST(PTSBESampleTest, SampleWithPTSBEReturnsEmptyWithNoTrajectories) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));
  auto &platform = cudaq::get_platform();
  platform.set_noise(&noise);

  auto result = sampleWithPTSBE(bellKernel, 1000);
  // No trajectories = empty result
  EXPECT_EQ(result.size(), 0);

  platform.reset_noise();
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
// CORE sample() INTEGRATION TESTS
// ============================================================================

// Test that cudaq::sample() with use_ptsbe=true requires a noise model
CUDAQ_TEST(PTSBESampleTest, CoreSampleWithUsePTSBERequiresNoiseModel) {
  cudaq::sample_options options;
  options.shots = 1000;
  options.use_ptsbe = true;
  // No noise model set

  // PTSBE requires noise model - should throw
  try {
    cudaq::sample(options, bellKernel);
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("noise model") != std::string::npos);
  }
}

// Test that use_ptsbe=false uses normal sample path (no exception)
CUDAQ_TEST(PTSBESampleTest, CoreSampleWithoutUsePTSBEUsesNormalPath) {
  cudaq::sample_options options;
  options.shots = 100;
  options.use_ptsbe = false;

  // Normal path should succeed without throwing
  auto result = cudaq::sample(options, bellKernel);
  EXPECT_GT(result.size(), 0);
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

// Test that sample_async() with use_ptsbe=true requires a noise model
CUDAQ_TEST(PTSBESampleTest, AsyncSampleWithUsePTSBERequiresNoiseModel) {
  cudaq::sample_options options;
  options.shots = 1000;
  options.use_ptsbe = true;
  // No noise model set

  try {
    auto future = cudaq::sample_async(options, 0, bellKernel);
    future.get();
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("noise model") != std::string::npos);
  }
}

// Test async PTSBE with noise model (returns empty since no trajectories)
CUDAQ_TEST(PTSBESampleTest, AsyncSampleWithPTSBEAndNoiseModel) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::depolarization_channel(0.01));

  cudaq::sample_options options;
  options.shots = 100;
  options.use_ptsbe = true;
  options.noise = noise;

  // Should not throw - PTSBE path with valid noise model
  auto future = cudaq::sample_async(options, 0, bellKernel);
  auto result = future.get();
  // With no trajectories generated (POC), result is empty
  EXPECT_EQ(result.size(), 0);
}

// Test that broadcast sample with use_ptsbe=true requires a noise model
CUDAQ_TEST(PTSBESampleTest, BroadcastSampleWithUsePTSBERequiresNoiseModel) {
  cudaq::sample_options options;
  options.shots = 100;
  options.use_ptsbe = true;
  // No noise model set

  auto params = cudaq::make_argset(std::vector<double>{0.5, 1.0, 1.5});

  EXPECT_THROW(cudaq::sample(options, rotationKernel, std::move(params)),
               std::runtime_error);
}

// Test broadcast PTSBE with noise model (returns empty results since no
// trajectories)
CUDAQ_TEST(PTSBESampleTest, BroadcastSampleWithPTSBEAndNoiseModel) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("rx", cudaq::depolarization_channel(0.01));

  cudaq::sample_options options;
  options.shots = 100;
  options.use_ptsbe = true;
  options.noise = noise;

  auto params = cudaq::make_argset(std::vector<double>{0.5, 1.0, 1.5});

  // Should not throw - PTSBE path with valid noise model
  auto results = cudaq::sample(options, rotationKernel, std::move(params));
  // Returns one result per parameter set
  EXPECT_EQ(results.size(), 3);
  // With no trajectories generated (POC), each result is empty
  for (const auto &result : results) {
    EXPECT_EQ(result.size(), 0);
  }
}

// Test broadcast PTSBE result count matches parameter count
CUDAQ_TEST(PTSBESampleTest, BroadcastPTSBEResultCountMatchesParams) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("rx", cudaq::depolarization_channel(0.01));

  cudaq::sample_options options;
  options.shots = 50;
  options.use_ptsbe = true;
  options.noise = noise;

  // Test with different parameter counts
  auto params5 =
      cudaq::make_argset(std::vector<double>{0.1, 0.2, 0.3, 0.4, 0.5});
  auto results5 = cudaq::sample(options, rotationKernel, std::move(params5));
  EXPECT_EQ(results5.size(), 5);

  auto params1 = cudaq::make_argset(std::vector<double>{1.57});
  auto results1 = cudaq::sample(options, rotationKernel, std::move(params1));
  EXPECT_EQ(results1.size(), 1);
}

#endif // !CUDAQ_BACKEND_DM && !CUDAQ_BACKEND_STIM && !CUDAQ_BACKEND_TENSORNET
       // && !CUDAQ_BACKEND_CUSTATEVEC_FP32
