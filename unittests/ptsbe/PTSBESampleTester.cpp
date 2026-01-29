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
#include "cudaq/ptsbe/PTSBESample.h"

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

// MCM kernel tests require nvq++ compiler for proper registerNames population.
// See targettests/ptsbe/ for LIT tests that verify MCM detection with nvq++.

} // namespace

// ============================================================================
// T030: TRACE CAPTURE TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchCapturesBellCircuit) {
  auto batch = capturePTSBatch(bellKernel);

  std::size_t count = 0;
  for (const auto &inst : batch.kernel_trace) {
    (void)inst;
    ++count;
  }
  EXPECT_EQ(count, 2);
}

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchPreservesGateNames) {
  auto batch = capturePTSBatch(bellKernel);

  auto it = batch.kernel_trace.begin();
  EXPECT_EQ(it->name, "h");
  ++it;
  EXPECT_EQ(it->name, "x");
}

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchHandlesKernelArgs) {
  auto batch = capturePTSBatch(rotationKernel, 1.57);

  auto it = batch.kernel_trace.begin();
  EXPECT_EQ(it->name, "rx");
  EXPECT_NEAR(it->params[0], 1.57, 0.01);
}

CUDAQ_TEST(PTSBESampleTest, CapturePTSBatchHandlesEmptyKernel) {
  auto batch = capturePTSBatch(emptyKernel);

  std::size_t count = 0;
  for (const auto &inst : batch.kernel_trace) {
    (void)inst;
    ++count;
  }
  EXPECT_EQ(count, 0);
}

// ============================================================================
// T031: MCM DETECTION TESTS
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

// MCM kernel tests require nvq++ compiler for proper registerNames population.
// See targettests/ptsbe/ for LIT tests that verify MCM detection with nvq++.

// ============================================================================
// T032: PTSBATCH CONSTRUCTION TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, PTSBatchHasCorrectMeasureQubits) {
  auto batch = capturePTSBatch(bellKernel);
  EXPECT_EQ(batch.measure_qubits.size(), 2);
  EXPECT_EQ(batch.measure_qubits[0], 0);
  EXPECT_EQ(batch.measure_qubits[1], 1);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchFromGHZHas3Qubits) {
  auto batch = capturePTSBatch(ghzKernel);

  std::size_t count = 0;
  for (const auto &inst : batch.kernel_trace) {
    (void)inst;
    ++count;
  }
  EXPECT_EQ(count, 3);
  EXPECT_EQ(batch.measure_qubits.size(), 3);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchTrajectoriesEmptyForPOC) {
  auto batch = capturePTSBatch(bellKernel);
  EXPECT_TRUE(batch.trajectories.empty());
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchQubitInfoPreserved) {
  auto batch = capturePTSBatch(bellKernel);

  auto it = batch.kernel_trace.begin();
  EXPECT_EQ(it->targets.size(), 1);
  EXPECT_EQ(it->targets[0].id, 0);

  ++it;
  EXPECT_EQ(it->controls.size(), 1);
  EXPECT_EQ(it->controls[0].id, 0);
  EXPECT_EQ(it->targets.size(), 1);
  EXPECT_EQ(it->targets[0].id, 1);
}

// ============================================================================
// T033: DISPATCH TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, SampleWithPTSBEThrowsNotImplemented) {
  EXPECT_THROW(sampleWithPTSBE(bellKernel, 1000), std::runtime_error);
}

CUDAQ_TEST(PTSBESampleTest, DispatchErrorMentionsNotImplemented) {
  PTSBatch batch;
  batch.measure_qubits = {0, 1};

  try {
    dispatchPTSBE(batch);
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    EXPECT_TRUE(msg.find("Not implemented") != std::string::npos ||
                msg.find("not implemented") != std::string::npos);
  }
}

CUDAQ_TEST(PTSBESampleTest, FullInterceptFlowCapturesAndDispatches) {
  auto batch = capturePTSBatch(bellKernel);

  std::size_t count = 0;
  for (const auto &inst : batch.kernel_trace) {
    (void)inst;
    ++count;
  }
  EXPECT_GT(count, 0);
  EXPECT_FALSE(batch.measure_qubits.empty());

  EXPECT_THROW(dispatchPTSBE(batch), std::runtime_error);
}

// ============================================================================
// T044: CORE sample() INTEGRATION TESTS
// These tests verify the end-to-end integration with cudaq::sample()
// ============================================================================

// Test that cudaq::sample() with use_ptsbe=true dispatches to PTSBE path
// and includes diagnostic information in the error message
CUDAQ_TEST(PTSBESampleTest, CoreSampleWithUsePTSBEDispatchesToPTSBE) {
  cudaq::sample_options options;
  options.shots = 1000;
  options.use_ptsbe = true;

  // PTSBE dispatch should throw with diagnostic info showing successful capture
  try {
    cudaq::sample(options, bellKernel);
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    // Verify we got through PTSBE path with diagnostic info
    EXPECT_TRUE(msg.find("PTSBE dispatch successful") != std::string::npos);
    EXPECT_TRUE(msg.find("2 instructions") != std::string::npos);
    EXPECT_TRUE(msg.find("2 measure qubits") != std::string::npos);
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

// Test PTSBE dispatch with GHZ kernel to verify larger circuit capture
CUDAQ_TEST(PTSBESampleTest, CoreSamplePTSBECapturesGHZCircuit) {
  cudaq::sample_options options;
  options.shots = 1000;
  options.use_ptsbe = true;

  try {
    cudaq::sample(options, ghzKernel);
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    // GHZ has 3 gates (h, cx, cx) and 3 qubits
    EXPECT_TRUE(msg.find("3 instructions") != std::string::npos);
    EXPECT_TRUE(msg.find("3 measure qubits") != std::string::npos);
  }
}

// Test PTSBE dispatch with parameterized kernel
CUDAQ_TEST(PTSBESampleTest, CoreSamplePTSBEHandlesKernelArgs) {
  cudaq::sample_options options;
  options.shots = 1000;
  options.use_ptsbe = true;

  try {
    cudaq::sample(options, rotationKernel, 1.57);
    FAIL() << "Expected exception not thrown";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    // rotationKernel has 2 gates (rx, ry) and 2 qubits
    EXPECT_TRUE(msg.find("2 instructions") != std::string::npos);
    EXPECT_TRUE(msg.find("2 measure qubits") != std::string::npos);
  }
}

#endif // !CUDAQ_BACKEND_DM && !CUDAQ_BACKEND_STIM && !CUDAQ_BACKEND_TENSORNET
       // && !CUDAQ_BACKEND_CUSTATEVEC_FP32
