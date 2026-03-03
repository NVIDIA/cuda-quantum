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
#include "cudaq/ptsbe/NoiseExtractor.h"
#include "cudaq/ptsbe/PTSBEExecutionData.h"
#include "cudaq/ptsbe/PTSBEOptions.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"
#include "cudaq/ptsbe/ShotAllocationStrategy.h"
#include "cudaq/ptsbe/strategies/ExhaustiveSamplingStrategy.h"

using namespace cudaq::ptsbe;
using namespace cudaq::ptsbe::detail;

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

struct xOp {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
  }
};

auto inlineNoiseKernel = []() __qpu__ {
  cudaq::qubit q;
  x(q);
  cudaq::apply_noise<cudaq::depolarization_channel>(0.1, q);
  mz(q);
};

} // namespace

// ============================================================================
// TRACE CAPTURE TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, TracePTSBatchCapturesBellCircuit) {
  auto batch = tracePTSBatch(bellKernel);
  // Bell circuit: h, x gates + mz measurements
  EXPECT_EQ(countInstructions(batch.trace, TraceInstructionType::Gate), 2);
}

CUDAQ_TEST(PTSBESampleTest, TracePTSBatchPreservesGateNames) {
  auto batch = tracePTSBatch(bellKernel);

  // Find Gate instructions
  std::vector<const TraceInstruction *> gates;
  for (const auto &inst : batch.trace)
    if (inst.type == TraceInstructionType::Gate)
      gates.push_back(&inst);

  ASSERT_GE(gates.size(), 2u);
  EXPECT_EQ(gates[0]->name, "h");
  EXPECT_EQ(gates[1]->name, "x");
}

CUDAQ_TEST(PTSBESampleTest, TracePTSBatchHandlesKernelArgs) {
  auto batch = tracePTSBatch(rotationKernel, 1.57);

  ASSERT_FALSE(batch.trace.empty());
  EXPECT_EQ(batch.trace[0].name, "rx");
  EXPECT_NEAR(batch.trace[0].params[0], 1.57, 0.01);
}

CUDAQ_TEST(PTSBESampleTest, TracePTSBatchHandlesEmptyKernel) {
  auto batch = tracePTSBatch(emptyKernel);
  EXPECT_TRUE(batch.trace.empty());
}

// ============================================================================
// MCM DETECTION TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, NoConditionalFeedbackWithEmptyRegisterNames) {
  cudaq::ExecutionContext ctx("tracer");
  EXPECT_FALSE(cudaq::detail::hasConditionalFeedback("", &ctx));
}

CUDAQ_TEST(PTSBESampleTest, DetectsConditionalFeedbackWithRegisterNames) {
  cudaq::ExecutionContext ctx("tracer");
  ctx.registerNames.push_back("mcm_0");
  EXPECT_TRUE(cudaq::detail::hasConditionalFeedback("", &ctx));
}

CUDAQ_TEST(PTSBESampleTest, ValidateKernelThrowsForMCMContext) {
  cudaq::ExecutionContext ctx("tracer");
  ctx.registerNames.push_back("mcm_result");
  try {
    validatePTSBEKernel("testKernel", ctx);
    FAIL() << "Expected an exception for MCM context";
  } catch (...) {
  }
}

CUDAQ_TEST(PTSBESampleTest, ValidateKernelNoThrowForValidContext) {
  cudaq::ExecutionContext ctx("tracer");
  EXPECT_NO_THROW(validatePTSBEKernel("testKernel", ctx));
}

CUDAQ_TEST(PTSBESampleTest, WarnNamedRegisters) {
  // __global__ only: no warning
  cudaq::ExecutionContext globalCtx("tracer");
  globalCtx.kernelTrace.appendMeasurement("mz", {{2, 0}}, "__global__");
  warnNamedRegisters("testKernel", globalCtx);
  EXPECT_FALSE(globalCtx.warnedNamedMeasurements);

  // Named register: sets flag
  cudaq::ExecutionContext namedCtx("tracer");
  namedCtx.kernelTrace.appendMeasurement("mz", {{2, 0}}, "my_register");
  warnNamedRegisters("testKernel", namedCtx);
  EXPECT_TRUE(namedCtx.warnedNamedMeasurements);

  // Second call is a no-op (flag already set)
  warnNamedRegisters("testKernel", namedCtx);
  EXPECT_TRUE(namedCtx.warnedNamedMeasurements);
}

// ============================================================================
// PTSBATCH CONSTRUCTION TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, PTSBatchHasCorrectMeasureQubits) {
  auto batch = tracePTSBatch(bellKernel);
  EXPECT_EQ(batch.measureQubits.size(), 2);
  EXPECT_EQ(batch.measureQubits[0], 0);
  EXPECT_EQ(batch.measureQubits[1], 1);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchFromGHZHas3Qubits) {
  auto batch = tracePTSBatch(ghzKernel);
  EXPECT_EQ(numQubits(batch.trace), 3);
  EXPECT_EQ(batch.measureQubits.size(), 3);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchTrajectoriesEmptyForPOC) {
  auto batch = tracePTSBatch(bellKernel);
  EXPECT_TRUE(batch.trajectories.empty());
}

// Kernel allocates 4 qubits but only measures q[0] and q[2].
// extractMeasureQubits derives the list from Measurement entries,
// so only the actually-measured qubits appear, in kernel order.
CUDAQ_TEST(PTSBESampleTest, PTSBatchSeparatedMeasureQubits) {
  auto batch = tracePTSBatch(separatedMeasureKernel);
  EXPECT_EQ(batch.measureQubits.size(), 2);
  EXPECT_EQ(batch.measureQubits[0], 0);
  EXPECT_EQ(batch.measureQubits[1], 2);
}

CUDAQ_TEST(PTSBESampleTest, PTSBatchQubitInfoPreserved) {
  auto batch = tracePTSBatch(bellKernel);

  // Find Gate instructions
  std::vector<const TraceInstruction *> gates;
  for (const auto &inst : batch.trace)
    if (inst.type == TraceInstructionType::Gate)
      gates.push_back(&inst);

  ASSERT_GE(gates.size(), 2u);
  // H gate targets qubit 0
  EXPECT_EQ(gates[0]->targets.size(), 1);
  EXPECT_EQ(gates[0]->targets[0], 0);
  // CNOT: control qubit 0, target qubit 1
  EXPECT_EQ(gates[1]->controls.size(), 1);
  EXPECT_EQ(gates[1]->controls[0], 0);
  EXPECT_EQ(gates[1]->targets.size(), 1);
  EXPECT_EQ(gates[1]->targets[0], 1);
}

// ============================================================================
// DISPATCH TESTS
// ============================================================================

// ptsbe::sample with matching noise model produces non-empty results
CUDAQ_TEST(PTSBESampleTest, SampleWithPTSBEReturnsResults) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  auto result = sample(noise, 50, bellKernel);
  // With noise matching circuit gates, trajectories are generated
  EXPECT_GT(result.size(), 0);
  EXPECT_EQ(result.get_total_shots(), 50);
}

// Kernel with no explicit mz() should implicitly measure all qubits,
// matching standard cudaq::sample() behavior.
CUDAQ_TEST(PTSBESampleTest, SampleImplicitMeasureAllQubits) {
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, cudaq::bit_flip_channel(0.1));

  auto result = sample(noise, 50, xOp{});
  EXPECT_EQ(result.get_total_shots(), 50);
  EXPECT_GT(result.size(), 0u);
}

// Test that a batch with valid trace but no trajectories returns empty results
CUDAQ_TEST(PTSBESampleTest, ExecuteWithEmptyTrajectoriesReturnsEmpty) {
  // Capture a valid trace first
  auto batch = tracePTSBatch(bellKernel);
  // Clear trajectories (they're already empty from tracePTSBatch)
  EXPECT_TRUE(batch.trajectories.empty());

  // No trajectories - should return empty results
  auto results = samplePTSBEWithLifecycle(batch);
  EXPECT_TRUE(results.empty());
}

CUDAQ_TEST(PTSBESampleTest, FullInterceptFlowCapturesTrace) {
  auto batch = tracePTSBatch(bellKernel);
  EXPECT_FALSE(batch.trace.empty());
  EXPECT_FALSE(batch.measureQubits.empty());

  // With no trajectories, should return empty (not throw)
  auto results = samplePTSBEWithLifecycle(batch);
  EXPECT_TRUE(results.empty());
}

// ============================================================================
// PTSBE SAMPLE API TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, RunSamplingPTSBEAcceptsShotAllocationStrategy) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("x", cudaq::depolarization_channel(0.01));

  auto &platform = cudaq::get_platform();
  platform.set_noise(&noise);

  PTSBEOptions ptsbe_opts;
  ptsbe_opts.shot_allocation =
      ShotAllocationStrategy(ShotAllocationStrategy::Type::UNIFORM);
  auto result =
      runSamplingPTSBE([]() { bellKernel(); }, platform,
                       cudaq::getKernelName(bellKernel), 50, ptsbe_opts);

  platform.reset_noise();
  (void)result;
}

CUDAQ_TEST(PTSBESampleTest, PTSBESampleWithShotAllocationOption) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.02));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.shot_allocation =
      ShotAllocationStrategy(ShotAllocationStrategy::Type::UNIFORM);

  auto result = sample(options, bellKernel);
  (void)result;
}

// Test that tracePTSBatch correctly captures GHZ circuit structure
CUDAQ_TEST(PTSBESampleTest, TracePTSBatchCapturesGHZCircuit) {
  auto batch = tracePTSBatch(ghzKernel);
  // GHZ has 3 gates (h, cx, cx)
  EXPECT_EQ(countInstructions(batch.trace, TraceInstructionType::Gate), 3);
  // 3 qubits
  EXPECT_EQ(batch.measureQubits.size(), 3);
}

// Test that tracePTSBatch correctly handles parameterized kernels
CUDAQ_TEST(PTSBESampleTest, TracePTSBatchHandlesParameterizedKernel) {
  auto batch = tracePTSBatch(rotationKernel, 1.57);
  // rotationKernel has 2 gates (rx, ry)
  EXPECT_EQ(countInstructions(batch.trace, TraceInstructionType::Gate), 2);
  // 2 qubits
  EXPECT_EQ(batch.measureQubits.size(), 2);
}

// End-to-end test for PTSBE pipeline.

// Pipeline:
// 1. capture trace
// 2. extract noise sites
// 3. generate trajectories
// 4. allocate shots
// 5. run PTSBE execution
// 6. aggregate results
// 7. verify allocation, verify total counts equals total shots.
CUDAQ_TEST(PTSBESampleTest, E2E_GenerateTrajectoriesAllocateShotsRunSample) {
  // Noise model: depolarization on "h"
  cudaq::noise_model noise;
  noise.add_channel("h", {0}, cudaq::depolarization_channel(0.01));

  // Capture raw trace from kernel
  cudaq::ExecutionContext traceCtx("tracer");
  auto &platform = cudaq::get_platform();
  platform.with_execution_context(traceCtx, [&]() { bellKernel(); });
  cleanupTracerQubits(traceCtx.kernelTrace);

  // Build PTSBE trace with noise model and extract noise sites
  PTSBatch batch;
  batch.trace = buildPTSBETrace(traceCtx.kernelTrace, noise);
  batch.measureQubits = extractMeasureQubits(batch.trace);
  EXPECT_FALSE(batch.trace.empty());
  EXPECT_FALSE(batch.measureQubits.empty());

  auto extraction = extractNoiseSites(batch.trace);
  ASSERT_GT(extraction.noise_sites.size(), 0)
      << "Expected at least one noise site for h gate";
  EXPECT_TRUE(extraction.all_unitary_mixtures);

  // Generate trajectories
  ExhaustiveSamplingStrategy strategy;
  const std::size_t max_trajectories = 24;
  auto trajectories =
      strategy.generateTrajectories(extraction.noise_sites, max_trajectories);
  ASSERT_GT(trajectories.size(), 0) << "Expected at least one trajectory";

  // Assign to batch
  batch.trajectories = std::move(trajectories);

  // Allocate shots across trajectories
  const std::size_t total_shots = 50;
  ShotAllocationStrategy shot_strategy(
      ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(batch.trajectories, total_shots, shot_strategy);

  // Verify allocation
  std::size_t sum_shots = 0;
  for (const auto &t : batch.trajectories)
    sum_shots += t.num_shots;
  EXPECT_EQ(sum_shots, total_shots);

  // Execute PTSBE and aggregate
  auto results = samplePTSBEWithLifecycle(batch);
  EXPECT_EQ(results.size(), batch.trajectories.size());

  auto result = aggregateResults(results);

  // Total counts should equal total shots
  EXPECT_EQ(result.get_total_shots(), total_shots);
}

// ============================================================================
// EXECUTION DATA INTEGRATION TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, ExecutionDataWarningEmittedOnceOnlyWhenAccessed) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options noDataOptions;
  noDataOptions.shots = 1;
  noDataOptions.noise = noise;

  testing::internal::CaptureStdout();
  auto noDataResult = sample(noDataOptions, bellKernel);
  EXPECT_FALSE(noDataResult.has_execution_data());
  auto noWarning = testing::internal::GetCapturedStdout();
  constexpr std::string_view warningToken = "PTSBE execution data API is "
                                            "experimental";
  EXPECT_EQ(noWarning.find(warningToken), std::string::npos);

  sample_options withDataOptions;
  withDataOptions.shots = 1;
  withDataOptions.noise = noise;
  withDataOptions.ptsbe.return_execution_data = true;

  testing::internal::CaptureStdout();
  auto withDataResult1 = sample(withDataOptions, bellKernel);
  ASSERT_TRUE(withDataResult1.has_execution_data());
  EXPECT_NO_THROW((void)withDataResult1.execution_data());
  EXPECT_NO_THROW((void)withDataResult1.execution_data());

  auto withDataResult2 = sample(withDataOptions, bellKernel);
  ASSERT_TRUE(withDataResult2.has_execution_data());
  EXPECT_NO_THROW((void)withDataResult2.execution_data());
  auto warningOutput = testing::internal::GetCapturedStdout();

  const auto first = warningOutput.find(warningToken);
  ASSERT_NE(first, std::string::npos) << "Expected warning was not emitted.";
  const auto second =
      warningOutput.find(warningToken, first + warningToken.size());
  EXPECT_EQ(second, std::string::npos) << "Warning emitted more than once.";
}

CUDAQ_TEST(PTSBESampleTest, SampleWithExecutionDataPopulatesData) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;

  auto result = sample(options, bellKernel);

  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();

  EXPECT_GT(data.instructions.size(), 0);
  // Bell circuit: h, x -> 2 gates
  EXPECT_EQ(data.count_instructions(TraceInstructionType::Gate), 2);
  // Noise on h gate -> at least 1 noise instruction
  EXPECT_GE(data.count_instructions(TraceInstructionType::Noise), 1);
  // 2 qubits measured
  EXPECT_EQ(data.count_instructions(TraceInstructionType::Measurement), 2);

  // First instruction should be the h gate
  EXPECT_EQ(data.instructions[0].type, TraceInstructionType::Gate);
  EXPECT_EQ(data.instructions[0].name, "h");

  // Noise instructions should have a channel attached
  for (const auto &inst : data.instructions) {
    if (inst.type == TraceInstructionType::Noise) {
      EXPECT_TRUE(inst.channel.has_value());
      EXPECT_TRUE(inst.channel->is_unitary_mixture());
    }
  }
}

CUDAQ_TEST(PTSBESampleTest, SampleWithoutExecutionDataHasNoData) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  // return_execution_data defaults to false

  auto result = sample(options, bellKernel);

  EXPECT_FALSE(result.has_execution_data());
}

// ============================================================================
// PTSBE SAMPLE RESULT TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, SampleResultDefaultConstruction) {
  sample_result r;
  EXPECT_FALSE(r.has_execution_data());
}

CUDAQ_TEST(PTSBESampleTest, SampleResultMoveFromBaseResult) {
  cudaq::CountsDictionary counts;
  counts["00"] = 50;
  counts["11"] = 50;
  cudaq::sample_result baseResult{cudaq::ExecutionResult(counts)};

  sample_result r(std::move(baseResult));
  EXPECT_FALSE(r.has_execution_data());
  // Inherited data should be accessible
  EXPECT_GT(r.size(), 0);
}

CUDAQ_TEST(PTSBESampleTest, SampleResultConstructionWithExecutionData) {
  cudaq::CountsDictionary counts;
  counts["00"] = 40;
  counts["11"] = 60;
  cudaq::sample_result baseResult{cudaq::ExecutionResult(counts)};

  PTSBEExecutionData executionData;
  executionData.instructions.push_back(
      TraceInstruction{TraceInstructionType::Gate, "h", {0}, {}, {}});
  executionData.instructions.push_back(
      TraceInstruction{TraceInstructionType::Measurement, "mz", {0}, {}, {}});

  sample_result r(std::move(baseResult), std::move(executionData));

  ASSERT_TRUE(r.has_execution_data());
  EXPECT_EQ(r.execution_data().instructions.size(), 2);
  EXPECT_EQ(r.execution_data().instructions[0].name, "h");
  EXPECT_EQ(r.execution_data().instructions[0].type,
            TraceInstructionType::Gate);
  EXPECT_EQ(r.execution_data().instructions[1].type,
            TraceInstructionType::Measurement);
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

CUDAQ_TEST(PTSBESampleTest, SampleResultExecutionDataThrowsWhenNotPresent) {
  sample_result r;
  try {
    r.execution_data();
    FAIL() << "expected an exception when execution data is not present";
  } catch (...) {
    // Expected: any exception
  }
}

CUDAQ_TEST(PTSBESampleTest, SampleResultSetExecutionData) {
  sample_result r;
  EXPECT_FALSE(r.has_execution_data());

  PTSBEExecutionData executionData;
  executionData.instructions.push_back(
      TraceInstruction{TraceInstructionType::Gate, "h", {0}, {}, {}});
  executionData.instructions.push_back(TraceInstruction{
      TraceInstructionType::Noise, "depolarizing", {0}, {}, {0.01}});
  executionData.instructions.push_back(
      TraceInstruction{TraceInstructionType::Measurement, "mz", {0}, {}, {}});

  r.set_execution_data(std::move(executionData));

  ASSERT_TRUE(r.has_execution_data());
  EXPECT_EQ(r.execution_data().instructions.size(), 3);
  EXPECT_EQ(r.execution_data().instructions[0].name, "h");
  EXPECT_EQ(r.execution_data().instructions[1].name, "depolarizing");
  EXPECT_EQ(r.execution_data().instructions[2].name, "mz");
}

CUDAQ_TEST(PTSBESampleTest, SampleAsyncWithOptions) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 50;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;

  auto future = sample_async(options, bellKernel);
  auto result = future.get();

  EXPECT_EQ(result.get_total_shots(), 50);
  ASSERT_TRUE(result.has_execution_data());
  EXPECT_GT(result.execution_data().instructions.size(), 0);
}

// Noise model goes out of scope before .get() -- must not crash because
// runSamplingAsyncPTSBE copies the noise model into the async lambda.
CUDAQ_TEST(PTSBESampleTest, SampleAsyncNoiseModelDestroyed) {
  async_sample_result future;
  {
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));
    future = sample_async(noise, 50, bellKernel);
  }
  auto result = future.get();
  EXPECT_EQ(result.get_total_shots(), 50);
  EXPECT_GT(result.size(), 0u);
}

// Exceptions thrown inside the async lambda must propagate to .get(),
// not cause std::terminate or std::future_error.
CUDAQ_TEST(PTSBESampleTest, SampleAsyncPropagatesException) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  auto &platform = cudaq::get_platform();
  platform.set_noise(&noise);

  auto future = runSamplingAsyncPTSBE(
      []() { throw std::runtime_error("injected async failure"); }, platform,
      "test_kernel", 10);

  platform.reset_noise();

  try {
    future.get();
    FAIL() << "Expected exception from async PTSBE sampling";
  } catch (const std::runtime_error &e) {
    EXPECT_NE(std::string(e.what()).find("injected async failure"),
              std::string::npos);
  } catch (const std::exception &e) {
    FAIL() << "Expected std::runtime_error, got: " << e.what();
  }
}

CUDAQ_TEST(PTSBESampleTest, BroadcastReturnsMultipleResults) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("rx", cudaq::depolarization_channel(0.01));
  noise.add_all_qubit_channel("ry", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 100;
  options.noise = noise;

  auto params = cudaq::make_argset(std::vector<double>{0.1, 0.5, 1.0});
  auto results = sample(options, rotationKernel, params);

  EXPECT_EQ(results.size(), 3);
  for (auto &r : results) {
    EXPECT_EQ(r.get_total_shots(), 100);
    EXPECT_FALSE(r.to_map().empty());
  }
}

CUDAQ_TEST(PTSBESampleTest, BroadcastResultCountMatchesParams) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("rx", cudaq::depolarization_channel(0.01));
  noise.add_all_qubit_channel("ry", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 50;
  options.noise = noise;

  std::vector<double> angles = {0.0, 0.25, 0.5, 0.75, 1.0};
  auto params = cudaq::make_argset(angles);
  auto results = sample(options, rotationKernel, params);

  EXPECT_EQ(results.size(), angles.size());
}

CUDAQ_TEST(PTSBESampleTest, NoiseCheckSimple) {
  cudaq::set_random_seed(13);
  cudaq::kraus_channel depol({cudaq::complex{0.99498743710662, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {0.99498743710662, 0.0}},
                             {cudaq::complex{0.0, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.0, 0.0}},
                             {cudaq::complex{0.0, 0.0},
                              {0.0, -0.05773502691896258},
                              {0.0, 0.05773502691896258},
                              {0.0, 0.0}},
                             {cudaq::complex{0.05773502691896258, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {-0.05773502691896258, 0.0}});
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, depol);

  auto result = sample(noise, 500, xOp{});
  EXPECT_EQ(result.get_total_shots(), 500);
  EXPECT_EQ(result.size(), 2);

  cudaq::noise_model emptyNoise;
  result = sample(emptyNoise, 500, xOp{});
  EXPECT_EQ(result.size(), 0);
}

// ============================================================================
// READOUT NOISE TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, SampleWithMzNoiseBitFlipFullFlip) {
  cudaq::noise_model noise;
  noise.add_channel("mz", {0}, cudaq::bit_flip_channel(1.0));

  auto result = sample(noise, 50, xOp{});

  // X|0>=|1>, BitFlip(1.0) on mz flips to "0"
  EXPECT_GT(result.count("0"), 40u);
  EXPECT_EQ(result.get_total_shots(), 50u);
}

CUDAQ_TEST(PTSBESampleTest, ExecutionDataIncludesMzNoise) {
  cudaq::noise_model noise;
  noise.add_channel("mz", {0}, cudaq::bit_flip_channel(0.1));

  sample_options options;
  options.shots = 100;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;

  auto result = sample(options, xOp{});

  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();

  EXPECT_EQ(data.count_instructions(TraceInstructionType::Gate), 1);
  EXPECT_GE(data.count_instructions(TraceInstructionType::Noise), 1);
  EXPECT_EQ(data.count_instructions(TraceInstructionType::Measurement), 1);
}

// ============================================================================
// POPULATE EXECUTION DATA TESTS
// ============================================================================

CUDAQ_TEST(PTSBESampleTest, PopulateExecutionDataFiltersZeroShotTrajectories) {
  PTSBEExecutionData executionData;
  executionData.instructions.push_back(
      {TraceInstructionType::Noise, "depolarization", {0}, {}, {}});

  std::vector<cudaq::KrausTrajectory> trajectories;
  auto t0 = cudaq::KrausTrajectory(0, {}, 0.7, 0);
  t0.num_shots = 50;
  auto t1 = cudaq::KrausTrajectory(1, {}, 0.2, 0);
  t1.num_shots = 0;
  auto t2 = cudaq::KrausTrajectory(2, {}, 0.1, 0);
  t2.num_shots = 30;
  trajectories.push_back(std::move(t0));
  trajectories.push_back(std::move(t1));
  trajectories.push_back(std::move(t2));

  cudaq::CountsDictionary counts0{{"00", 30}, {"11", 20}};
  cudaq::CountsDictionary counts2{{"00", 20}, {"11", 10}};
  std::vector<cudaq::sample_result> results;
  results.push_back(cudaq::sample_result{cudaq::ExecutionResult{counts0}});
  results.push_back(
      cudaq::sample_result{cudaq::ExecutionResult{cudaq::CountsDictionary{}}});
  results.push_back(cudaq::sample_result{cudaq::ExecutionResult{counts2}});

  populateExecutionDataTrajectories(executionData, std::move(trajectories),
                                    std::move(results));

  EXPECT_EQ(executionData.trajectories.size(), 2u);
  EXPECT_EQ(executionData.trajectories[0].trajectory_id, 0u);
  EXPECT_EQ(executionData.trajectories[1].trajectory_id, 2u);
}

CUDAQ_TEST(PTSBESampleTest, ZeroShotTrajectoriesNotReturnedInE2E) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  sample_options options;
  options.shots = 10;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;
  options.ptsbe.max_trajectories = 1000;

  auto result = sample(options, bellKernel);

  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();

  for (const auto &traj : data.trajectories) {
    EXPECT_GT(traj.num_shots, 0u)
        << "Trajectory " << traj.trajectory_id << " has 0 shots in output";
  }
}

CUDAQ_TEST(PTSBESampleTest, InlineApplyNoise) {
  sample_options options;
  options.shots = 50;
  options.ptsbe.return_execution_data = true;

  auto result = sample(options, inlineNoiseKernel);

  EXPECT_EQ(result.get_total_shots(), 50u);
  EXPECT_GT(result.size(), 0u);

  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();

  auto gateCount = data.count_instructions(TraceInstructionType::Gate);
  EXPECT_EQ(gateCount, 1u);

  auto noiseCount = data.count_instructions(TraceInstructionType::Noise);
  EXPECT_GE(noiseCount, 1u);
}

#endif // !CUDAQ_BACKEND_DM && !CUDAQ_BACKEND_STIM && !CUDAQ_BACKEND_TENSORNET
       // && !CUDAQ_BACKEND_CUSTATEVEC_FP32
