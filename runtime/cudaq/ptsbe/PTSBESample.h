/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file PTSBESample.h
/// @brief PTSBE sample API and execution internals
///
/// Provides the `cudaq::ptsbe` sample API:
///   ptsbe::sample_options opts;
///   opts.noise = noise_model;
///   auto result = ptsbe::sample(opts, kernel, args...);
///
/// `ptsbe::sample()` returns `ptsbe::sample_result` (subclass of
/// `cudaq::sample_result`) which may optionally carry execution data.
///
/// Limitations:
/// - PTSBE does not support mid-circuit measurements (MCM)
/// - Dynamic circuits with conditional logic are rejected
/// - Supports only unitary mixture noise
///

#pragma once

#include "NoiseExtractor.h"
#include "PTSBEExecutionData.h"
#include "PTSBEOptions.h"
#include "PTSBESampleResult.h"
#include "PTSBESampler.h"
#include "ShotAllocationStrategy.h"
#include "common/ExecutionContext.h"
#include "common/Future.h"
#include "common/NoiseModel.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/platform.h"
#include "cudaq/platform/QuantumExecutionQueue.h"
#include "cudaq/runtime/logger/logger.h"
#include <future>
#include <optional>
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe {

/// @brief Check if kernel has conditional feedback (dynamic circuit)
///
/// PTSBE requires static circuits where the gate sequence is deterministic.
/// Dynamic circuits with measurement-dependent control flow cannot be
/// pre-trajectory sampled because the gate sequence depends on runtime
/// measurement outcomes.
///
/// Detection uses two mechanisms:
/// 1. MLIR-compiled kernels: Check registered quake code for
///    qubitMeasurementFeedback attribute
/// 2. Library mode: Check registerNames populated during tracing when
///    __nvqpp__MeasureResultBoolConversion is called
///
/// @param kernelName Name of the kernel (for MLIR lookup)
/// @param ctx ExecutionContext populated after tracing (for library mode)
/// @return true if conditional feedback detected
bool hasConditionalFeedback(const std::string &kernelName,
                            const ExecutionContext &ctx);

/// @brief Validate kernel eligibility for PTSBE execution
///
/// Checks all constraints required for PTSBE trajectory-based simulation:
/// - No conditional feedback on measurement results (dynamic circuits) or
/// mid-circuit measurements
///
/// @param kernelName Name of the kernel being validated
/// @param ctx ExecutionContext populated after kernel tracing
/// @throws std::runtime_error if kernel is not eligible for PTSBE
void validatePTSBEKernel(const std::string &kernelName,
                         const ExecutionContext &ctx);

/// @brief Validate platform preconditions for PTSBE execution
void validatePTSBEPreconditions(
    quantum_platform &platform,
    std::optional<std::size_t> qpu_id = std::nullopt);

/// @brief Build the PTSBE instruction sequence from a raw cudaq::Trace.
///
/// @param trace Raw circuit trace (may contain Gate, Noise, and Measurement)
/// @param noise_model Noise model used to resolve inline apply_noise channels
/// @return PTSBETrace with resolved channels for Noise entries
[[nodiscard]] PTSBETrace buildPTSBETrace(const cudaq::Trace &trace,
                                         const cudaq::noise_model &noise_model);

/// @brief Extract measured qubit IDs from the trace's Measurement entries.
///
/// Scans the trace for Measurement instructions and collects their target
/// qubit IDs in the order they first appear. Duplicates are suppressed so
/// each qubit appears at most once while preserving the kernel's measurement
/// ordering.
///
/// @param trace PTSBE trace containing Gate, Noise, and Measurement entries
/// @return Ordered, de-duplicated vector of measured qubit indices
std::vector<std::size_t>
extractMeasureQubits(std::span<const TraceInstruction> trace);

/// @brief Deallocate qubit IDs leaked by the tracer context on the simulator
///
/// In the MLIR/JIT path, qubit allocation
/// (__quantum__rt__qubit_allocate_array in NVQIR.cpp) goes directly to the
/// simulator's allocateQubits, which increments the simulator's
/// QuditIdTracker::currentId. However, CircuitSimulator::deallocateQubits
/// is a no-op when an execution context is set (including tracer), so the
/// kernel's qubit deallocation never returns IDs to the simulator's tracker.
///
/// Without cleanup, each PTSBE tracer pass accumulates qubit IDs on the
/// simulator (first call gets [0,1], next gets [2,3], etc.). This causes
/// noise model key mismatches (noise defined for qubit [0] but the trace
/// now has qubit [2]) and eventual memory exhaustion on density-matrix
/// simulators.
///
/// This function collects all qubit IDs from the kernel trace and
/// deallocates them from the simulator. Must be called AFTER
/// with_execution_context returns (when the execution context is null and
/// deallocateQubits will actually execute).
///
/// @param kernelTrace Captured kernel trace containing qubit IDs
void cleanupTracerQubits(const Trace &kernelTrace);

/// @brief Build PTSBEExecutionData with interleaved instructions (no
/// trajectories)
///
/// Converts the internal kernel trace into the user-facing
/// PTSBEExecutionData format. For each gate in the kernel trace, a Gate
/// instruction is added. If the noise model defines noise at that gate, a
/// Noise instruction follows. Measurement instructions are appended for all
/// measured qubits. The trajectories vector is left empty.
///
/// @param kernelTrace Captured kernel trace
/// @param noiseModel Noise model for identifying noise sites
/// @return PTSBEExecutionData with interleaved instructions and empty
///         trajectories
PTSBEExecutionData
buildExecutionDataInstructions(const cudaq::Trace &kernelTrace,
                               const noise_model &noiseModel);

/// @brief Populate trajectories on an existing PTSBEExecutionData
///
/// Takes ownership of trajectories and results. Remaps each KrausSelection's
/// circuit_location from noise-site index to the corresponding Noise
/// instruction index in PTSBEExecutionData.instructions (derived by scanning
/// the instruction list). Populates measurement_counts from per-trajectory
/// execution results.
///
/// @param executionData PTSBEExecutionData to populate (must have instructions
///        already set)
/// @param trajectories Executed trajectories
/// @param perTrajectoryResults Per-trajectory sample results
void populateExecutionDataTrajectories(
    PTSBEExecutionData &executionData,
    std::vector<cudaq::KrausTrajectory> trajectories,
    std::vector<cudaq::sample_result> perTrajectoryResults);

/// @brief Build complete PTSBatch with noise extraction and trajectory
/// generation
///
/// Extracts noise sites from the kernel trace, generates trajectories via the
/// configured strategy (or default probabilistic), and allocates shots.
///
/// @param kernelTrace Captured kernel trace (moved into the returned batch)
/// @param noiseModel Noise model for extracting noise sites
/// @param options PTSBE configuration options
/// @param shots Total number of shots to allocate
/// @return PTSBatch ready for execution
PTSBatch buildPTSBatchWithTrajectories(cudaq::Trace &&kernelTrace,
                                       const noise_model &noiseModel,
                                       const PTSBEOptions &options,
                                       std::size_t shots);

/// @brief Run PTSBE sampling (internal API matching runSampling pattern)
///
/// Captures the kernel trace, builds PTSBEExecutionData, generates
/// trajectories, executes them, and aggregates results. Optionally attaches
/// the execution data (with trajectories and measurement counts) to the
/// result when return_execution_data is enabled.
///
/// The noise model must be set on the platform before calling this function
/// (validated by validatePTSBEPreconditions).
///
/// @tparam KernelFunctor Wrapped kernel functor type
/// @param wrappedKernel Functor that invokes the quantum kernel
/// @param platform Reference to the quantum platform
/// @param kernelName Name of the kernel (for diagnostics and MCM detection)
/// @param shots Number of shots for trajectory allocation
/// @param options PTSBE configuration options
/// @return ptsbe::sample_result with optional execution data
/// @throws std::runtime_error if platform is not a simulator, noise model is
///         missing, or dynamic circuit detected
template <typename KernelFunctor>
sample_result runSamplingPTSBE(KernelFunctor &&wrappedKernel,
                               quantum_platform &platform,
                               const std::string &kernelName, std::size_t shots,
                               const PTSBEOptions &options = PTSBEOptions{}) {
  validatePTSBEPreconditions(platform);

  // Use platform noise if set; otherwise empty model
  static const cudaq::noise_model kEmptyNoiseModel;
  const auto *noisePtr = platform.get_noise();
  const auto &noiseModel = noisePtr ? *noisePtr : kEmptyNoiseModel;

  // Stage 0: Capture trace via ExecutionContext("tracer")
  cudaq::info("[ptsbe] Capturing circuit trace for kernel '{}'", kernelName);
  ExecutionContext traceCtx("tracer");
  platform.with_execution_context(traceCtx, [&]() { wrappedKernel(); });
  cleanupTracerQubits(traceCtx.kernelTrace);
  cudaq::info("[ptsbe] Trace captured: {} qubits, {} instructions",
              traceCtx.kernelTrace.getNumQudits(),
              traceCtx.kernelTrace.getNumInstructions());

  // Stage 1: Validate kernel eligibility (no dynamic circuits)
  validatePTSBEKernel(kernelName, traceCtx);

  // Stage 2: Build execution data if requested (skip overhead otherwise)
  std::optional<PTSBEExecutionData> executionData;
  if (options.return_execution_data)
    executionData =
        buildExecutionDataInstructions(traceCtx.kernelTrace, noiseModel);

  // Stage 3: Build PTSBatch with trajectory generation and shot allocation
  auto batch = buildPTSBatchWithTrajectories(std::move(traceCtx.kernelTrace),
                                             noiseModel, options, shots);
  cudaq::info("[ptsbe] Allocated {} shots across {} trajectories",
              batch.totalShots(), batch.trajectories.size());

  // Stage 4: Execute PTSBE with life-cycle management
  cudaq::info("[ptsbe] Executing batched simulation");
  auto perTrajectoryResults = samplePTSBEWithLifecycle(batch);

  // Stage 5: Aggregate per-trajectory results
  sample_result result(aggregateResults(perTrajectoryResults));

  // Stage 6: Attach trajectories and set execution data on result if requested
  if (executionData) {
    populateExecutionDataTrajectories(*executionData,
                                      std::move(batch.trajectories),
                                      std::move(perTrajectoryResults));
    result.set_execution_data(std::move(*executionData));
  }

  cudaq::info("[ptsbe] Complete: {} unique bitstrings from {} shots",
              result.size(), result.get_total_shots());
  return result;
}

/// @brief Capture kernel trace and construct PTSBatch (for testing)
///
/// Helper function that captures trace and builds PTSBatch without dispatching.
/// Used by tests to verify trace capture and batch construction independently
/// of execution. Builds the PTSBE trace with an empty noise model (no Noise
/// entries). To build with a noise model, use buildPTSBatchWithTrajectories.
///
/// @tparam QuantumKernel Quantum kernel type
/// @tparam Args Kernel argument types
/// @param kernel Quantum kernel to trace
/// @param args Kernel arguments
/// @return PTSBatch with trace, empty trajectories, and measureQubits
/// @throws std::runtime_error if MCM detected
template <typename QuantumKernel, typename... Args>
PTSBatch capturePTSBatch(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext traceCtx("tracer");
  auto &platform = get_platform();
  platform.with_execution_context(
      traceCtx, [&]() { kernel(std::forward<Args>(args)...); });
  cleanupTracerQubits(traceCtx.kernelTrace);

  auto kernelName = cudaq::getKernelName(kernel);
  validatePTSBEKernel(kernelName, traceCtx);

  static const cudaq::noise_model kEmptyNoiseModel;
  PTSBatch batch;
  batch.trace = buildPTSBETrace(traceCtx.kernelTrace, kEmptyNoiseModel);
  batch.measureQubits = extractMeasureQubits(batch.trace);
  return batch;
}

/// @brief Return type for asynchronous PTSBE sampling
using async_sample_result = std::future<sample_result>;

/// @brief Run PTSBE sampling with asynchronous dispatch
///
/// Uses the get_state_async pattern: enqueues a void QuantumTask on the
/// platform's QPU execution queue with a self-managed promise/future pair.
/// This preserves the full ptsbe::sample_result type (including execution
/// data) without slicing through KernelExecutionTask.
///
/// @tparam KernelFunctor Wrapped kernel functor type
/// @param wrappedKernel Functor that invokes the quantum kernel
/// @param platform Reference to the quantum platform
/// @param kernelName Name of the kernel (for diagnostics and MCM detection)
/// @param shots Number of shots for trajectory allocation
/// @param options PTSBE configuration options
/// @param qpu_id The QPU ID to execute on
/// @return future resolving to ptsbe::sample_result
/// @throws std::runtime_error if platform is remote (PTSBE is local-only)
template <typename KernelFunctor>
async_sample_result
runSamplingAsyncPTSBE(KernelFunctor &&wrappedKernel, quantum_platform &platform,
                      const std::string &kernelName, std::size_t shots,
                      const PTSBEOptions &options = PTSBEOptions{},
                      std::size_t qpu_id = 0) {
  // Validate upfront so exceptions are thrown in calling thread
  validatePTSBEPreconditions(platform, qpu_id);

  std::promise<sample_result> promise;
  auto future = promise.get_future();

  QuantumTask task = detail::make_copyable_function(
      [p = std::move(promise), shots, kernelName, &platform, options,
       kernel = std::forward<KernelFunctor>(wrappedKernel)]() mutable {
        p.set_value(
            runSamplingPTSBE(kernel, platform, kernelName, shots, options));
      });

  platform.enqueueAsyncTask(qpu_id, task);
  return future;
}

/// @brief Sample options for PTSBE execution
///
/// @param shots Number of shots to run for the given kernel
/// @param noise Noise model (required for PTSBE)
/// @param ptsbe PTSBE-specific configuration (execution data, strategy, etc.)
struct sample_options {
  std::size_t shots = 1000;
  cudaq::noise_model noise;
  PTSBEOptions ptsbe;
};

/// @brief Sample the given quantum kernel with PTSBE using a noise model
///
/// @param noise The noise model (required for PTSBE)
/// @param shots The number of shots to collect
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return ptsbe::sample_result with optional execution data
template <typename QuantumKernel, typename... Args>
sample_result sample(const cudaq::noise_model &noise, std::size_t shots,
                     QuantumKernel &&kernel, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&noise);

  sample_result result =
      runSamplingPTSBE([&]() mutable { kernel(std::forward<Args>(args)...); },
                       platform, kernelName, shots);

  platform.reset_noise();
  return result;
}

/// @brief Sample the given quantum kernel with PTSBE using sample_options
///
/// @param options PTSBE sample options (shots, noise, PTSBE configuration)
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return ptsbe::sample_result with measurement counts and optional execution
///         data
template <typename QuantumKernel, typename... Args>
sample_result sample(const sample_options &options, QuantumKernel &&kernel,
                     Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&options.noise);

  sample_result result =
      runSamplingPTSBE([&]() mutable { kernel(std::forward<Args>(args)...); },
                       platform, kernelName, options.shots, options.ptsbe);

  platform.reset_noise();
  return result;
}

/// @brief Asynchronously sample with PTSBE using a noise model
///
/// @param noise The noise model (required for PTSBE)
/// @param shots The number of shots to collect
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return future resolving to ptsbe::sample_result
template <typename QuantumKernel, typename... Args>
async_sample_result sample_async(const cudaq::noise_model &noise,
                                 std::size_t shots, QuantumKernel &&kernel,
                                 Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&noise);

  return runSamplingAsyncPTSBE(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, shots);
}

/// @brief Asynchronously sample with PTSBE using sample_options
///
/// @param options PTSBE sample options (shots, noise, PTSBE configuration)
/// @param kernel The kernel expression, must contain final measurements
/// @param args The variadic concrete arguments for evaluation of the kernel
/// @return future resolving to ptsbe::sample_result
template <typename QuantumKernel, typename... Args>
async_sample_result sample_async(const sample_options &options,
                                 QuantumKernel &&kernel, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  platform.set_noise(&options.noise);

  return runSamplingAsyncPTSBE(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, options.shots, options.ptsbe);
}

/// @brief Sample with PTSBE over a set of argument packs (broadcast)
///
/// For each element in the ArgumentSet, runs ptsbe::sample() and collects
/// the results. PTSBE is simulator-only so no multi-QPU distribution is used.
///
/// @param options PTSBE sample options (shots, noise, PTSBE configuration)
/// @param kernel The kernel expression, must contain final measurements
/// @param params ArgumentSet with one vector per kernel parameter
/// @return Vector of ptsbe::sample_result, one per argument set element
template <typename QuantumKernel, typename... Args>
std::vector<sample_result> sample(const sample_options &options,
                                  QuantumKernel &&kernel,
                                  ArgumentSet<Args...> &params) {
  auto N = std::get<0>(params).size();
  std::vector<sample_result> results;
  results.reserve(N);

  for (std::size_t i = 0; i < N; i++) {
    auto result = std::apply(
        [&](auto &...vecs) { return sample(options, kernel, vecs[i]...); },
        params);
    results.push_back(std::move(result));
  }
  return results;
}

} // namespace cudaq::ptsbe
