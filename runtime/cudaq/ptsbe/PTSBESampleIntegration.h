/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file PTSBESampleIntegration.h
/// @brief PTSBE sample integration for CORE cudaq::sample() flow
///
/// Provides PTSBE (Pre-Trajectory Sampling with Batch Execution)
/// integration for the CORE sample API.
///
/// Usage:
///   auto result = cudaq::ptsbe::sampleWithPTSBE(kernel, shots, args...);
///
/// Limitations:
/// - PTSBE does not support mid-circuit measurements (MCM)
/// - Dynamic circuits with conditional logic are rejected
/// - Supports only unitary mixture noise
///

#pragma once

#include "PTSBEOptions.h"
#include "PTSBESampler.h"
#include "PTSBETrace.h"
#include "ShotAllocationStrategy.h"
#include "common/ExecutionContext.h"
#include "common/Future.h"
#include "common/NoiseModel.h"
#include "cudaq/platform.h"
#include <optional>
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

/// @brief Check if context indicates mid-circuit measurements (legacy API)
bool hasMidCircuitMeasurements(const ExecutionContext &ctx);

/// @brief Throw if mid-circuit measurements detected (legacy API)
void throwIfMidCircuitMeasurements(const ExecutionContext &ctx);

/// @brief Validate platform preconditions for PTSBE execution
void validatePTSBEPreconditions(
    quantum_platform &platform,
    std::optional<std::size_t> qpu_id = std::nullopt);

/// @brief Extract measurement qubits from kernel trace
///
/// For PTSBE POC, assumes terminal measurements on all qubits in the circuit.
/// Future work may extract explicit measurement operations from the trace.
///
/// @param trace Captured kernel trace
/// @return Vector of qubit indices [0, 1, ..., numQubits-1]
std::vector<std::size_t> extractMeasureQubits(const Trace &trace);

/// @brief Build PTSBETrace with interleaved instructions (no trajectories)
///
/// Converts the internal kernel trace into the user-facing PTSBETrace format.
/// For each gate in the kernel trace, a Gate instruction is added. If the noise
/// model defines noise at that gate, a Noise instruction follows. Measurement
/// instructions are appended for all measured qubits. The trajectories vector
/// is left empty.
///
/// @param kernelTrace Captured kernel trace
/// @param noiseModel Noise model for identifying noise sites
/// @return PTSBETrace with interleaved instructions and empty trajectories
PTSBETrace buildPTSBETraceInstructions(const cudaq::Trace &kernelTrace,
                                       const noise_model &noiseModel);

/// @brief Populate trajectories on an existing PTSBETrace
///
/// Takes ownership of trajectories and results. Remaps each KrausSelection's
/// circuit_location from noise-site index to the corresponding Noise
/// instruction index in PTSBETrace.instructions (derived by scanning the
/// instruction list). Populates measurement_counts from per-trajectory
/// execution results.
///
/// @param trace PTSBETrace to populate (must have instructions already set)
/// @param trajectories Executed trajectories
/// @param perTrajectoryResults Per-trajectory sample results
void populatePTSBETraceTrajectories(
    PTSBETrace &trace, std::vector<cudaq::KrausTrajectory> trajectories,
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
/// Internal function called from cudaq::sample() when ptsbe_options is set.
/// Captures the kernel trace, converts it to a PTSBETrace early, generates
/// trajectories, executes them, and aggregates results. Optionally attaches
/// the PTSBETrace (with trajectories) to the result when trace_output is
/// enabled.
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
/// @return Aggregated sample_result from all trajectories
/// @throws std::runtime_error if platform is not a simulator, noise model is
///         missing, or dynamic circuit detected
template <typename KernelFunctor>
sample_result runSamplingPTSBE(KernelFunctor &&wrappedKernel,
                               quantum_platform &platform,
                               const std::string &kernelName, std::size_t shots,
                               const PTSBEOptions &options = PTSBEOptions{}) {
  validatePTSBEPreconditions(platform);

  // Get noise model from platform (validated non-null by preconditions)
  const auto &noiseModel = *platform.get_noise();

  // Stage 0: Capture trace via ExecutionContext("tracer")
  ExecutionContext traceCtx("tracer");
  platform.with_execution_context(traceCtx, [&]() { wrappedKernel(); });

  // Stage 1: Validate kernel eligibility (no dynamic circuits)
  validatePTSBEKernel(kernelName, traceCtx);

  // Stage 2: Convert kernel trace to PTSBETrace (instructions only)
  auto ptsbeTrace =
      buildPTSBETraceInstructions(traceCtx.kernelTrace, noiseModel);

  // Stage 3: Build PTSBatch with trajectory generation and shot allocation
  auto batch = buildPTSBatchWithTrajectories(std::move(traceCtx.kernelTrace),
                                             noiseModel, options, shots);

  // Stage 4: Execute PTSBE with life-cycle management
  auto perTrajectoryResults = samplePTSBEWithLifecycle(batch, "sample");

  // Stage 5: Aggregate per-trajectory results
  auto result = aggregateResults(perTrajectoryResults);

  // Stage 6: Attach trajectories to trace and set on result if requested
  if (options.trace_output) {
    populatePTSBETraceTrajectories(ptsbeTrace, std::move(batch.trajectories),
                                   std::move(perTrajectoryResults));
    result.set_ptsbe_trace(std::move(ptsbeTrace));
  }

  return result;
}

/// @brief Capture kernel trace and construct PTSBatch (for testing)
///
/// Helper function that captures trace and builds PTSBatch without dispatching.
/// Used by tests to verify trace capture and batch construction independently
/// of execution.
///
/// @tparam QuantumKernel Quantum kernel type
/// @tparam Args Kernel argument types
/// @param kernel Quantum kernel to trace
/// @param args Kernel arguments
/// @return PTSBatch with kernelTrace, empty trajectories, and measureQubits
/// @throws std::runtime_error if MCM detected
template <typename QuantumKernel, typename... Args>
PTSBatch capturePTSBatch(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext traceCtx("tracer");
  auto &platform = get_platform();
  platform.with_execution_context(
      traceCtx, [&]() { kernel(std::forward<Args>(args)...); });

  throwIfMidCircuitMeasurements(traceCtx);

  PTSBatch batch;
  batch.kernelTrace = std::move(traceCtx.kernelTrace);
  batch.measureQubits = extractMeasureQubits(batch.kernelTrace);
  return batch;
}

/// @brief Run PTSBE sampling with `async` dispatch
///
/// Internal function called from cudaq::sample_async() when ptsbe_options is
/// set. Creates a KernelExecutionTask that runs PTSBE and enqueues it for
/// async execution.
///
/// PTSBE does not support remote execution - this function will reject
/// remote platforms.
///
/// @tparam KernelFunctor Wrapped kernel functor type
/// @param wrappedKernel Functor that invokes the quantum kernel
/// @param platform Reference to the quantum platform
/// @param kernelName Name of the kernel (for diagnostics and MCM detection)
/// @param shots Number of shots for trajectory allocation
/// @param options PTSBE configuration options
/// @param qpu_id The QPU ID to execute on
/// @param shot_allocation Strategy for allocating shots across trajectories
/// @return async_result<sample_result> that resolves to the sampling result
/// @throws std::runtime_error if platform is remote (PTSBE is local-only)
template <typename KernelFunctor>
async_result<sample_result>
runSamplingAsyncPTSBE(KernelFunctor &&wrappedKernel, quantum_platform &platform,
                      const std::string &kernelName, std::size_t shots,
                      const PTSBEOptions &options = PTSBEOptions{},
                      std::size_t qpu_id = 0) {
  // Validate upfront so exceptions are thrown in calling thread
  validatePTSBEPreconditions(platform, qpu_id);

  // Create async task that runs PTSBE
  KernelExecutionTask task(
      [shots, kernelName, &platform, options,
       kernel = std::forward<KernelFunctor>(wrappedKernel)]() mutable {
        return runSamplingPTSBE(kernel, platform, kernelName, shots, options);
      });

  return async_result<sample_result>(
      details::future(platform.enqueueAsyncTask(qpu_id, task)));
}

/// @brief PTSBE sample implementation (convenience wrapper for testing)
///
/// Simplified interface for testing. Wraps the kernel and calls
/// runSamplingPTSBE.
///
/// @tparam QuantumKernel Quantum kernel type (lambda or functor)
/// @tparam Args Kernel argument types
/// @param kernel Quantum kernel to sample
/// @param shots Number of shots (passed to trajectory generation)
/// @param args Kernel arguments
/// @return Aggregated sample_result from all trajectories
/// @throws std::runtime_error if dynamic circuit detected or not implemented
template <typename QuantumKernel, typename... Args>
sample_result sampleWithPTSBE(QuantumKernel &&kernel, std::size_t shots,
                              Args &&...args) {
  auto &platform = get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  return runSamplingPTSBE(
      [&]() mutable { kernel(std::forward<Args>(args)...); }, platform,
      kernelName, shots);
}

} // namespace cudaq::ptsbe
