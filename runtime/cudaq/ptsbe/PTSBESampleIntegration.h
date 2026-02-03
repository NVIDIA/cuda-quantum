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

#include "PTSBESampler.h"
#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include "cudaq/simulators.h"
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
void validatePTSBEEligibility(const std::string &kernelName,
                              const ExecutionContext &ctx);

/// @brief Check if context indicates mid-circuit measurements (legacy API)
bool hasMidCircuitMeasurements(const ExecutionContext &ctx);

/// @brief Throw if mid-circuit measurements detected (legacy API)
void throwIfMidCircuitMeasurements(const ExecutionContext &ctx);

/// @brief Extract measurement qubits from kernel trace
///
/// For PTSBE POC, assumes terminal measurements on all qubits in the circuit.
/// Future work may extract explicit measurement operations from the trace.
///
/// @param trace Captured kernel trace
/// @return Vector of qubit indices [0, 1, ..., numQubits-1]
std::vector<std::size_t> extractMeasureQubits(const Trace &trace);

/// @brief Run PTSBE sampling (internal API matching runSampling pattern)
///
/// Internal function called from cudaq::sample() when use_ptsbe=true.
/// Matches the signature pattern of details::runSampling for consistency.
///
/// @tparam KernelFunctor Wrapped kernel functor type
/// @param wrappedKernel Functor that invokes the quantum kernel
/// @param platform Reference to the quantum platform
/// @param kernelName Name of the kernel (for diagnostics and MCM detection)
/// @param shots Number of shots for trajectory allocation
/// @return Aggregated sample_result from all trajectories
/// @throws std::runtime_error if dynamic circuit detected or not implemented
template <typename KernelFunctor>
sample_result
runSamplingPTSBE(KernelFunctor &&wrappedKernel, quantum_platform &platform,
                 const std::string &kernelName, std::size_t shots) {
  // Stage 0: Capture trace via ExecutionContext("tracer")
  ExecutionContext traceCtx("tracer");
  platform.set_exec_ctx(&traceCtx);
  wrappedKernel();
  platform.reset_exec_ctx();

  // Stage 1: Validate kernel eligibility (no dynamic circuits)
  validatePTSBEEligibility(kernelName, traceCtx);

  // Stage 2: Construct PTSBatch from trace
  PTSBatch batch;
  batch.kernelTrace = std::move(traceCtx.kernelTrace);
  batch.measureQubits = extractMeasureQubits(batch.kernelTrace);

  // Stage 3: Execute PTSBE with lifecycle management
  auto results = samplePTSBEWithLifecycle(batch, "sample");

  return aggregateResults(results);
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
  platform.set_exec_ctx(&traceCtx);
  kernel(std::forward<Args>(args)...);
  platform.reset_exec_ctx();

  throwIfMidCircuitMeasurements(traceCtx);

  PTSBatch batch;
  batch.kernelTrace = std::move(traceCtx.kernelTrace);
  batch.measureQubits = extractMeasureQubits(batch.kernelTrace);
  return batch;
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
