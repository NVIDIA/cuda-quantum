/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file PTSBESample.h
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

#include "PTSBEInterface.h"
#include "common/ExecutionContext.h"
#include "cudaq/platform.h"
#include <stdexcept>

namespace cudaq {
// Forward declaration from cudaq.h
bool kernelHasConditionalFeedback(const std::string &kernelName);
} // namespace cudaq

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
///
inline bool hasConditionalFeedback(const std::string &kernelName,
                                   const ExecutionContext &ctx) {
  // Check MLIR-compiled kernel metadata first
  if (cudaq::kernelHasConditionalFeedback(kernelName))
    return true;

  // Fallback: check library mode detection via registerNames
  return !ctx.registerNames.empty();
}

/// @brief Validate kernel eligibility for PTSBE execution
///
/// Checks all constraints required for PTSBE trajectory-based simulation:
/// - No conditional feedback on measurement results (dynamic circuits) or mid-circuit measurements
///
/// @param kernelName Name of the kernel being validated
/// @param ctx ExecutionContext populated after kernel tracing
/// @throws std::runtime_error if kernel is not eligible for PTSBE
///
inline void validatePTSBEEligibility(const std::string &kernelName,
                                     const ExecutionContext &ctx) {
  if (hasConditionalFeedback(kernelName, ctx)) {
    throw std::runtime_error(
        "PTSBE does not support dynamic circuits. "
        "Circuits with conditional logic based on measurement outcomes "
        "cannot currently be pre-trajectory sampled. The gate sequence must be "
        "deterministic for trajectory generation.");
  }
}

// Legacy API for backward compatibility with existing tests
inline bool hasMidCircuitMeasurements(const ExecutionContext &ctx) {
  return !ctx.registerNames.empty();
}

inline void throwIfMidCircuitMeasurements(const ExecutionContext &ctx) {
  if (hasMidCircuitMeasurements(ctx)) {
    throw std::runtime_error(
        "PTSBE does not support mid-circuit measurements. "
        "Circuits with conditional logic based on measurement outcomes "
        "cannot be pre-trajectory sampled.");
  }
}

/// @brief Extract measurement qubits from kernel trace
///
/// For PTSBE POC, assumes terminal measurements on all qubits in the circuit.
/// Future work may extract explicit measurement operations from the trace.
///
/// @param trace Captured kernel trace
/// @return Vector of qubit indices [0, 1, ..., numQudits-1]
inline std::vector<std::size_t> extractMeasureQubits(const Trace &trace) {
  std::vector<std::size_t> qubits;
  auto numQudits = trace.getNumQudits();
  qubits.reserve(numQudits);
  for (std::size_t i = 0; i < numQudits; ++i) {
    qubits.push_back(i);
  }
  return qubits;
}

/// @brief Dispatch PTSBatch to simulator for execution
///
/// Entry point for PTSBE execution after batch construction.
/// Currently a stub that throws "not implemented" as full trajectory
/// generation and simulator dispatch is future work.
///
/// @param batch PTSBatch with kernel_trace and measure_qubits
/// @return Aggregated sample_result (future implementation)
/// @throws std::runtime_error Always, until full implementation
///
/// Future implementation will:
/// 1. Generate trajectories via PreTrajectorySamplingEngine
/// 2. Call executePTSBE with simulator and batch
/// 3. Return aggregated results
inline sample_result dispatchPTSBE(const PTSBatch &batch) {
  // Count instructions for diagnostic output
  std::size_t instruction_count = 0;
  for (const auto &inst : batch.kernel_trace) {
    (void)inst;
    ++instruction_count;
  }

  throw std::runtime_error(
      "PTSBE dispatch successful but execution not implemented. "
      "Captured: " +
      std::to_string(instruction_count) + " instructions, " +
      std::to_string(batch.measure_qubits.size()) + " measure qubits, " +
      std::to_string(batch.trajectories.size()) + " trajectories. "
      "Full trajectory generation requires future implementation.");
}

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
sample_result runSamplingPTSBE(KernelFunctor &&wrappedKernel,
                                quantum_platform &platform,
                                const std::string &kernelName,
                                std::size_t shots) {
  // Stage 0: Capture trace via ExecutionContext("tracer")
  ExecutionContext trace_ctx("tracer");
  platform.set_exec_ctx(&trace_ctx);
  wrappedKernel();
  platform.reset_exec_ctx();

  // Stage 1: Validate kernel eligibility (no dynamic circuits)
  validatePTSBEEligibility(kernelName, trace_ctx);

  // Stage 2: Construct PTSBatch from trace
  PTSBatch batch;
  batch.kernel_trace = std::move(trace_ctx.kernelTrace);
  batch.measure_qubits = extractMeasureQubits(batch.kernel_trace);

  // Stage 3: Dispatch to executePTSBE
  return dispatchPTSBE(batch);
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
/// @return PTSBatch with kernel_trace, empty trajectories, and measure_qubits
/// @throws std::runtime_error if MCM detected
template <typename QuantumKernel, typename... Args>
PTSBatch capturePTSBatch(QuantumKernel &&kernel, Args &&...args) {
  ExecutionContext trace_ctx("tracer");
  auto &platform = get_platform();
  platform.set_exec_ctx(&trace_ctx);
  kernel(std::forward<Args>(args)...);
  platform.reset_exec_ctx();

  throwIfMidCircuitMeasurements(trace_ctx);

  PTSBatch batch;
  batch.kernel_trace = std::move(trace_ctx.kernelTrace);
  batch.measure_qubits = extractMeasureQubits(batch.kernel_trace);
  return batch;
}

/// @brief PTSBE sample implementation (convenience wrapper for testing)
///
/// Simplified interface for testing. Wraps the kernel and calls runSamplingPTSBE.
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
