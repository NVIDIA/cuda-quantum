/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include "PTSBEExecutionData.h"
#include "PTSBESampleResult.h"
#include "common/ExecutionContext.h"
#include "cudaq/ptsbe/policy.h"
#include "cudaq/qis/execution_manager.h"
#include <cstddef>
#include <string>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Batch specification for PTSBE execution
struct PTSBatch {
  /// @brief PTSBE instruction sequence (Gate, Noise, Measurement interleaved
  /// with resolved channels). Built by buildPTSBETrace from the raw kernel
  /// trace and noise model. All downstream execution code works from this.
  PTSBETrace trace;

  /// @brief Sampled noise trajectories
  std::vector<cudaq::KrausTrajectory> trajectories;

  /// @brief Qubits to measure (terminal measurements)
  /// NOTE: This currently only applies to kernels that are terminal measurement
  /// only which is a limitation of the current PTSBE implementation.
  std::vector<std::size_t> measureQubits;

  /// @brief Populate per-shot sequential bitstring data on the result. When
  /// false (default), only aggregated counts are produced.
  bool includeSequentialData = false;

  /// @brief Calculate total shots across all trajectories
  std::size_t totalShots() const;
};

} // namespace cudaq::ptsbe

namespace cudaq::ptsbe::detail {

/// @brief Aggregate per-trajectory sample results into a single result
cudaq::sample_result
aggregateResults(const std::vector<cudaq::sample_result> &results);

/// @brief Execute PTSBE batch on current simulator
///
/// Handles runtime precision and custom simulator dispatch:
/// 1. Uses isSinglePrecision() to determine float vs double
/// 2. Checks BatchSimulator interface for custom simulator implementations
/// 3. Falls back to samplePTSBEGeneric if no custom implementation
///
/// Caller must have set up ExecutionContext and allocated qubits
/// on the simulator before calling this function.
///
/// @param batch PTSBatch with trace, trajectories, measureQubits, and
///        includeSequentialData flag
/// @return Per-trajectory sample results
/// @throws std::runtime_error if simulator cast fails or contract violated
std::vector<cudaq::sample_result> samplePTSBE(const PTSBatch &batch);

/// @brief Finalize a PTSBE execution for the given policy
///
/// @param policy Policy for the PTSBE execution
/// @return Aggregated sample result
/// @throws std::runtime_error if the policy has no batch and the active
///         context is not trace-capturing
ptsbe::sample_result finalizePTSBE(const cudaq::ptsbe::sample_policy &policy);

/// @brief Allocate the batch qubits on the circuit simulator
void allocateBatchQubits(std::size_t nQubits);

/// @brief Release the batch qubits from the circuit simulator
void releaseBatchQubits(std::size_t nQubits);

/// @brief Execute the policy's batch through the standard policy machinery
///
/// @param policy Policy carrying the PTSBatch (batch must be non-null)
/// @return Aggregated sample result. Per-trajectory results are stored on
///         the policy
inline ptsbe::sample_result
executeBatch(const cudaq::ptsbe::sample_policy &policy) {
  const auto nQubits = numQubits(policy.batch->trace);

  cudaq::ExecutionContext ctx(cudaq::ptsbe::sample_policy::name,
                              policy.batch->totalShots());
  ctx.kernelName = policy.kernelName;

  try {
    return cudaq::detail::with_policy_and_ctx(policy, ctx, [&] {
      return cudaq::ExecutionManager::with_default_em(
          policy, [&] { allocateBatchQubits(nQubits); });
    });
  } catch (...) {
    releaseBatchQubits(nQubits);
    throw;
  }
}

/// @brief Execute PTSBE with full life-cycle management (registry-based)
///
/// Convenience wrapper over executeBatch for callers that only need the
/// per-trajectory results.
///
/// @param batch PTSBE specification (includes includeSequentialData flag)
/// @return Per-trajectory sample results
/// @throws std::runtime_error if simulator cast fails or gate conversion fails
inline std::vector<cudaq::sample_result>
samplePTSBEWithLifecycle(const PTSBatch &batch) {
  cudaq::ptsbe::sample_policy policy;
  policy.batch = &batch;
  policy.shots = batch.totalShots();
  executeBatch(policy);
  return std::move(policy.perTrajectoryResults);
}

} // namespace cudaq::ptsbe::detail
