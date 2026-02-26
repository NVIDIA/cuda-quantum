/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include "PTSBEExecutionData.h"
#include <concepts>
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

  /// @brief Calculate total shots across all trajectories
  std::size_t totalShots() const;
};

/// @brief Concept for simulators supporting a customized PTSBE implementation
///
/// Enables compile-time detection of simulator PTSBE support with zero runtime
/// overhead. Simulators opting into PTSBE should implement sampleWithPTSBE
/// returning per-trajectory results. We use a concept to avoid exposing
/// the simulator base class to PTSBE.
template <typename SimulatorType>
concept PTSBECapable = requires(SimulatorType &sim, const PTSBatch &batch) {
  {
    sim.sampleWithPTSBE(batch)
  } -> std::same_as<std::vector<cudaq::sample_result>>;
};

/// @brief Aggregate per-trajectory sample results into a single result
cudaq::sample_result
aggregateResults(const std::vector<cudaq::sample_result> &results);

/// @brief Execute PTSBE batch on current simulator
///
/// Handles both runtime precision dispatch and compile-time concept dispatch:
/// 1. Uses isSinglePrecision() to determine float vs double
/// 2. Checks PTSBECapable concept for custom simulator implementations
/// 3. Falls back to samplePTSBEGeneric if no custom implementation
///
/// Caller must have set up ExecutionContext and allocated qubits
/// on the simulator before calling this function.
///
/// @param batch PTSBatch with trace, trajectories, and measureQubits
/// @return Per-trajectory sample results
/// @throws std::runtime_error if simulator cast fails or contract violated
std::vector<cudaq::sample_result> samplePTSBE(const PTSBatch &batch);

/// @brief Execute PTSBE with full life-cycle management (registry-based)
///
/// Convenience function that handles the complete simulator life-cycle:
/// 1. Gets current simulator from registry
/// 2. Creates ExecutionContext with specified type
/// 3. Sets context on simulator and allocates qubits
/// 4. Calls samplePTSBE for precision dispatch and trajectory execution
/// 5. Deallocates qubits and resets context
///
/// @param batch PTSBE specification
/// @param contextType ExecutionContext type (default: "ptsbe-sample").
/// @return Per-trajectory sample results
/// @throws std::runtime_error if simulator cast fails or gate conversion fails
std::vector<cudaq::sample_result>
samplePTSBEWithLifecycle(const PTSBatch &batch,
                         const std::string &contextType = "ptsbe-sample");

} // namespace cudaq::ptsbe
