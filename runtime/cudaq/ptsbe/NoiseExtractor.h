/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSBEExecutionData.h"
#include "PTSSamplingStrategy.h"
#include "common/NoiseModel.h"
#include "common/Trace.h"
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Build the PTSBE instruction sequence from a raw cudaq::Trace.
///
/// Converts QuditInfo targets/controls to plain qubit indices. All instruction
/// types are preserved. Gate and measurement entries pass through. Noise
/// entries (from apply_noise) have their channels resolved via the noise model.
/// The resulting vector defines the unified index space for circuit_location
/// referenced byNoisePoint.
///
/// @param trace Raw circuit trace (may contain Gate, Noise, and Measurement)
/// @param noise_model Noise model used to resolve inline apply_noise channels
/// @return PTSBETrace with resolved channels for Noise entries
[[nodiscard]] PTSBETrace buildPTSBETrace(const cudaq::Trace &trace,
                                         const cudaq::noise_model &noise_model);

/// @brief Result of noise site extraction from a circuit
struct NoiseExtractionResult {
  /// @brief Ordered list of noise sites extracted from the circuit
  /// Each entry corresponds to a gate location where noise can be applied.
  /// Order matches circuit execution order (instruction sequence).
  std::vector<NoisePoint> noise_sites;

  /// @brief Total number of instructions in the circuit
  std::size_t total_instructions;

  /// @brief Number of instructions with noise applied
  std::size_t noisy_instructions;

  /// @brief Whether all extracted channels are unitary mixtures
  bool all_unitary_mixtures;
};

/// @brief Extract noise sites from a PTSBE trace.
///
/// Scans the PTSBE trace (produced by buildPTSBETrace) for Noise-type entries
/// and creates NoisePoints from them. All channels are already resolved in the
/// trace. Validates unitary mixture properties and collects
/// noise sites with their trace positions as circuit_location.
///
/// @param ptsbeTrace PTSBE trace from buildPTSBETrace
///                                 unitary mixture (default: true). PTSBE
///                                 requires all channels to be unitary
///                                 mixtures.
/// @return NoiseExtractionResult containing ordered noise sites and statistics
/// @throws std::invalid_argument if a channel cannot be converted to a unitary
///         mixture
[[nodiscard]] NoiseExtractionResult
extractNoiseSites(std::span<const TraceInstruction> ptsbeTrace,
                  bool validate_unitary_mixture = true);

} // namespace cudaq::ptsbe
