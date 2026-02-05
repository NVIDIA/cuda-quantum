/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSSamplingStrategy.h"
#include "common/NoiseModel.h"
#include "common/Trace.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::ptsbe {

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

/// @brief Extract noise sites from a circuit trace given a noise model
///
/// Iterates through the circuit instructions and queries the noise model for
/// applicable noise channels. For each noisy instruction, creates a NoisePoint
/// with the Kraus operators and probabilities.
///
/// @param trace Captured circuit trace containing instructions
/// @param noise_model Noise model defining error channels
/// @param validate_unitary_mixture If true, throws if any channel is not a
///                                 unitary mixture (default: true)
/// @param tolerance Numerical tolerance for validation (default: 1e-6)
/// @return NoiseExtractionResult containing ordered noise sites and statistics
/// @throws std::invalid_argument if validation fails and validate_unitary_mixture
///         is true
[[nodiscard]] NoiseExtractionResult
extractNoiseSites(const cudaq::Trace &trace,
                  const cudaq::noise_model &noise_model,
                  bool validate_unitary_mixture = true, double tolerance = 1e-6);

} // namespace cudaq::ptsbe
