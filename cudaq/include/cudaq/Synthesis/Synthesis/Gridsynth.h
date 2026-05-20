/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include "llvm/Support/LogicalResult.h"

#include <cmath>

/// Grid synthesis algorithm for optimal Clifford+T approximation of R_z(θ).
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Algorithm 7.6.

namespace cudaq::synth {

namespace details {
// Default timeout for the Diophantine solver.
// If the solver exceeds this time on a particular candidate, the candidate is
// skipped and the next one is tried. Higher values improve optimality at the
// cost of worst-case latency.
inline constexpr i32 DEFAULT_DIOPHANTINE_TIMEOUT_MS = 200;

// Default timeout for integer factoring within the Diophantine solver
// Hard composites that exceed this timeout cause the candidate to be skipped.
inline constexpr i32 DEFAULT_FACTORING_TIMEOUT_MS = 50;
} // namespace details

/// Internal `gridsynth` algorithm.
///
/// This is the core no-exception implementation of Algorithm 7.6.
/// Returns failure() if the epsilon region is degenerate or the search
/// space is exhausted before finding a valid solution.
///
/// @param theta Target rotation angle R_z(θ)
/// @param epsilon Approximation precision
/// @param diophantine_timeout_ms Timeout for Diophantine solving
/// @param factoring_timeout_ms Timeout for integer factoring
/// @return The unitary as a DOmegaUnitary, or failure()
///
llvm::FailureOr<DOmegaUnitary> gridsynth_unitary(
    const Real &theta, const Real &epsilon,
    i32 diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
    i32 factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

/// Main `gridsynth` algorithm.
///
/// Convenience wrapper: calls `gridsynth_unitary()` to find the
// `DOmegaUnitary`, then `kmm_synthesize()` for exact synthesis. The result
/// is in `Matsumoto-Amano` normal form with minimum T-count.
///
/// @param theta Target rotation angle
/// @param epsilon Approximation precision
/// @param diophantine_timeout_ms Timeout for Diophantine solving
/// @param factoring_timeout_ms Timeout for integer factoring
/// @return The synthesized circuit or failure()
///
llvm::FailureOr<Circuit>
gridsynth(const Real &theta, const Real &epsilon,
          i32 diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
          i32 factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

} // namespace cudaq::synth
