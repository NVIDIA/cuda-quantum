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
#include "cudaq/Synthesis/Support/Result.h"

#include <cmath>

/// Grid synthesis algorithm for optimal Clifford+T approximation of R_z(θ).
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Algorithm 7.6.

namespace cudaq::synth {

namespace details {
// Default timeout for the Diophantine solver (step 2c of Algorithm 7.6).
// If the solver exceeds this time on a particular candidate u, the
// candidate is skipped and the next one is tried. Higher values improve
// optimality at the cost of worst-case latency.
inline constexpr i32 DEFAULT_DIOPHANTINE_TIMEOUT_MS = 200;

// Default timeout for integer factoring within the Diophantine solver
// (step 2b of Algorithm 7.6, using Pollard-Brent). Hard composites
// that exceed this timeout cause the candidate to be skipped.
inline constexpr i32 DEFAULT_FACTORING_TIMEOUT_MS = 50;
} // namespace details

/// Internal `gridsynth` algorithm — returns a DOmegaUnitary approximation.
///
/// This is the core no-exception implementation of Algorithm 7.6.
/// Returns failure() if the epsilon region is degenerate or the search
/// space is exhausted before finding a valid solution.
///
/// @param theta Target rotation angle R_z(θ)
/// @param epsilon Approximation precision ε
/// @param diophantine_timeout_ms Timeout for Diophantine solving (step 2c)
/// @param factoring_timeout_ms Timeout for integer factoring (step 2b)
/// @return FailureOr<DOmegaUnitary> — the synthesized unitary or failure()
///
FailureOr<DOmegaUnitary> gridsynth_unitary(
    const Real &theta, const Real &epsilon,
    i32 diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
    i32 factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

/// Main `gridsynth` algorithm — returns a Clifford+T Circuit.
///
/// Convenience wrapper: calls gridsynth_unitary() to find the DOmegaUnitary,
/// then kmm_synthesize() for exact synthesis. The result is in
/// `Matsumoto-Amano` normal form with minimum T-count.
///
/// Gate alphabet: T (π/8), H (Hadamard), S (phase), X (Pauli-X),
/// W (global phase ω = e^{iπ/4}).
///
/// @param theta Target rotation angle
/// @param epsilon Approximation precision
/// @param diophantine_timeout_ms Timeout for Diophantine solving
/// @param factoring_timeout_ms Timeout for integer factoring
/// @return Synthesized Circuit, or failure() if the ε-region is degenerate.
///
FailureOr<Circuit>
gridsynth(const Real &theta, const Real &epsilon,
          i32 diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
          i32 factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

} // namespace cudaq::synth
