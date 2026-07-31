/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Grid synthesis: optimal Clifford+T approximation of R_z(theta)
//===----------------------------------------------------------------------===//
//
// Reference: Ross & Selinger, arXiv:1403.2975, Algorithm 7.6.

namespace details {

/// Per-candidate budget for the Diophantine solver. Larger values let the
/// algorithm push through harder `factorizations` and find smaller-T-count
/// solutions; the trade-off is worst-case latency per candidate. On timeout
/// the candidate is dropped and the search moves on.
inline constexpr int32_t DEFAULT_DIOPHANTINE_TIMEOUT_MS = 200;

/// Per-attempt budget for Pollard-rho integer factoring inside the
/// Diophantine solver. Hard composites that exceed this budget cause the
/// enclosing candidate to be skipped.
inline constexpr int32_t DEFAULT_FACTORING_TIMEOUT_MS = 50;

/// Largest denominator exponent k that `gridsynth_unitary` will scan before
/// giving up, as a function of the requested precision.
///
/// Ross & Selinger bound the T-count by 4*log2(1/epsilon) + K (Theorem 8.5),
/// and Lemma 7.3 ties the T-count to the denominator exponent as 2k-2 or 2k,
/// so a solution is expected by k ~ 2*log2(1/epsilon). Scanning well past that
/// means the search is not converging -- the grid only gets denser as k grows,
/// so the cause is the Diophantine solver starving on its timeouts, not a
/// shortage of candidates. Bounding k turns that into a reported failure the
/// caller can retry with larger budgets, instead of an unbounded loop.
///
/// Exposed for testing; `epsilon` must be finite and strictly positive.
int64_t max_denominator_exponent(const Real &epsilon);

/// Working precision, in bits, that the MPFR-backed reals need for a synthesis
/// run targeting `epsilon`.
///
/// Representing a target of accuracy epsilon takes about log2(1/epsilon)
/// significant bits. The 4x factor supplies guard bits so that rounding in
/// `gridsynth's` iterative arithmetic (candidate enumeration, Diophantine
/// solving) stays well below the epsilon budget, and the +64 floor keeps a
/// sane minimum for loose epsilon. This is an empirical heuristic.
///
/// `epsilon` must be finite and strictly positive.
mpfr_prec_t required_precision(const Real &epsilon);

} // namespace details

/// Core grid-synthesis search.
///
/// Runs Algorithm 7.6 until it produces a DOmegaUnitary approximating
/// R_z(theta) to within `epsilon` in the operator norm. Returns failure() if
/// `theta` is not finite, if `epsilon` is not finite and strictly positive,
/// if the epsilon region is degenerate, or if the search exhausts its budgets
/// without finding a valid solution.
///
/// @param theta                  Target rotation angle.
/// @param epsilon                Approximation precision, must be finite > 0.
/// @param diophantine_timeout_ms Per-candidate Diophantine budget.
/// @param factoring_timeout_ms   Per-attempt integer-factoring budget.
llvm::FailureOr<DOmegaUnitary> gridsynth_unitary(
    const Real &theta, const Real &epsilon,
    int32_t diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
    int32_t factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

/// End-to-end `gridsynth` entry point: search for a DOmegaUnitary via
/// `gridsynth_unitary`, then realize it as an explicit Clifford+T circuit
/// in Matsumoto-`Amano` normal form with minimum T-count via
/// `kmm_synthesize`.
llvm::FailureOr<Circuit> gridsynth(
    const Real &theta, const Real &epsilon,
    int32_t diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
    int32_t factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

} // namespace cudaq::synth
