/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Grid synthesis controls
//===----------------------------------------------------------------------===//

namespace details {

/// Pollard-rho iterations one factoring attempt may spend. The solver's own
/// bound, L = 1.1774 * 10^(digits/4), is unreachable at deep epsilon (1.2e10
/// for a 40-digit composite), so without this cap only a clock ends a hard
/// attempt and the result depends on host speed.
inline constexpr uint64_t DEFAULT_MAX_FACTORING_ITERATIONS = 500000;

/// The same, summed over every factoring attempt one grid candidate makes.
///
/// This is not implied by the per-attempt cap and the restart cap, because the
/// restart cap is per composite, not per candidate: the counter resets
/// whenever the top of the factor stack changes. A candidate that keeps making
/// partial progress (splitting off a factor, pushing the two `cofactors`,
/// failing on those) resets it every time and can spend without bound.
///
/// Measured over 38 tuning angles x 1e-30..1e-40. Lifting this cap costs 1.97x
/// total runtime and 2.2x p95, uniformly across tolerances, and raises
/// restarts 222 -> 560 while changing the summed T-count by +2 in 64830. The
/// implied bound (per-attempt x restarts = 4M) recovers almost none of that,
/// so the useful value is well below it.
inline constexpr uint64_t DEFAULT_MAX_CANDIDATE_ITERATIONS =
    4 * DEFAULT_MAX_FACTORING_ITERATIONS;

/// Consecutive failed attempts allowed on one composite. Each re-rolls a fresh
/// random `a`, so retrying is worth something, but with steep diminishing
/// returns: over 1191 restart chains, 98.8% factored on the first attempt and
/// none exceeded depth 3.
inline constexpr uint32_t DEFAULT_MAX_FACTORING_RESTARTS = 8;

/// Steps one ODGP line scan may take without yielding. The scan's a-range is
/// finite but can be astronomically wide (past 1e15 at k=100), harmless only
/// because a solution normally turns up in a few steps; on a line that has
/// none the scan never finishes. `gridsynth(pi/4, 1e-15)` hung on exactly that.
inline constexpr uint64_t DEFAULT_MAX_ODGP_SCAN_STEPS = 1 << 16;

} // namespace details

/// Controls on the work one `gridsynth` call may do.
///
/// These count work, not time, which is what makes them tunable: the same
/// inputs do the same work on every machine, so a default measured on one host
/// holds on another and a seeded run replays exactly. `timeout` is an escape
/// hatch, not a tuning knob -- setting it puts host speed back into the result.
///
/// What they all trade. The search (Ross & Selinger, arXiv:1403.2975,
/// Algorithm 7.6) walks the denominator exponent k = 0, 1, 2, ... upward. At
/// each k it enumerates grid candidates u, and for each candidate asks whether
/// the Diophantine equation conj(t) * t = 1 - conj(u) * u is solvable. The
/// first solvable candidate ends the search. The T-count of the circuit is
/// 2k - 2 or 2k (Lemma 7.3), so a solution found at a smaller k is a strictly
/// shorter circuit.
///
/// Every budget below therefore has the same shape. It bounds work spent
/// looking at one candidate, and spending less risks abandoning a candidate
/// that was solvable, pushing the answer out to a larger k. So the axis is
/// runtime against T-count, and the direction is always the same (a cheaper
/// setting is never shorter). Which stage each one bounds:
///
///   `maxOdgpScanSteps`       enumerating candidates u  (sec. 4-5)
///   `maxFactoringIterations` one factoring attempt     (sec. 6, App. C)
///   `maxCandidateIterations` all attempts for one u    (sec. 6, App. C)
///   `maxFactoringRestarts`   re-rolls on one composite (sec. 6, App. C)
///
/// The two factoring budgets are not interchangeable, though, and the
/// difference decides which one to reach for. Swept one at a time over 38
/// angles x 3 seeds:
///
/// `maxCandidateIterations` is a floor, not a dial. At 1e-40, 500k costs +12
/// T gates and 1M costs +10, the shipped 2M reaches the best T-count observed,
/// and 4M and 8M buy nothing further while runtime doubles (4.2s -> 8.3s over
/// the corpus). Set it too low and circuits get longer. Raising it past the
/// default only costs time.
///
/// `maxFactoringIterations` is the one that keeps paying, because it decides
/// when a candidate is abandoned, 14 gates for 1.5x runtime at 1e-35. Raise
/// that one if a caller wants shorter circuits. Note its effect is not
/// monotone at 1e-40 (a different budget abandons different candidates and
/// lands on a different k), so single-point comparisons there are unreliable.
struct GridsynthOptions {
  /// Seed for the random parameter Pollard-rho draws at the start of each
  /// factoring attempt (sec. 6, and Rabin's `primality` test draws from the
  /// same stream). Unset draws from `std::random_device`, so two calls on
  /// identical inputs try different attempts and their runtimes can differ by
  /// orders of magnitude. Set it when a run has to replay exactly.
  std::optional<uint64_t> seed = std::nullopt;

  /// Pollard-rho iterations one factoring attempt may spend before giving up
  /// on its composite.
  ///
  /// Solving conj(t) * t = xi for a candidate needs the prime factorization of
  /// xi's norm (Theorem 6.2, Proposition C.24), and factoring is where
  /// essentially all per-candidate time goes. Lower abandons hard composites
  /// sooner (cheaper on every call, at the cost of T gates on the candidates it
  /// gives up on).
  uint64_t maxFactoringIterations = details::DEFAULT_MAX_FACTORING_ITERATIONS;

  /// The same iterations, summed over every factoring attempt made for one
  /// grid candidate. Bounds the candidate as a whole, which neither the
  /// per-attempt cap nor `maxFactoringRestarts` does (see
  /// `DEFAULT_MAX_CANDIDATE_ITERATIONS` for why, and for the measurement).
  uint64_t maxCandidateIterations = details::DEFAULT_MAX_CANDIDATE_ITERATIONS;

  /// Consecutive failed attempts allowed on one composite before the candidate
  /// is abandoned. Each attempt re-rolls the random parameter, so a retry is a
  /// genuinely different search rather than a repeat (but with steep
  /// diminishing returns, hence a default far above the observed need).
  uint32_t maxFactoringRestarts = details::DEFAULT_MAX_FACTORING_RESTARTS;

  /// Steps one one-dimensional grid-problem scan may take without yielding a
  /// solution.
  ///
  /// This bounds the other stage. The two-dimensional problem of finding
  /// candidates u is solved as a family of one-dimensional problems along grid
  /// lines (Lemma 5.6, sec. 4), and a line carrying no solution can be scanned
  /// essentially forever. So this governs the supply of candidates rather
  /// than the effort spent on each, and trades against the factoring budgets
  /// rather than with them. A call starved here fails with `KExhausted` having
  /// solved nothing, instead of returning a longer circuit.
  uint64_t maxOdgpScanSteps = details::DEFAULT_MAX_ODGP_SCAN_STEPS;

  /// Optional wall-clock limit on the whole call. Unset -- the reproducible
  /// configuration -- leaves the budgets above as the only limits; a run that
  /// hits this reports it in `GridsynthStats`.
  std::optional<std::chrono::milliseconds> timeout = std::nullopt;
};

} // namespace cudaq::synth
