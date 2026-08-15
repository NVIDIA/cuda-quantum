/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Grid synthesis statistics
//===----------------------------------------------------------------------===//

/// Why a `gridsynth` call ended.
///
/// `gridsynth` otherwise reports only success or `failure()`, which collapses
/// "this tolerance is unreachable by construction" together with "the search
/// ran out of room" -- distinctions a caller needs in order to decide whether
/// retrying differently could possibly help.
enum class GridsynthOutcome : uint8_t {
  /// A unitary meeting the tolerance was found by the grid search.
  Success,

  /// The tolerance is loose enough that a Clifford (zero T gates) already
  /// meets it, so no search ran.
  ZeroTShortcut,

  /// theta was not finite, or epsilon was not finite and strictly positive.
  InvalidInput,

  /// The epsilon region degenerated -- epsilon is too large or too small for
  /// the region construction to produce a usable ellipse.
  DegenerateEpsilonRegion,

  /// Upright preprocessing of the region failed, so the search never started.
  /// Distinct from a degenerate region: the region was built, the grid
  /// operator derived from it was not usable.
  PreprocessingFailed,

  /// The search scanned every denominator exponent up to `k_max` without a
  /// solvable candidate. Retrying with a larger budget may help; a larger
  /// epsilon certainly will.
  KExhausted,
};

/// Counters describing the work one `gridsynth` call performed.
///
/// Passed in by non-owning pointer and left untouched when null, so an
/// uninstrumented call pays nothing. Plain (non-atomic) counters: one call is
/// single-threaded, and the per-thread RNG makes concurrent calls independent.
///
/// Counters are updated as the search runs rather than assembled at the end,
/// so a caller watching from another thread can read progress out of a call
/// that has not returned -- including one it is about to abandon for running
/// too long. Only `outcome` is written once, at the end; treat it as valid
/// only after the call returns.
///
/// These exist because wall-clock time cannot distinguish the two ways a run
/// gets expensive -- grinding one hard candidate versus enumerating many
/// candidates that never yield a solvable equation. Those want opposite fixes,
/// so a default chosen without separating them is a guess.
struct GridsynthStats {
  /// How the call ended. Meaningful whether or not the call succeeded.
  GridsynthOutcome outcome = GridsynthOutcome::Success;

  /// Denominator exponent the search reached, and the bound it was allowed.
  /// `k_reached == k_max` on a failure is the signature of a search that ran
  /// out of room rather than one that stalled on a single candidate.
  int64_t k_reached = 0;
  int64_t k_max = 0;

  /// Grid candidates the TDGP stepper produced, summed over every k.
  int64_t candidates_enumerated = 0;

  /// Candidates dropped by the cheap residue gate before any factoring. The
  /// remainder is what the Diophantine solver was actually asked about.
  int64_t candidates_residue_rejected = 0;

  /// Diophantine solves attempted, and how many returned a solution. A large
  /// call count with few successes means the cost is candidate throughput,
  /// not per-candidate effort.
  int64_t diophantine_calls = 0;
  int64_t diophantine_successes = 0;

  /// Integer-factoring attempts made by the Diophantine solver, and how many
  /// returned a factor. `factoring_restarts` counts re-rolls on a composite
  /// that a previous attempt failed to split (the work the per-composite
  /// restart cap bounds).
  int64_t factoring_calls = 0;
  int64_t factoring_successes = 0;
  int64_t factoring_restarts = 0;

  /// Pollard-rho iterations summed over every attempt. This is the
  /// machine-independent measure of factoring effort, unlike the wall-clock
  /// budget that currently ends an attempt.
  int64_t factoring_iterations_total = 0;

  /// MPFR working precision, in bits, the call ran at. Enumeration cost scales
  /// with this, so it is the denominator for any comparison of per-k cost
  /// across tolerances.
  int64_t working_precision_bits = 0;

  /// Wall-clock nanoseconds spent enumerating candidates and solving their
  /// Diophantine equations. The shipped budget controls only move the second.
  int64_t enumeration_ns = 0;
  int64_t diophantine_ns = 0;
};

} // namespace cudaq::synth
