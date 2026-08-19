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
/// ran out of room" -- a distinction the caller needs to decide whether
/// retrying differently could help.
enum class GridsynthOutcome : uint8_t {
  /// A unitary meeting the tolerance was found by the grid search.
  Success,

  /// A Clifford (zero T gates) already met the tolerance, so no search ran.
  ZeroTShortcut,

  /// theta was not finite, or epsilon was not finite and strictly positive.
  InvalidInput,

  /// epsilon is too large or too small for the region construction to produce
  /// a usable ellipse.
  DegenerateEpsilonRegion,

  /// Upright `preprocessing` failed, so the search never started. Unlike a
  /// degenerate region, the region was built; the grid operator was not usable.
  PreprocessingFailed,

  /// Every denominator exponent up to `k_max` was scanned without a solvable
  /// candidate. A larger budget may help; a larger epsilon certainly will.
  KExhausted,

  /// `GridsynthOptions::timeout` expired. Only reachable when the caller opted
  /// into a wall-clock limit, and the only outcome here that can differ between
  /// two machines running the same inputs.
  TimedOut,
};

/// Counters describing the work one `gridsynth` call performed.
///
/// Passed by non-owning pointer and left untouched when null, so an
/// `uninstrumented` call pays nothing. Plain (non-atomic) counters: one call is
/// single-threaded, and the per-thread RNG makes concurrent calls independent.
///
/// Counters are updated as the search runs, so a caller watching from another
/// thread can read progress out of a call that has not returned -- including
/// one it is about to abandon. Only `outcome` is written at the end; treat it
/// as valid only after the call returns.
///
/// These exist because wall-clock time cannot distinguish the two ways a run
/// gets expensive: grinding one hard candidate versus enumerating many that
/// never yield a solvable equation. Those want opposite fixes.
struct GridsynthStats {
  /// How the call ended. Meaningful whether or not the call succeeded.
  GridsynthOutcome outcome = GridsynthOutcome::Success;

  /// Denominator exponent reached, and the bound allowed. `k_reached == k_max`
  /// on a failure means the search ran out of room rather than stalling on a
  /// single candidate.
  int64_t k_reached = 0;
  int64_t k_max = 0;

  /// Grid candidates the TDGP stepper produced, summed over every k.
  int64_t candidates_enumerated = 0;

  /// Candidates dropped by the cheap residue gate before any factoring. The
  /// remainder is what the Diophantine solver was actually asked about.
  int64_t candidates_residue_rejected = 0;

  /// Diophantine solves attempted, and how many returned a solution. Many
  /// calls with few successes means the cost is candidate throughput, not
  /// per-candidate effort.
  int64_t diophantine_calls = 0;
  int64_t diophantine_successes = 0;

  /// Integer-factoring attempts, and how many returned a factor.
  /// `factoring_restarts` counts re-rolls on a composite a previous attempt
  /// failed to split.
  int64_t factoring_calls = 0;
  int64_t factoring_successes = 0;
  int64_t factoring_restarts = 0;

  /// Solves abandoned because a work budget ran out -- the restart limit or
  /// the per-candidate iteration budget -- rather than because the equation
  /// has no solution. Without this the two are indistinguishable, and they say
  /// opposite things about a budget: one that never exhausts is not buying
  /// anything, one that exhausts constantly is costing T gates.
  int64_t candidates_budget_exhausted = 0;

  /// Attempts `GridsynthOptions::timeout` ended before their iteration budget
  /// ran out, and the same one level up. Both must be zero for a run to
  /// reproduce across machines, which they are unless a timeout was set.
  int64_t factoring_wall_clock_exits = 0;
  int64_t diophantine_wall_clock_exits = 0;

  /// Pollard-rho iterations summed over every attempt: the machine-independent
  /// measure of factoring effort, and what `maxFactoringIterations` bounds.
  int64_t factoring_iterations_total = 0;

  /// MPFR working precision the call ran at. Enumeration cost scales with it,
  /// so it is the denominator for comparing per-k cost across tolerances.
  int64_t working_precision_bits = 0;

  /// Nanoseconds spent enumerating candidates and solving their equations.
  /// The budgets in `GridsynthOptions` only move the second.
  int64_t enumeration_ns = 0;
  int64_t diophantine_ns = 0;
};

} // namespace cudaq::synth
