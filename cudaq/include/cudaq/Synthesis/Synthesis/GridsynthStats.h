/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <atomic>
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

namespace details {

/// A field written only by the searching thread and readable from any other
/// while the search runs.
///
/// One thread writes, so relaxed ordering suffices and increment lowers to a
/// plain load, add and store (no lock). What it buys is definedness. Reading
/// a plain scalar while another thread writes it is a data race, whatever
/// word-sized loads happen to do on a given target.
///
/// Copyable so `GridsynthStats` still is, a copy snapshots the current value.
template <typename T>
class Relaxed {
public:
  Relaxed() = default;
  /* implicit */ Relaxed(T initial) : value_(initial) {}
  Relaxed(const Relaxed &other) : value_(other.load()) {}
  Relaxed &operator=(const Relaxed &other) { return *this = other.load(); }
  Relaxed &operator=(T next) {
    value_.store(next, std::memory_order_relaxed);
    return *this;
  }

  T load() const { return value_.load(std::memory_order_relaxed); }
  /* implicit */ operator T() const { return load(); }

  Relaxed &operator++() { return *this = load() + 1; }
  T operator++(int) {
    T previous = load();
    *this = previous + 1;
    return previous;
  }
  Relaxed &operator+=(T addend) { return *this = load() + addend; }

private:
  std::atomic<T> value_{};
};

} // namespace details

/// Counters describing the work one `gridsynth` call performed.
///
/// Passed by non-owning pointer and left untouched when null, so an
/// `uninstrumented` call pays nothing.
///
/// Counters are updated as the search runs, so a caller watching from another
/// thread can read progress out of a call that has not returned -- including
/// one it is about to abandon. That is what every field being a
/// `details::Relaxed` is for. Fields are independent, so a reader mid-search
/// can see one updated and another not. Only a reader that waits for the call
/// to return sees a consistent set. `outcome` is written once, at the end.
struct GridsynthStats {
  /// How the call ended. Meaningful whether or not the call succeeded.
  details::Relaxed<GridsynthOutcome> outcome = GridsynthOutcome::Success;

  /// Denominator exponent reached, and the bound allowed. `k_reached == k_max`
  /// on a failure means the search ran out of room rather than stalling on a
  /// single candidate.
  details::Relaxed<int64_t> k_reached = 0;
  details::Relaxed<int64_t> k_max = 0;

  /// Grid candidates the TDGP stepper produced, summed over every k.
  details::Relaxed<int64_t> candidates_enumerated = 0;

  /// Candidates dropped by the cheap residue gate before any factoring. The
  /// remainder is what the Diophantine solver was actually asked about.
  details::Relaxed<int64_t> candidates_residue_rejected = 0;

  /// Diophantine solves attempted, and how many returned a solution. Many
  /// calls with few successes means the cost is candidate throughput, not
  /// per-candidate effort.
  details::Relaxed<int64_t> diophantine_calls = 0;
  details::Relaxed<int64_t> diophantine_successes = 0;

  /// Integer-factoring attempts, and how many returned a factor.
  /// `factoring_restarts` counts re-rolls on a composite a previous attempt
  /// failed to split.
  details::Relaxed<int64_t> factoring_calls = 0;
  details::Relaxed<int64_t> factoring_successes = 0;
  details::Relaxed<int64_t> factoring_restarts = 0;

  /// Solves abandoned because a work budget ran out -- the restart limit or
  /// the per-candidate iteration budget -- rather than because the equation
  /// has no solution. Without this the two are indistinguishable, and they say
  /// opposite things about a budget: one that never exhausts is not buying
  /// anything, one that exhausts constantly is costing T gates.
  details::Relaxed<int64_t> candidates_budget_exhausted = 0;

  /// Attempts `GridsynthOptions::timeout` ended before their iteration budget
  /// ran out, and the same one level up. Both must be zero for a run to
  /// reproduce across machines, which they are unless a timeout was set.
  details::Relaxed<int64_t> factoring_wall_clock_exits = 0;
  details::Relaxed<int64_t> diophantine_wall_clock_exits = 0;

  /// Pollard-rho iterations summed over every attempt: the machine-independent
  /// measure of factoring effort, and what `maxFactoringIterations` bounds.
  details::Relaxed<int64_t> factoring_iterations_total = 0;

  /// MPFR working precision the call ran at. Enumeration cost scales with it,
  /// so it is the denominator for comparing per-k cost across tolerances.
  details::Relaxed<int64_t> working_precision_bits = 0;

  /// Nanoseconds spent enumerating candidates and solving their equations.
  /// The budgets in `GridsynthOptions` only move the second.
  details::Relaxed<int64_t> enumeration_ns = 0;
  details::Relaxed<int64_t> diophantine_ns = 0;
};

} // namespace cudaq::synth
