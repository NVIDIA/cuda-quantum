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

/// Pollard-rho iterations one factoring attempt may spend.
///
/// The solver's own bound, L = 1.1774 * 10^(digits/4), is unreachable at deep
/// epsilon -- 1.2e10 for a 40-digit composite -- so without this cap the only
/// thing that ends a hard attempt is a clock, and the result then depends on
/// how fast the host is. Measured over the full corpus, capping here is what
/// makes a run reproducible; see `maxCandidateIterations` for why the
/// per-candidate sum has to be bounded as well.
inline constexpr uint64_t DEFAULT_MAX_FACTORING_ITERATIONS = 500000;

/// Pollard-rho iterations one grid candidate may spend across all of the
/// factoring attempts its Diophantine solve makes.
///
/// Bounding only the per-attempt budget moves the unbounded loop up one level
/// rather than removing it: a candidate whose composite never splits just
/// starts attempt after attempt, each one inside its own budget.
inline constexpr uint64_t DEFAULT_MAX_CANDIDATE_ITERATIONS =
    4 * DEFAULT_MAX_FACTORING_ITERATIONS;

/// Consecutive failed factoring attempts allowed on one composite.
///
/// A retry re-rolls a fresh random `a`, so retrying is a real strategy rather
/// than a repeat of the same work -- but with diminishing returns. Measured
/// over 1191 restart chains, 98.8% factor on the first attempt and none
/// exceeded depth 3.
inline constexpr uint32_t DEFAULT_MAX_FACTORING_RESTARTS = 8;

} // namespace details

/// Controls on the work one `gridsynth` call may do.
///
/// The budgets below count work, not time, which is what makes them tunable:
/// the same inputs and the same options do the same work on every machine, so
/// a default chosen from measurements on one host holds on another and a run
/// can be replayed exactly. A wall-clock limit cannot do either -- it turns
/// host speed into an input, and the setting that is generous on a workstation
/// silently truncates the search on a loaded CI runner.
///
/// `timeout` is therefore an escape hatch rather than a tuning knob: it is
/// unset by default, and setting it reintroduces exactly the host dependence
/// the budgets remove. Prefer lowering the budgets.
struct GridsynthOptions {
  /// Seed for the internal Pollard-rho RNG. Unset draws from
  /// `std::random_device`, so repeated calls on the same input explore
  /// different factoring attempts and their runtimes can differ by orders of
  /// magnitude. Set it to make a run replayable.
  std::optional<uint64_t> seed = std::nullopt;

  /// Rho iterations one factoring attempt may spend. Lower gives up on hard
  /// composites sooner, which costs T gates on the affected candidates and
  /// saves time everywhere.
  uint64_t maxFactoringIterations = details::DEFAULT_MAX_FACTORING_ITERATIONS;

  /// Rho iterations one grid candidate may spend in total, summed over its
  /// factoring attempts. Below `maxFactoringIterations` it is the only budget
  /// that binds.
  uint64_t maxCandidateIterations = details::DEFAULT_MAX_CANDIDATE_ITERATIONS;

  /// Consecutive failed factoring attempts allowed on one composite, each
  /// re-rolling the rho parameters.
  uint32_t maxFactoringRestarts = details::DEFAULT_MAX_FACTORING_RESTARTS;

  /// Optional wall-clock limit on the whole call, checked between candidates
  /// and inside the factoring loop. Unset means the budgets above are the only
  /// limits, which is the reproducible configuration; a run that hits this
  /// limit is host-dependent and reports it in `GridsynthStats`.
  std::optional<std::chrono::milliseconds> timeout = std::nullopt;
};

} // namespace cudaq::synth
