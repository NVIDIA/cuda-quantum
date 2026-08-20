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

/// The same, summed over one candidate's attempts. Capping only the
/// per-attempt budget moves the unbounded loop up a level rather than removing
/// it: a composite that never splits just starts attempt after attempt.
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
struct GridsynthOptions {
  /// Seed for the Pollard-rho RNG. Unset draws from `std::random_device`, so
  /// repeated calls explore different factoring attempts and their runtimes
  /// can differ by orders of magnitude.
  std::optional<uint64_t> seed = std::nullopt;

  /// Rho iterations one factoring attempt may spend. Lower gives up on hard
  /// composites sooner: cheaper everywhere, at the cost of T gates on the
  /// candidates it abandons.
  uint64_t maxFactoringIterations = details::DEFAULT_MAX_FACTORING_ITERATIONS;

  /// Rho iterations one candidate may spend across its attempts.
  uint64_t maxCandidateIterations = details::DEFAULT_MAX_CANDIDATE_ITERATIONS;

  /// Consecutive failed factoring attempts allowed on one composite.
  uint32_t maxFactoringRestarts = details::DEFAULT_MAX_FACTORING_RESTARTS;

  /// Steps one ODGP line scan may take without yielding. Bounds candidate
  /// enumeration, a separate cost from the factoring budgets above.
  uint64_t maxOdgpScanSteps = details::DEFAULT_MAX_ODGP_SCAN_STEPS;

  /// Optional wall-clock limit on the whole call. Unset -- the reproducible
  /// configuration -- leaves the budgets above as the only limits; a run that
  /// hits this reports it in `GridsynthStats`.
  std::optional<std::chrono::milliseconds> timeout = std::nullopt;
};

} // namespace cudaq::synth
