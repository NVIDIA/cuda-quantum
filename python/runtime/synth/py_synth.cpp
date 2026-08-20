/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#include "llvm/Support/LogicalResult.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <variant>

namespace {

using RealArg = std::variant<double, std::string>;

/// Converts a Python float or str argument to a Real. `name` is the Python
/// parameter name, used to build the ValueError message for malformed
/// strings.
cudaq::synth::Real toReal(const RealArg &arg, const char *name) {
  if (std::holds_alternative<double>(arg))
    return cudaq::synth::Real(std::get<double>(arg));
  const std::string &str = std::get<std::string>(arg);
  std::optional<cudaq::synth::Real> parsed =
      cudaq::synth::Real::from_string(str);
  if (!parsed)
    throw nanobind::value_error(
        (std::string(name) + ": invalid numeric string '" + str + "'").c_str());
  return *std::move(parsed);
}

/// Shared body of the two gridsynth entry points. Writes the counters into
/// `stats` so the caller can decide whether to surface them.
std::string
gridsynthImpl(RealArg theta, RealArg epsilon, uint64_t max_factoring_iterations,
              uint64_t max_candidate_iterations,
              uint32_t max_factoring_restarts, uint64_t max_odgp_scan_steps,
              std::optional<uint64_t> seed, std::optional<int64_t> timeout_ms,
              cudaq::synth::GridsynthStats &stats) {
  // Parse epsilon once at the stock precision, which is ample to read off its
  // magnitude, and validate before deriving anything from it.
  cudaq::synth::Real epsilonProbe = toReal(epsilon, "epsilon");
  if (!epsilonProbe.is_finite() || !(epsilonProbe > 0))
    throw nanobind::value_error("epsilon must be finite and strictly positive");

  // Raise the working precision to match the requested epsilon, then re-parse
  // both inputs so they are materialized at that precision: a Real built
  // beforehand keeps the old precision and would cap the whole run.
  cudaq::synth::ScopedDefaultPrecision precision(
      cudaq::synth::details::required_precision(epsilonProbe));

  cudaq::synth::Real thetaReal = toReal(theta, "theta");
  cudaq::synth::Real epsilonReal = toReal(epsilon, "epsilon");

  if (!thetaReal.is_finite())
    throw nanobind::value_error("theta must be finite");

  cudaq::synth::GridsynthOptions options;
  options.seed = seed;
  options.maxFactoringIterations = max_factoring_iterations;
  options.maxCandidateIterations = max_candidate_iterations;
  options.maxFactoringRestarts = max_factoring_restarts;
  options.maxOdgpScanSteps = max_odgp_scan_steps;
  if (timeout_ms) {
    if (*timeout_ms <= 0)
      throw nanobind::value_error("timeout_ms must be positive");
    options.timeout = std::chrono::milliseconds(*timeout_ms);
  }

  llvm::FailureOr<cudaq::synth::Circuit> result = llvm::failure();
  {
    nanobind::gil_scoped_release nogil;
    result = cudaq::synth::gridsynth(thetaReal, epsilonReal, options, &stats);
  }
  if (llvm::failed(result)) {
    // A timeout is the caller's own limit, not a property of the inputs, so
    // "search exhausted" would send them looking in the wrong place.
    if (stats.outcome == cudaq::synth::GridsynthOutcome::TimedOut)
      throw nanobind::value_error(
          "gridsynth: timeout_ms expired before a Clifford+T approximation "
          "was found");
    throw nanobind::value_error(
        "gridsynth: failed to synthesize a Clifford+T approximation "
        "(degenerate epsilon region or search exhausted)");
  }
  return result->to_string();
}

std::string gridsynthBinding(RealArg theta, RealArg epsilon,
                             uint64_t max_factoring_iterations,
                             uint64_t max_candidate_iterations,
                             uint32_t max_factoring_restarts,
                             uint64_t max_odgp_scan_steps,
                             std::optional<uint64_t> seed,
                             std::optional<int64_t> timeout_ms) {
  cudaq::synth::GridsynthStats stats;
  return gridsynthImpl(theta, epsilon, max_factoring_iterations,
                       max_candidate_iterations, max_factoring_restarts,
                       max_odgp_scan_steps, seed, timeout_ms, stats);
}

/// Name of a GridsynthOutcome, for the stats dict below.
const char *outcomeName(cudaq::synth::GridsynthOutcome outcome) {
  switch (outcome) {
  case cudaq::synth::GridsynthOutcome::Success:
    return "success";
  case cudaq::synth::GridsynthOutcome::ZeroTShortcut:
    return "zero_t_shortcut";
  case cudaq::synth::GridsynthOutcome::InvalidInput:
    return "invalid_input";
  case cudaq::synth::GridsynthOutcome::DegenerateEpsilonRegion:
    return "degenerate_epsilon_region";
  case cudaq::synth::GridsynthOutcome::PreprocessingFailed:
    return "preprocessing_failed";
  case cudaq::synth::GridsynthOutcome::KExhausted:
    return "k_exhausted";
  case cudaq::synth::GridsynthOutcome::TimedOut:
    return "timed_out";
  }
  return "unknown";
}

/// gridsynth plus its counters, as (gates, stats dict).
///
/// Private on purpose: the shape of a public statistics API belongs to the
/// pending synth API restructure. This exists so budget tuning can be
/// measured rather than guessed at.
nanobind::tuple gridsynthWithStatsBinding(RealArg theta, RealArg epsilon,
                                          uint64_t max_factoring_iterations,
                                          uint64_t max_candidate_iterations,
                                          uint32_t max_factoring_restarts,
                                          uint64_t max_odgp_scan_steps,
                                          std::optional<uint64_t> seed,
                                          std::optional<int64_t> timeout_ms) {
  cudaq::synth::GridsynthStats stats;
  std::string gates = gridsynthImpl(
      theta, epsilon, max_factoring_iterations, max_candidate_iterations,
      max_factoring_restarts, max_odgp_scan_steps, seed, timeout_ms, stats);

  nanobind::dict out;
  out["outcome"] = outcomeName(stats.outcome);
  out["k_reached"] = stats.k_reached.load();
  out["k_max"] = stats.k_max.load();
  out["candidates_enumerated"] = stats.candidates_enumerated.load();
  out["candidates_residue_rejected"] = stats.candidates_residue_rejected.load();
  out["candidates_restart_limited"] = stats.candidates_restart_limited.load();
  out["candidates_iteration_limited"] =
      stats.candidates_iteration_limited.load();
  out["diophantine_calls"] = stats.diophantine_calls.load();
  out["diophantine_successes"] = stats.diophantine_successes.load();
  out["factoring_calls"] = stats.factoring_calls.load();
  out["factoring_successes"] = stats.factoring_successes.load();
  out["factoring_restarts"] = stats.factoring_restarts.load();
  out["factoring_wall_clock_exits"] = stats.factoring_wall_clock_exits.load();
  out["diophantine_wall_clock_exits"] =
      stats.diophantine_wall_clock_exits.load();
  out["factoring_iterations_total"] = stats.factoring_iterations_total.load();
  out["working_precision_bits"] = stats.working_precision_bits.load();
  out["enumeration_ns"] = stats.enumeration_ns.load();
  out["diophantine_ns"] = stats.diophantine_ns.load();
  return nanobind::make_tuple(gates, out);
}

double rzErrorBinding(RealArg theta, const std::string &gates) {
  cudaq::synth::Real thetaReal = toReal(theta, "theta");
  if (!thetaReal.is_finite())
    throw nanobind::value_error("theta must be finite");

  llvm::FailureOr<cudaq::synth::Circuit> circuit =
      cudaq::synth::Circuit::from_string(gates);
  if (llvm::failed(circuit))
    throw nanobind::value_error(
        ("rz_error: invalid gate string '" + gates +
         "'; expected only H, S, T, X, W, or the identity sentinel I")
            .c_str());

  cudaq::synth::Real error = cudaq::synth::rz_approximation_error(
      cudaq::synth::DOmegaUnitary::from_gates(*circuit), thetaReal);

  // Round up so the result never under-reports the true error. Callers use it
  // in `rz_error(...) <= epsilon` checks.
  return mpfr_get_d(error.get_mpfr(), MPFR_RNDU);
}

std::string normalizedBinding(const std::string &gates) {
  llvm::FailureOr<cudaq::synth::Circuit> circuit =
      cudaq::synth::Circuit::from_string(gates);
  if (llvm::failed(circuit))
    throw nanobind::value_error(
        ("normalized: invalid gate string '" + gates +
         "'; expected only H, S, T, X, W, or the identity sentinel I")
            .c_str());
  return circuit->normalized().to_string();
}

} // namespace

NB_MODULE(_cudaq_synth, m) {
  m.doc() = "Internal bindings for the Clifford+T rotation synthesis "
            "library (cudaq-synth).";

  m.def(
      "_normalized", &normalizedBinding, nanobind::arg("gates"),
      R"doc(Return the exact Matsumoto-Amano normal form of a gate string.)doc");

  m.def("_gridsynth_with_stats", &gridsynthWithStatsBinding,
        nanobind::arg("theta"), nanobind::arg("epsilon"),
        nanobind::arg("max_factoring_iterations") =
            cudaq::synth::details::DEFAULT_MAX_FACTORING_ITERATIONS,
        nanobind::arg("max_candidate_iterations") =
            cudaq::synth::details::DEFAULT_MAX_CANDIDATE_ITERATIONS,
        nanobind::arg("max_factoring_restarts") =
            cudaq::synth::details::DEFAULT_MAX_FACTORING_RESTARTS,
        nanobind::arg("max_odgp_scan_steps") =
            cudaq::synth::details::DEFAULT_MAX_ODGP_SCAN_STEPS,
        nanobind::arg("seed") = nanobind::none(),
        nanobind::arg("timeout_ms") = nanobind::none(),
        R"doc(Private: gridsynth plus its work counters, as (gates, stats).

Not part of the public API and not re-exported from cudaq.synth. The shape
of a public statistics surface is part of the pending synth API restructure;
this exists so budget tuning can be measured rather than guessed at.)doc");

  m.def(
      "gridsynth", &gridsynthBinding, nanobind::arg("theta"),
      nanobind::arg("epsilon"),
      nanobind::arg("max_factoring_iterations") =
          cudaq::synth::details::DEFAULT_MAX_FACTORING_ITERATIONS,
      nanobind::arg("max_candidate_iterations") =
          cudaq::synth::details::DEFAULT_MAX_CANDIDATE_ITERATIONS,
      nanobind::arg("max_factoring_restarts") =
          cudaq::synth::details::DEFAULT_MAX_FACTORING_RESTARTS,
      nanobind::arg("max_odgp_scan_steps") =
          cudaq::synth::details::DEFAULT_MAX_ODGP_SCAN_STEPS,
      nanobind::arg("seed") = nanobind::none(),
      nanobind::arg("timeout_ms") = nanobind::none(),
      R"doc(Synthesize a Clifford+T circuit approximating R_z(theta) to precision epsilon.

Implements the grid-synthesis algorithm of Ross & Selinger (arXiv:1403.2975,
Algorithm 7.6). The returned gate string is in Matsumoto-Amano normal form
with minimum T-count up to the search budgets below.

Precision is measured in the operator norm (a.k.a. spectral norm, the
induced 2-norm ||A|| = sigma_max(A)). The synthesized unitary U satisfies
||R_z(theta) - U|| <= epsilon. This is the norm used in Ross & Selinger
section 7.1, equation (13).

    How the budgets trade. The search walks a denominator exponent k upward,
    and at each k enumerates candidate circuits and tests each by factoring an
    integer. The first candidate that works ends the search. The T-count is
    2k-2 or 2k, so a candidate solved at a smaller k is a strictly shorter
    circuit. Every budget below bounds work spent on one candidate, so
    lowering any of them is faster and never produces a shorter circuit.

Args:
    theta: Target rotation angle (float, or decimal str for arbitrary precision).
    epsilon: Approximation precision in operator norm, must be > 0
        (float, or str).
    max_factoring_iterations: Pollard-rho iterations one factoring
        attempt may spend before giving up on its composite. Nearly all
        synthesis time goes here, and this is the budget that decides when
        a candidate is abandoned, so it is the one to raise to trade
        runtime for fewer T gates. Default 500000.
    max_candidate_iterations: The same iterations, summed over every
        factoring attempt made for one candidate. Bounds worst-case time
        per call, which the two options around it do not. They are per
        attempt and per composite. Doubling it costs ~1.8x runtime and
        buys no T gates, so it is a tail control, not a quality one.
        Default 2000000.
    max_factoring_restarts: Consecutive failed factoring attempts allowed
        on one composite before the candidate is abandoned. Each attempt
        re-rolls a random parameter, so a retry searches differently
        rather than repeating. Default 8, already well above the measured
        need.
    max_odgp_scan_steps: Steps one enumeration line scan may take without
        producing a candidate. Candidates are found by scanning along grid
        lines, and a line carrying none can be scanned indefinitely. This
        governs the supply of candidates rather than effort per candidate,
        so starving it makes synthesis fail rather than return a longer
        circuit. Default 65536.
    seed: Seed for the internal factoring RNG. Default None draws from
        system entropy, so repeated calls explore different factoring
        attempts and their runtimes can differ by orders of magnitude.
        Pass an integer to make a run replayable.
    timeout_ms: Optional wall-clock limit on the whole call. Default None
        is the reproducible configuration: the budgets above count work
        rather than time, so the same inputs do the same work on any
        machine. An escape hatch, not a tuning knob.

Returns:
    A string of gate characters from the alphabet {H, S, T, X, W}, where
    H is Hadamard, S is the phase gate (S = T^2), T is the pi/8 gate,
    X is Pauli-X, and W is the scalar global-phase gate
    W = omega * I with omega = e^{i*pi/4} (i.e. W^8 = I).

    Characters are listed in **matrix-multiplication order**: the leftmost
    character is the leftmost matrix factor, so a string "G0 G1 ... Gn-1"
    denotes the unitary U = G0 * G1 * ... * G(n-1). When read as a circuit
    diagram (i.e. order of application to a state), gates are applied
    right-to-left: G(n-1) first, G0 last.

    The identity is returned as the single character 'I'.

Raises:
    ValueError: if theta or epsilon is a string that does not parse as a
        number, if theta is not finite, if epsilon is not finite and
        strictly positive, or if synthesis fails (degenerate epsilon region
        or search space exhausted).
)doc");

  m.def(
      "rz_error", &rzErrorBinding, nanobind::arg("theta"),
      nanobind::arg("gates"),
      R"doc(Operator-norm distance between R_z(theta) and a Clifford+T gate string.

Reconstructs the exact unitary U denoted by `gates` over D[omega] and returns
||R_z(theta) - U||, the spectral norm (largest singular value) of the
difference. This is the same norm the epsilon argument of gridsynth is
measured in (Ross & Selinger, arXiv:1403.2975, section 7.1, equation (13)),
so a synthesized sequence always satisfies
rz_error(theta, gridsynth(theta, epsilon)) <= epsilon.

Args:
    theta: Target rotation angle (float, or decimal str for arbitrary
        precision).
    gates: Gate string over {H, S, T, X, W}, in matrix-multiplication order.
        The identity sentinel 'I' is accepted anywhere and contributes no
        gate.

Returns:
    The approximation error as a float. The value is computed at full
    MPFR precision and rounded toward zero on the way out to a double.

Raises:
    ValueError: if theta is a string that does not parse as a number, if
        theta is not finite, or if gates contains a character outside
        {H, S, T, X, W, I}.
)doc");
}
