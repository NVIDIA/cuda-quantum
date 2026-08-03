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
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

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

std::string gridsynthBinding(RealArg theta, RealArg epsilon,
                             int diophantine_timeout_ms,
                             int factoring_timeout_ms) {
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

  llvm::FailureOr<cudaq::synth::Circuit> result = llvm::failure();
  {
    nanobind::gil_scoped_release nogil;
    result = cudaq::synth::gridsynth(
        thetaReal, epsilonReal, diophantine_timeout_ms, factoring_timeout_ms);
  }
  if (llvm::failed(result))
    throw nanobind::value_error(
        "gridsynth: failed to synthesize a Clifford+T approximation "
        "(degenerate epsilon region or search exhausted)");
  return result->to_string();
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

} // namespace

NB_MODULE(_cudaq_synth, m) {
  m.doc() = "Internal bindings for the Clifford+T rotation synthesis "
            "library (cudaq-synth).";

  m.def(
      "gridsynth", &gridsynthBinding, nanobind::arg("theta"),
      nanobind::arg("epsilon"), nanobind::arg("diophantine_timeout_ms") = 200,
      nanobind::arg("factoring_timeout_ms") = 50,
      R"doc(Synthesize a Clifford+T circuit approximating R_z(theta) to precision epsilon.

Implements the grid-synthesis algorithm of Ross & Selinger (arXiv:1403.2975,
Algorithm 7.6). The returned gate string is in Matsumoto-Amano normal form
with minimum T-count up to search timeouts. 

Precision is measured in the operator norm (a.k.a. spectral norm, the
induced 2-norm ||A|| = sigma_max(A)). The synthesized unitary U satisfies
||R_z(theta) - U|| <= epsilon. This is the norm used in Ross & Selinger
section 7.1, equation (13).

Args:
    theta: Target rotation angle (float, or decimal str for arbitrary precision).
    epsilon: Approximation precision in operator norm, must be > 0
        (float, or str).
    diophantine_timeout_ms: Per-candidate timeout for the Diophantine
        solver. Higher values improve optimality at the cost of
        worst-case latency. Default 200.
    factoring_timeout_ms: Per-candidate timeout for integer factoring
        inside the Diophantine solver. Default 50.

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
