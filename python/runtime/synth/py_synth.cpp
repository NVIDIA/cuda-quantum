/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Support/Result.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>

#include <string>
#include <variant>

namespace {

using RealArg = std::variant<double, std::string>;

cudaq::synth::Real toReal(const RealArg &arg) {
  if (std::holds_alternative<double>(arg))
    return cudaq::synth::Real(std::get<double>(arg));
  return cudaq::synth::Real(std::get<std::string>(arg));
}

std::string gridsynthBinding(RealArg theta, RealArg epsilon,
                             int diophantine_timeout_ms,
                             int factoring_timeout_ms) {
  cudaq::synth::Real thetaReal = toReal(theta);
  cudaq::synth::Real epsilonReal = toReal(epsilon);

  if (!(epsilonReal > 0))
    throw nanobind::value_error("epsilon must be strictly positive");

  auto result = cudaq::synth::gridsynth(
      thetaReal, epsilonReal, diophantine_timeout_ms, factoring_timeout_ms);
  if (cudaq::synth::failed(result))
    throw nanobind::value_error(
        "gridsynth: failed to synthesize a Clifford+T approximation "
        "(degenerate epsilon region or search exhausted)");
  return result->to_string();
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
with minimum T-count.

Args:
    theta: Target rotation angle (float, or str for arbitrary precision).
    epsilon: Approximation precision, must be > 0 (float, or str).
    diophantine_timeout_ms: Per-candidate timeout for the Diophantine
        solver. Higher values improve optimality at the cost of
        worst-case latency. Default 200.
    factoring_timeout_ms: Per-candidate timeout for integer factoring
        inside the Diophantine solver. Default 50.

Returns:
    A string of gate characters from the alphabet {H, S, T, X, W}. The
    identity is returned as the single character 'I'.

Raises:
    ValueError: if epsilon <= 0, or if synthesis fails (degenerate
        epsilon region or search space exhausted).
)doc");
}
