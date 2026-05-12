/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Ring/Domega.h"
#include "Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Support/Result.h"

/// Diophantine equation solver for the gridsynth algorithm.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §6 (Theorem 6.2)
/// and Appendix C (detailed proofs).
///
/// PROBLEM: Given ξ ∈ D[√2], find t ∈ D[ω] such that t†t = ξ.
/// This is equation (9) in the paper, and corresponds to step 2(c)
/// of Algorithm 7.6.
///
/// CONTEXT: After finding a grid-problem candidate u ∈ D[ω] satisfying
/// u ∈ R_ε and u● ∈ D̄, we need t such that t†t + u†u = 1, i.e.,
/// t†t = 1 - u†u = ξ. If such t exists, then U = [[u, -t†], [t, u†]]
/// is a valid Clifford+T unitary approximating R_z(θ).
///
/// TIMEOUTS: Factoring and Diophantine solving have configurable
/// timeouts (in milliseconds). When factoring times out for a
/// particular candidate u, the algorithm skips to the next candidate
/// (step 2(b) of Algorithm 7.6). This corresponds to the "near-optimal"
/// case of Proposition 8.8.

namespace cudaq::synth {

/// Solve t†t = ξ for ξ ∈ D[√2] (dyadic case).
///
/// Solves the norm equation for xi ∈ Z[i, 1/√2] by reducing to a norm equation
/// in Z[√2], using δ = 1 + ω to handle odd √2-denominator exponents, and then
/// lifting the solution back to Z[ω] with the appropriate dyadic denominator.
///
/// @param xi The element ξ ∈ D[√2] to decompose.
/// @param diophantine_timeout Total timeout in milliseconds for the solver.
/// @param factoring_timeout Timeout in milliseconds for each factoring attempt.
/// @return t ∈ D[ω] such that t†t = ξ, or failure() if no solution found.
FailureOr<DOmega> diophantine_dyadic(const DSqrt2 &xi, i32 diophantine_timeout,
                                     i32 factoring_timeout);

} // namespace cudaq::synth
