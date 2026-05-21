/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "llvm/Support/LogicalResult.h"

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Diophantine norm-equation solver
//===----------------------------------------------------------------------===//
//
// Reference: Ross & Selinger, arXiv:1403.2975, sec. 6 (Theorem 6.2) and
// Appendix C (detailed proofs).
//
// Problem: given xi in D[sqrt(2)], find t in D[omega] with
// conj(t) * t = xi. This is equation (9) of the paper and step 2(c) of
// Algorithm 7.6.
//
// In context: after the grid problem yields a candidate u in D[omega] with
// u in R_epsilon and conj_sq2(u) in the closed unit disk, the gridsynth
// driver needs a partner t with conj(t) * t = 1 - conj(u) * u. Combined,
// (u, t) lift to the Clifford+T unitary
//   U = [[ u, -conj(t) ], [ t, conj(u) ]]
// approximating R_z(theta) within the chosen epsilon.

/// Solve conj(t) * t = xi for xi in D[sqrt(2)] (the dyadic case).
///
/// Reduces from D[sqrt(2)] to a Z[sqrt(2)] norm equation -- multiplying by
/// delta = 1 + omega absorbs odd sqrt(2)-denominator exponents (Lemma C.25)
/// -- then lifts the Z[omega] solution back with the appropriate dyadic
/// denominator.
///
/// Both timeouts are advisory; on expiry the solver returns failure() and
/// the caller is expected to move on to the next grid-problem candidate
/// (the "near-optimal" case of Proposition 8.8).
///
/// @param xi                  The element to decompose.
/// @param diophantine_timeout Wall-clock budget for the whole solve, ms.
/// @param factoring_timeout   Wall-clock budget per Pollard-rho attempt, ms.
/// @return t in D[omega] with conj(t) * t = xi, or failure() if no solution
///         is found within the budgets.
llvm::FailureOr<DOmega> diophantine_dyadic(const DSqrt2 &xi,
                                           i32 diophantine_timeout,
                                           i32 factoring_timeout);

} // namespace cudaq::synth
