/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include "llvm/Support/LogicalResult.h"

#include <cmath>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Grid synthesis: optimal Clifford+T approximation of R_z(theta)
//===----------------------------------------------------------------------===//
//
// Reference: Ross & Selinger, arXiv:1403.2975, Algorithm 7.6.

namespace details {

/// Per-candidate budget for the Diophantine solver. Larger values let the
/// algorithm push through harder factorisations and find smaller-T-count
/// solutions; the trade-off is worst-case latency per candidate. On timeout
/// the candidate is dropped and the search moves on.
inline constexpr int32_t DEFAULT_DIOPHANTINE_TIMEOUT_MS = 200;

/// Per-attempt budget for Pollard-rho integer factoring inside the
/// Diophantine solver. Hard composites that exceed this budget cause the
/// enclosing candidate to be skipped.
inline constexpr int32_t DEFAULT_FACTORING_TIMEOUT_MS = 50;

} // namespace details

/// Core grid-synthesis search.
///
/// Runs Algorithm 7.6 until it produces a DOmegaUnitary approximating
/// R_z(theta) to within `epsilon`. Returns failure() if the epsilon region
/// is degenerate or the search exhausts its budgets without finding a valid
/// solution.
///
/// @param theta                  Target rotation angle.
/// @param epsilon                Approximation precision.
/// @param diophantine_timeout_ms Per-candidate Diophantine budget.
/// @param factoring_timeout_ms   Per-attempt integer-factoring budget.
llvm::FailureOr<DOmegaUnitary> gridsynth_unitary(
    const Real &theta, const Real &epsilon,
    int32_t diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
    int32_t factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

/// End-to-end gridsynth entry point: search for a DOmegaUnitary via
/// `gridsynth_unitary`, then realise it as an explicit Clifford+T circuit
/// in Matsumoto-Amano normal form with minimum T-count via
/// `kmm_synthesize`.
llvm::FailureOr<Circuit> gridsynth(
    const Real &theta, const Real &epsilon,
    int32_t diophantine_timeout_ms = details::DEFAULT_DIOPHANTINE_TIMEOUT_MS,
    int32_t factoring_timeout_ms = details::DEFAULT_FACTORING_TIMEOUT_MS);

} // namespace cudaq::synth
