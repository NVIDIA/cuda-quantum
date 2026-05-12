/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/Interval.h"
#include "Math/Ring/Dsqrt2.h"
#include "Math/Ring/Zsqrt2.h"
#include "Support/Generator.h"

#include <cmath>
#include <mpfr.h>

namespace cudaq::synth {

/// One-Dimensional Grid Problem (ODGP) solver.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §4 (Proposition 4.5).
///
/// The one-dimensional grid problem (Definition 4.3):
///   Given intervals I and J, find all α ∈ Z[√2] satisfying α ∈ I and α● ∈ J.
///
/// Algorithm: Initial shift to center the search, iterative `λ-rescaling` until
/// the interval is narrow enough, then direct enumeration with exact bounds
/// checks. Variants support parity constraints (Lemma 5.5) and scaled
/// problems (Proposition 5.21).
///
/// All functions return lazy generators. Solutions are produced on demand in
/// lexicographic (a, b) order. Early termination (destroying the generator
/// before exhaustion) is safe and releases all resources via RAII.

// NOTE: All coroutine functions take parameters by value to avoid the
// dangling-reference pitfall: coroutine frames store copies of parameters,
// but for reference parameters only the reference (pointer) is copied.
// If the caller passed a temporary, the reference dangles after the
// coroutine's first suspension point.

// Core ODGP (Definition 4.3): find all α ∈ Z[√2] with α ∈ I and α● ∈ J.
generator<ZSqrt2> solve_odgp(Interval I, Interval J);

// With parity constraint (for ω-offset case, Lemma 5.5)
generator<ZSqrt2> solve_odgp_with_parity(Interval I, Interval J,
                                         ZSqrt2 parity_hint);

// Scaled ODGP (Proposition 5.21): find all α ∈ (1/√2^denom_exp)Z[√2] with
// α ∈ I and α● ∈ J
generator<DSqrt2> solve_odgp_scaled(Interval I, Interval J, Integer denom_exp);

// Scaled with parity
generator<DSqrt2> solve_odgp_scaled_with_parity(Interval I, Interval J,
                                                Integer denom_exp,
                                                DSqrt2 parity_hint);

} // namespace cudaq::synth
