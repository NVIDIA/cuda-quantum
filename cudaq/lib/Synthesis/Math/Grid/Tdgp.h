/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/GridOp.h"
#include "Support/Generator.h"

namespace cudaq::synth {
class ConvexSet;
class DOmega;
class Rectangle;
class Interval;
} // namespace cudaq::synth

namespace cudaq::synth {

/// TDGP (Two-Dimensional Grid Problem) solver for finding elements in D[ω].
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §5.1-5.7
/// (Theorem 5.18, Propositions 5.21-5.22).
///
/// The two-dimensional grid problem (Definition 5.3):
///   Given convex sets A, B ⊂ R², find u ∈ Z[ω] with u ∈ A and u● ∈ B.
///
/// The scaled variant (Definition 5.20) for fixed k ≥ 0:
///   Find u ∈ (1/√2^k)Z[ω] with u ∈ A and u● ∈ B.
///
/// Algorithm (Theorem 5.18, combining all pieces):
///
/// 1. UPRIGHT PREPROCESSING: The ToUpright module finds a special grid
///    operator G making G(A) and G●(B) upright (Theorem 5.16).
///    The TDGP solver receives the precomputed G⁻¹, bounding boxes
///    bboxA and bboxB of the transformed ellipses.
///
/// 2. REDUCTION TO UPRIGHT RECTANGLES (Lemma 5.6):
///    By Lemma 5.5, any u ∈ Z[ω] has the form u = α + βi (even case)
///    or u = α + βi + ω (odd case), where α, β ∈ Z[√2].
///    The 2D grid constraints decompose into independent 1D constraints
///    on the real and imaginary parts.
///
/// 3. ONE-DIMENSIONAL SOLVING:
///    - Solve ODGP for x-coordinates: α ∈ bboxA.x, α● ∈ bboxB.x
///    - Solve ODGP for y-coordinates: β ∈ bboxA.y, β● ∈ bboxB.y
///    These use the bounding boxes from step 1.
///
/// 4. LINE SCANNING (the "line intersection" approach):
///    For each y-solution β, parameterize a line through the candidates:
///      z(t) = z₀ + t·v  where z₀ uses the first x-solution and v is the
///      step direction (scaled by the grid spacing).
///    Intersect this line with both A and B to find the valid range of t,
///    then solve a 1D scaled ODGP with parity for t.
///
/// 5. FILTERING: Transform candidates back through G⁻¹ and verify
///    membership in the original (non-upright) sets A and B.
///
/// The caller iterates k = 0, 1, 2, ... calling solve_tdgp(k) for each
/// denominator exponent until a valid solution is found. This provides
/// enumeration in order of increasing T-count (Lemma 7.3, Proposition 5.22).
///
/// DEVIATION from paper: The paper handles both the even (u = α + βi)
/// and odd (u = α + βi + ω) cases as two separate 1D grid problems
/// (Lemma 5.6). This implementation handles both cases implicitly
/// through the scaled-with-parity ODGP variant, using the fattened
/// bounding boxes (bboxA_y_fattened, bboxB_y_fattened) to account
/// for the ω-offset.
///
/// LAZINESS: Returns a generator<DOmega> that produces solutions on demand.
/// The entire pipeline (x-ODGP, y-ODGP, parity-ODGP) is lazy, so early
/// termination by the caller propagates through the full chain without
/// wasting computation on unused solutions.

/// Solve the scaled TDGP for a given denominator exponent k.
///
/// Returns a lazy generator of all u ∈ (1/√2^k)·Z[ω] with u ∈ setA and
/// u● ∈ setB. Solutions are produced on demand; destroy the generator to
/// stop enumeration early.
///
/// @param k Denominator exponent (determines grid scale 1/√2^k)
/// @param setA First convex set constraint (epsilon region in `gridsynth`)
/// @param setB Second convex set constraint (unit disk in `gridsynth`)
/// @param opG_inv Inverse of upright transformation (from ToUpright)
/// @param bboxA Bounding box for transformed setA
/// @param bboxB Bounding box for transformed setB
/// @param bboxA_y_fattened Fattened y-interval for setA (handles ω-offset)
/// @param bboxB_y_fattened Fattened y-interval for setB (handles ω-offset)
/// @return Lazy generator of solutions (may be empty)
generator<DOmega> solve_tdgp(Integer k, const ConvexSet &setA,
                             const ConvexSet &setB, const GridOp &opG_inv,
                             Rectangle bboxA, Rectangle bboxB,
                             Interval bboxA_y_fattened,
                             Interval bboxB_y_fattened);

} // namespace cudaq::synth
