/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Interval.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Grid/Odgp.h"
#include "Support/Stepper.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"

#include <optional>
#include <string>

namespace cudaq::synth {
class ConvexSet;
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
/// The caller iterates k = 0, 1, 2, ... calling `solve_tdgp(k, ...)` for each
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
/// LAZINESS: Implemented as a hand-rolled stepper (no coroutines). The entire
/// pipeline (x-ODGP, y-ODGP, parity-ODGP) is lazy, so early termination by
/// the caller propagates through the full chain without wasting computation
/// on unused solutions.

/// Lazy stepper for the scaled TDGP at denominator exponent k.
///
/// Yields all u ∈ (1/√2^k)·Z[ω] satisfying u ∈ setA and u● ∈ setB.
///
/// Usage:
///   for (const DOmega &z : TdgpStepper(k, ...)) { ... }
///
/// Composes `OdgpScaledStepper` (for the β-iteration) and
/// `OdgpScaledWithParityStepper` (for the α-iteration per β). The
/// transformed-membership filter `setA.contains(z_tr) && setB.contains(...)`
/// is applied inside `next()`; rejected candidates do not increment the
/// yield counter and are silently skipped.
///
/// Non-copyable, non-movable.
class TdgpStepper : public StepperBase<TdgpStepper, DOmega> {
public:
  TdgpStepper(Integer k, const ConvexSet &setA, const ConvexSet &setB,
              const GridOp &opG_inv, Rectangle bboxA, Rectangle bboxB,
              Interval bboxA_y_fattened, Interval bboxB_y_fattened);
  ~TdgpStepper();

  TdgpStepper(const TdgpStepper &) = delete;
  TdgpStepper &operator=(const TdgpStepper &) = delete;
  TdgpStepper(TdgpStepper &&) = delete;
  TdgpStepper &operator=(TdgpStepper &&) = delete;

  const DOmega *next();

private:
  // Constants from constructor.
  Integer k_;
  const ConvexSet *setA_;
  const ConvexSet *setB_;
  GridOp opG_inv_;
  Rectangle bboxA_;
  Rectangle bboxB_;
  Interval bboxA_y_fattened_;
  Interval bboxB_y_fattened_;

  // Computed once in constructor.
  DSqrt2 alpha0_;
  DSqrt2 dx_;
  DOmega v_common_;
  DOmega v_conj_;
  Real two_pow_k_;

  // β-iteration; emplaced if construction succeeds (i.e. alpha0_ exists).
  std::optional<OdgpScaledStepper> beta_gen_;
  // Current β (the value most recently advanced from beta_gen_) and the
  // alpha-iteration over it.
  DSqrt2 current_beta_;
  std::optional<OdgpScaledWithParityStepper> alpha_gen_;

  // Output buffer.
  DOmega last_sol_;

  // Diagnostics / state.
  bool exhausted_ = false;
  int yielded_ = 0;
  int skipped_betas_ = 0;
  std::string close_reason_;

  /// Advance beta_gen_ to the next β with a non-empty (A,B) intersection
  /// interval, set current_beta_, and emplace alpha_gen_. Returns false if
  /// β-iteration is exhausted (no more candidates).
  bool advance_to_next_beta();
};

/// Solve the scaled TDGP for a given denominator exponent k.
///
/// Returns a lazy stepper of all u ∈ (1/√2^k)·Z[ω] with u ∈ setA and
/// u● ∈ setB. Solutions are produced on demand; destroy the stepper to stop
/// enumeration early.
///
/// @param k Denominator exponent (determines grid scale 1/√2^k)
/// @param setA First convex set constraint (epsilon region in `gridsynth`)
/// @param setB Second convex set constraint (unit disk in `gridsynth`)
/// @param opG_inv Inverse of upright transformation (from ToUpright)
/// @param bboxA Bounding box for transformed setA
/// @param bboxB Bounding box for transformed setB
/// @param bboxA_y_fattened Fattened y-interval for setA (handles ω-offset)
/// @param bboxB_y_fattened Fattened y-interval for setB (handles ω-offset)
/// @return Lazy stepper of solutions (may be empty)
TdgpStepper solve_tdgp(Integer k, const ConvexSet &setA, const ConvexSet &setB,
                       const GridOp &opG_inv, Rectangle bboxA, Rectangle bboxB,
                       Interval bboxA_y_fattened, Interval bboxB_y_fattened);

} // namespace cudaq::synth
