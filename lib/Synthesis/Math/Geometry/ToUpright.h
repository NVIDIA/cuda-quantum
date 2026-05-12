/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/Ellipse.h"
#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Integer.h"
#include "cudaq/Synthesis/Support/Result.h"

#include <cmath>
#include <string>

/// ToUpright: Iterative algorithm for transforming a pair of ellipses to
/// an "upright" position using special grid operators.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §5.4 (Theorem 5.16)
/// and Appendix A (Step Lemma, Lemma A.5).
///
/// OVERVIEW:
/// Given two ellipses A and B (representing the convex sets in a two-
/// dimensional grid problem), this module finds a special grid operator G
/// such that G(A) and G●(B) are both 1/6-upright (Definition 5.7).
///
/// Once the ellipses are upright, the grid problem can be efficiently
/// solved by reducing it to upright-rectangle enumeration (Lemma 5.8),
/// which in turn reduces to one-dimensional grid problems (Lemma 5.6).
///
/// ALGORITHM (proof of Theorem 5.16):
/// 1. Normalize the ellipse pair so both D-matrices have determinant 1.
/// 2. Repeatedly apply the Step Lemma (Lemma A.5) to reduce the skew
///    Skew(D,Δ) = b² + β² by at least 10% per iteration.
/// 3. Stop when Skew ≤ 15, which guarantees both ellipses are 1/6-upright
///    (since uprightness = π/(4√(b²+1)) ≥ π/16 ≥ 1/6).
/// 4. Accumulate the product G = G₁ · G₂ · ... · Gₙ of all applied
///    grid operators.
///
/// Complexity: O(log(1/M)) arithmetic operations where M is the initial
/// uprightness (Theorem 5.16).
///
/// STEP LEMMA IMPLEMENTATION (Appendix A):
/// The `step_lemma` function implements the case analysis from §A.1-A.6:
///
/// - Z operation (Remark A.20): negates anti-diagonal entries of the
///   state. Applied when B.b < 0 to ensure β ≥ 0 WLOG.
///
/// - X operation (Remark A.20): swaps diagonal entries of the state.
///   Applied when bias(A)·bias(B) < 1 to ensure z + ζ ≥ 0 WLOG.
///
/// - Shift / Sigma operation (§A.1, Lemma A.11): adjusts the bias
///   (ζ - z) to [-1, 1] using the shift operators σ and τ. Applied
///   when the bias is too extreme.
///
/// - S operation: a combined shift for very extreme bias values.
///
/// - R operation (§A.2, Lemma A.13): a rotation-like grid operator
///   applied when both z and ζ are in [-0.8, 0.8]. Uses the grid
///   operator R from Figure 6 of the paper.
///
/// - K operation (§A.3, Lemma A.15): applied when b,β ≥ 0, z ≤ 0.3,
///   and 0.8 ≤ ζ.
///
/// - K● operation: the √2-conjugated version of K, applied when the
///   roles of (D,Δ) are swapped.
///
/// - A operation (§A.4, Lemma A.17): a parameterized shear, applied
///   when b,β ≥ 0 and z,ζ ≥ 0.3. The parameter n = max(1, ⌊λ^c/2⌋).
///
/// - B operation (§A.5, Lemma A.19): applied when b ≤ 0 ≤ β and
///   z,ζ ≥ -0.2. The parameter n = max(1, ⌊λ^c/√2⌋).
///
/// DEVIATION from paper: The paper uses exact thresholds derived from
/// the algebraic analysis (e.g., sinh_λ(0.8), cosh_λ(1.3), g(1/4λ)).
/// This implementation uses approximate floating-point thresholds for
/// the case selection (e.g., bias > 33.971, bias < 0.029437), which
/// are the numerical values of the algebraic bounds. This is safe
/// because the thresholds only affect efficiency (convergence rate),
/// not correctness—any grid operator that reduces skew works.
///
/// DEVIATION: The paper decomposes G = opG_l * opG_r where opG_l
/// accumulates shift operators (which are not themselves grid operators)
/// and opG_r accumulates grid operators. The final result G = opG_l * opG_r
/// is a valid special grid operator.
namespace cudaq::synth {

// Pair-level computed quantities

/// Combined skew of the ellipse pair: b²_A + b²_B (equation (33)).
///
/// The Step Lemma guarantees ≥10% reduction of this quantity per iteration.
/// The algorithm terminates when pair_skew ≤ 15.
inline Real pair_skew(const Ellipse &A, const Ellipse &B) {
  return skew(A) + skew(B);
}

/// Bias ratio of the ellipse pair: bias(B) / bias(A) = (d_B/a_B) / (d_A/a_A).
///
/// Used by the Step Lemma case-selection logic to choose the reduction
/// operator (S, Sigma, R, K, A, B operations).
inline Real pair_bias(const Ellipse &A, const Ellipse &B) {
  return bias(B) / bias(A);
}

/// Joint transformation: applies G to A and G● to B (Definition A.3).
/// Returns failure() if any transform step fails (e.g. non-special GridOp).
LogicalResult apply_grid_op(Ellipse &A, Ellipse &B, const GridOp &g);

/// Apply a reduction step: transforms the ellipse pair by new_opG and
/// accumulates new_opG into opG_r (the right factor of the total operator).
/// Implements (D,Δ) ↦ (D,Δ)·G from Definition A.3.
/// Returns failure() propagated from apply_grid_op.
LogicalResult reduction(Ellipse &A, Ellipse &B, GridOp &opG_r,
                        const GridOp &new_opG);

/// Shift operation (§A.1, Definition A.7):
/// (D,Δ)·Shift^n = (σⁿDσⁿ, τⁿΔτⁿ)
/// This preserves skew while adjusting bias by 2n (Lemma A.8).
/// Not a grid operator itself, but used to normalize the bias before
/// applying the actual reduction grid operators.
void shift_ellipses(Ellipse &A, Ellipse &B, const Integer &n);

/// Step Lemma implementation (Lemma A.5):
/// Given a state with Skew ≥ 15, finds a special grid operator G
/// such that Skew((D,Δ)·G) ≤ 0.9 · Skew(D,Δ).
/// Sets 'end' to true when Skew ≤ 15 (algorithm complete).
/// Returns failure() if any internal operation fails.
LogicalResult step_lemma(Ellipse &A, Ellipse &B, GridOp &opG_l,
                         GridOp &opG_r, bool &end);

// Result structure for to_upright
struct UprightResult {
  GridOp opG;
  Rectangle bboxA;
  Rectangle bboxB;

  UprightResult(const GridOp &op, const Rectangle &bA, const Rectangle &bB)
      : opG(op), bboxA(bA), bboxB(bB) {}

  /// Returns "UprightResult(opG=..., bboxA=..., bboxB=...)" delegating to
  /// GridOp::to_string() and Rectangle::to_string(). Intended for logging.
  std::string to_string() const {
    return "UprightResult(opG=" + opG.to_string() +
           ", bboxA=" + bboxA.to_string() +
           ", bboxB=" + bboxB.to_string() + ")";
  }
};

/// Transform an ellipse pair to upright position (Theorem 5.16).
/// Returns the accumulated grid operator G and bounding boxes of
/// G(setA) and G●(setB), or failure() if the ellipses are degenerate.
FailureOr<UprightResult> to_upright(const Ellipse &setA, const Ellipse &setB);

} // namespace cudaq::synth
