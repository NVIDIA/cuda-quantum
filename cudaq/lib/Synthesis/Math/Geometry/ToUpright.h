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
#include "cudaq/Synthesis/Math/Integer.h"
#include "llvm/Support/LogicalResult.h"

#include <cmath>
#include <string>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// To-upright preprocessing
//===----------------------------------------------------------------------===//
//
// Reference: Ross & Selinger, arXiv:1403.2975, sec. 5.4 (Theorem 5.16) and
// Appendix A (Step Lemma, Lemma A.5).
//
// Goal. Given a pair of ellipses (A, B) representing the convex sets in a 2D
// grid problem, find a special grid operator G such that G(A) and
// conj_sq2(G)(B) are both 1/6-upright (Definition 5.7). With the pair in
// upright form the grid problem reduces to upright-rectangle enumeration
// (Lemma 5.8), which in turn factors into one-dimensional grid problems
// (Lemma 5.6).
//
// Algorithm (proof of Theorem 5.16).
//   1. Normalise the pair so both D-matrices have determinant 1.
//   2. Iteratively apply the Step Lemma (Lemma A.5) to drop the skew
//      Skew(D, Delta) = b^2 + beta^2 by at least 10% per step.
//   3. Stop once Skew <= 15; the pair is then 1/6-upright because
//      uprightness = pi / (4 * sqrt(b^2 + 1)) >= pi / 16 >= 1/6.
//   4. Accumulate the product G = G_1 * G_2 * ... * G_n of all applied
//      grid operators.
// Complexity O(log(1/M)) arithmetic operations, where M is the initial
// uprightness (Theorem 5.16).
//
// Step Lemma cases (Appendix A.1 - A.6). The `step_lemma` function dispatches
// over the following operations; each one consumes one iteration:
//
//   Z   (Remark A.20)  Negates anti-diagonal entries of the state. Applied
//                      when B.b < 0 to enforce beta >= 0.
//   X   (Remark A.20)  Swaps diagonal entries. Applied when
//                      bias(A) * bias(B) < 1 to enforce z + zeta >= 0.
//   Sigma (Lemma A.11) Bias shift via the shift operators sigma, tau,
//                      brings (zeta - z) back into [-1, 1].
//   S                  Combined shift used for extreme bias values.
//   R   (Lemma A.13)   Rotation-like grid operator (Figure 6 of the paper)
//                      applied when both z and zeta are in [-0.8, 0.8].
//   K   (Lemma A.15)   Applied when b, beta >= 0, z <= 0.3, 0.8 <= zeta.
//   K*                 The sqrt(2)-conjugate of K, for the swapped roles
//                      case.
//   A   (Lemma A.17)   Parameterised shear: applied when b, beta >= 0 and
//                      z, zeta >= 0.3, with n = max(1, floor(lambda^c / 2)).
//   B   (Lemma A.19)   Applied when b <= 0 <= beta and z, zeta >= -0.2,
//                      with n = max(1, floor(lambda^c / sqrt(2))).
//
// Deviation from the paper: the paper expresses the case thresholds
// symbolically (sinh_lambda(0.8), cosh_lambda(1.3), g(1/(4 * lambda)), ...);
// this implementation pre-evaluates them to floating-point constants (e.g.
// bias > 33.971, bias < 0.029437). The thresholds only affect convergence
// rate, not correctness -- any grid operator that strictly reduces skew
// works.
//
// Deviation: G is built as G = opG_l * opG_r where opG_l accumulates shift
// operators (which are not themselves grid operators in isolation) and
// opG_r accumulates honest grid operators. Their product is a valid special
// grid operator.

//===----------------------------------------------------------------------===//
// Pair-level computed quantities
//===----------------------------------------------------------------------===//

/// Combined skew of an ellipse pair: skew(A) + skew(B) = b_A^2 + b_B^2
/// (equation (33)). The Step Lemma guarantees at least a 10% reduction of
/// this quantity per iteration; the loop terminates once it drops to 15 or
/// below.
inline Real pair_skew(const Ellipse &A, const Ellipse &B) {
  return skew(A) + skew(B);
}

/// Bias ratio of a pair: bias(B) / bias(A). Drives the Step Lemma case
/// selection (S, Sigma, R, K, A, B).
inline Real pair_bias(const Ellipse &A, const Ellipse &B) {
  return bias(B) / bias(A);
}

//===----------------------------------------------------------------------===//
// Step-Lemma building blocks
//===----------------------------------------------------------------------===//

/// Joint transformation (Definition A.3): apply G to A and conj_sq2(G) to B.
/// Returns failure() if any transform step fails (e.g. a non-special G in
/// the Fallback path).
llvm::LogicalResult apply_grid_op(Ellipse &A, Ellipse &B, const GridOp &g);

/// Apply one reduction step and accumulate the operator into `opG_r` -- the
/// right factor of the running product. Implements (D, Delta) -> (D, Delta)
/// * G from Definition A.3. Propagates failure from `apply_grid_op`.
llvm::LogicalResult reduction(Ellipse &A, Ellipse &B, GridOp &opG_r,
                              const GridOp &new_opG);

/// Shift operation (Definition A.7): (D, Delta) * Shift^n = (sigma^n D
/// sigma^n, tau^n Delta tau^n). Preserves skew while adjusting bias by 2*n
/// (Lemma A.8). Not itself a grid operator -- used to renormalise the bias
/// before applying the real reduction operators.
void shift_ellipses(Ellipse &A, Ellipse &B, const Integer &n);

/// Step Lemma (Lemma A.5): given Skew >= 15, find a special grid operator G
/// with Skew((D, Delta) * G) <= 0.9 * Skew(D, Delta). Sets `end` to true
/// once Skew <= 15 (algorithm complete). Returns failure() if any internal
/// operation fails.
llvm::LogicalResult step_lemma(Ellipse &A, Ellipse &B, GridOp &opG_l,
                               GridOp &opG_r, bool &end);

//===----------------------------------------------------------------------===//
// Result of to_upright
//===----------------------------------------------------------------------===//

struct UprightResult {
  GridOp opG;
  Rectangle bboxA;
  Rectangle bboxB;

  UprightResult(const GridOp &op, const Rectangle &bA, const Rectangle &bB)
      : opG(op), bboxA(bA), bboxB(bB) {}

  /// "UprightResult(opG=..., bboxA=..., bboxB=...)" for debug logging.
  std::string to_string() const {
    return "UprightResult(opG=" + opG.to_string() +
           ", bboxA=" + bboxA.to_string() + ", bboxB=" + bboxB.to_string() +
           ")";
  }
};

/// Drive an ellipse pair to upright position (Theorem 5.16). Returns the
/// accumulated grid operator G together with the bounding boxes of G(setA)
/// and conj_sq2(G)(setB), or failure() if either input ellipse is
/// degenerate or an intermediate transform fails.
llvm::FailureOr<UprightResult> to_upright(const Ellipse &setA,
                                          const Ellipse &setB);

} // namespace cudaq::synth
