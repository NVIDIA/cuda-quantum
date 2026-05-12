/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Geometry/ToUpright.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "Support/LogMacros.h"

#include <algorithm>

using namespace cudaq::synth;

namespace {
/// floorsqrt: Floor of square root for Float.
Integer floor_sqrt(const Real &x) { return floor_to_integer(sqrt(x)); }
} // namespace

namespace cudaq::synth {

LogicalResult apply_grid_op(Ellipse &A, Ellipse &B, const GridOp &g) {
  // Use to_real_mat rather than to_mat: avoids allocating 4 imaginary GMP
  // objects that are never used (grid operators are real matrices).
  auto g_mat = to_real_mat(g);
  Real M00 = std::move(g_mat[0][0]);
  Real M01 = std::move(g_mat[0][1]);
  Real M10 = std::move(g_mat[1][0]);
  Real M11 = std::move(g_mat[1][1]);

  static const Real tol(1e-30);
  Real det = M00 * M11 - M01 * M10;

  if (abs(det) < tol) {
    // Singular: both A and B fall back to exact inversion via GridOp::inv().
    if (failed(A.transform_by_gridop_mat(TransformMode::Fallback, M00, M01, M10,
                                         M11, M00, M01, M10, M11, tol, &g)))
      return failure();
    GridOp g_conj = conj_sq2(g);
    if (failed(B.transform_by_gridop(g_conj, TransformMode::Fallback, M00, M01,
                                     M10, M11, tol)))
      return failure();
    return success();
  }

  // Non-singular: compute F⁻¹ once using 1 division + 4 multiplications.
  Real inv_det = 1 / det;
  Real inv00 = M11 * inv_det;
  Real inv01 = -(M01 * inv_det);
  Real inv10 = -(M10 * inv_det);
  Real inv11 = M00 * inv_det;

  if (failed(A.transform_by_gridop_mat(TransformMode::Direct, M00, M01, M10,
                                       M11, inv00, inv01, inv10, inv11, tol,
                                       &g)))
    return failure();

  ZOmega u0c = g.u0().conj_sq2();
  ZOmega u1c = g.u1().conj_sq2();
  Real M00c, M10c, M01c, M11c;
  to_real_imag(u0c, M00c, M10c);
  to_real_imag(u1c, M01c, M11c);
  if (failed(B.transform_by_gridop_mat(TransformMode::Conjugate, M00c, M01c,
                                       M10c, M11c, inv00, inv01, inv10, inv11,
                                       tol, &g)))
    return failure();

  return success();
}

LogicalResult reduction(Ellipse &A, Ellipse &B, GridOp &opG_r,
                        const GridOp &new_opG) {
  if (failed(apply_grid_op(A, B, new_opG)))
    return failure();
  opG_r = new_opG * opG_r;
  return success();
}

void shift_ellipses(Ellipse &A, Ellipse &B, const Integer &n) {
  ZSqrt2 lambda_n = pow(ZSqrt2::lambda(), n);
  // ZSqrt2::lambda() is a unit, so inv() always succeeds.
  ZSqrt2 lambda_inv_n = *inv(lambda_n);
  Real lambda_n_real = to_real(lambda_n);
  Real lambda_inv_n_real = to_real(lambda_inv_n);

  A.scale_a(lambda_inv_n_real);
  A.scale_d(lambda_n_real);
  B.scale_a(lambda_n_real);
  B.scale_d(lambda_inv_n_real);

  if (n.is_odd())
    B.flip_b();
}

LogicalResult step_lemma(Ellipse &A, Ellipse &B, GridOp &opG_l, GridOp &opG_r,
                         bool &end) {
  CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: A={}, B={}", A, B);

  if (B.b() < 0) {
    static const GridOp OP_Z(ZOmega(0, 0, 0, 1), ZOmega(0, -1, 0, 0));
    if (failed(reduction(A, B, opG_r, OP_Z)))
      return failure();
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied Z");
    return success();
  }

  Real bias_A = bias(A);
  Real bias_B = bias(B);
  Real pair_bias_val = bias_B / bias_A;

  CUDAQ_SYNTH_LOG_TRACE("synth.upright",
                   "step_lemma: bias_A={}, bias_B={}, pair_bias={}",
                   bias_A, bias_B, pair_bias_val);

  // X operation: if A.bias * B.bias < 1
  if (bias_A * bias_B < 1) {
    static const GridOp OP_X(ZOmega(0, 1, 0, 0), ZOmega(0, 0, 0, 1));
    if (failed(reduction(A, B, opG_r, OP_X)))
      return failure();
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied X");
    return success();
  }

  // Both S and Sigma use log(ZSqrt2::lambda()) — cache as a function-local
  // static so it is computed only on the first call that reaches this point.
  static const Real lambda_real = to_real(ZSqrt2::lambda());

  // S operation: extreme bias values
  if (pair_bias_val > 33.971 || pair_bias_val < 0.029437) {
    static const GridOp OP_S(ZOmega(-1, 0, 1, 1), ZOmega(1, -1, 1, 0));
    Integer n = round_to_integer(log(pair_bias_val) / log(lambda_real) / 8);
    if (failed(reduction(A, B, opG_r, pow(OP_S, n))))
      return failure();
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied S^{}", n);
    return success();
  }

  Real skew = pair_skew(A, B);
  // Done check: computed lazily, only needed if Z/X/S didn't trigger.
  if (skew <= 15) {
    end = true;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright",
                     "step_lemma: pair_skew={} <= 15, done", skew);
    return success();
  }

  // Sigma operation: moderate bias values
  if (pair_bias_val > 5.8285 || pair_bias_val < 0.17157) {
    Integer n = round_to_integer(log(pair_bias_val) / log(lambda_real) / 4);
    shift_ellipses(A, B, n);
    if (n >= 0) {
      static const GridOp SIGMA_L_POS(ZOmega(-1, 0, 1, 1), ZOmega(0, 1, 0, 0));
      static const GridOp SIGMA_R_POS(ZOmega(0, 0, 0, 1), ZOmega(1, -1, 1, 0));
      opG_l = opG_l * pow(SIGMA_L_POS, n);
      opG_r = pow(SIGMA_R_POS, n) * opG_r;
    } else {
      static const GridOp SIGMA_L_NEG(ZOmega(-1, 0, 1, -1), ZOmega(0, 1, 0, 0));
      static const GridOp SIGMA_R_NEG(ZOmega(0, 0, 0, 1), ZOmega(1, 1, 1, 0));
      opG_l = opG_l * pow(SIGMA_L_NEG, -n);
      opG_r = pow(SIGMA_R_NEG, -n) * opG_r;
    }
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied Sigma^{}", n);
    return success();
  }

  // R operation: both biases in moderate range
  if (0.24410 <= bias_A && bias_A <= 4.0968 && 0.24410 <= bias_B &&
      bias_B <= 4.0968) {
    static const GridOp OP_R(ZOmega(0, 0, 1, 0), ZOmega(1, 0, 0, 0));
    if (failed(reduction(A, B, opG_r, OP_R)))
      return failure();
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied R");
    return success();
  }

  // K operation: A.b >= 0 and A.bias <= 1.6969
  if (A.b() >= 0 && bias_A <= 1.6969) {
    static const GridOp OP_K(ZOmega(-1, -1, 0, 0), ZOmega(0, -1, 1, 0));
    if (failed(reduction(A, B, opG_r, OP_K)))
      return failure();
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied K");
    return success();
  }

  // K_conj_sq2 operation: A.b >= 0 and B.bias <= 1.6969
  if (A.b() >= 0 && bias_B <= 1.6969) {
    static const GridOp OP_K_conj_sq2(ZOmega(1, -1, 0, 0),
                                      ZOmega(0, -1, -1, 0));
    if (failed(reduction(A, B, opG_r, OP_K_conj_sq2)))
      return failure();
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied K_conj");
    return success();
  }

  // A operation: A.b >= 0
  if (A.b() >= 0) {
    Integer n = std::max(Integer(1), floor_sqrt(std::min(bias_A, bias_B)) / 2);
    GridOp OP_A_n(ZOmega(0, 0, 0, 1), ZOmega(0, 1, 0, 2 * n));
    if (failed(reduction(A, B, opG_r, OP_A_n)))
      return failure();
    end = false;
    CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied A(n={})", n);
    return success();
  }

  // B operation: fallback case
  Integer n = std::max(Integer(1), floor_sqrt(std::min(bias_A, bias_B) / 2));
  GridOp OP_B_n(ZOmega(0, 0, 0, 1), ZOmega(n, 1, -n, 0));
  if (failed(reduction(A, B, opG_r, OP_B_n)))
    return failure();
  end = false;
  CUDAQ_SYNTH_LOG_TRACE("synth.upright", "step_lemma: applied B(n={})", n);
  return success();
}

FailureOr<UprightResult> to_upright(const Ellipse &setA, const Ellipse &setB) {
  CUDAQ_SYNTH_LOG_DEBUG("synth.upright", "to_upright: starting");
  Ellipse A = setA;
  if (failed(A.normalize()))
    return failure();
  Ellipse B = setB;
  if (failed(B.normalize()))
    return failure();

  GridOp opG_l = GridOp::identity();
  GridOp opG_r = GridOp::identity();

  i32 iterations = 0;
  bool done = false;
  while (!done) {
    if (failed(step_lemma(A, B, opG_l, opG_r, done)))
      return failure();
    ++iterations;
  }

  CUDAQ_SYNTH_LOG_DEBUG("synth.upright",
                   "to_upright: converged after {} step_lemma iterations",
                   iterations);

  // opG is built from normalized ellipses; apply it to the original (non-
  // normalized) copies to get the correct upright bboxes. shift_ellipses()
  // modifies A/B directly in ways not captured by the D' = I^T D I congruence
  // transform, so we cannot derive the original-upright bboxes from A/B alone.
  GridOp opG = opG_l * opG_r;
  Ellipse A_upright = setA;
  Ellipse B_upright = setB;
  if (failed(apply_grid_op(A_upright, B_upright, opG)))
    return failure();

  FailureOr<Rectangle> bboxA_or = bbox(A_upright);
  if (failed(bboxA_or))
    return failure();
  FailureOr<Rectangle> bboxB_or = bbox(B_upright);
  if (failed(bboxB_or))
    return failure();

  CUDAQ_SYNTH_LOG_TRACE("synth.upright",
                   "to_upright: opG={}, bboxA={}, bboxB={}, "
                   "uprightness_A={}, uprightness_B={}",
                   opG, *bboxA_or, *bboxB_or,
                   A_upright.area() / bboxA_or->area(),
                   B_upright.area() / bboxB_or->area());
  return UprightResult(opG, *bboxA_or, *bboxB_or);
}

} // namespace cudaq::synth
