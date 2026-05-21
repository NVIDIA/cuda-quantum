/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Geometry/ToUpright.h"
#include "Support/StreamOps.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "cudaq-synth"

using namespace cudaq::synth;

namespace {

/// floor(sqrt(x)) as an arbitrary-precision Integer.
Integer floor_sqrt(const Real &x) { return floor_to_integer(sqrt(x)); }

} // namespace

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// apply_grid_op
//===----------------------------------------------------------------------===//

llvm::LogicalResult apply_grid_op(Ellipse &A, Ellipse &B, const GridOp &g) {
  // Use to_real_mat rather than to_mat: a grid operator is a real matrix,
  // and we never read the imaginary parts -- skip allocating them.
  auto g_mat = to_real_mat(g);
  Real M00 = std::move(g_mat[0][0]);
  Real M01 = std::move(g_mat[0][1]);
  Real M10 = std::move(g_mat[1][0]);
  Real M11 = std::move(g_mat[1][1]);

  static const Real tol(1e-30);
  Real det = M00 * M11 - M01 * M10;

  if (abs(det) < tol) {
    // Near-singular forward matrix: both A and B must take the Fallback
    // (exact GridOp::inv) path; the algebraic closed-form for the inverse
    // would lose all precision here.
    if (llvm::failed(A.transform_by_gridop_mat(TransformMode::Fallback, M00,
                                               M01, M10, M11, M00, M01, M10,
                                               M11, tol, &g)))
      return llvm::failure();
    GridOp g_conj = conj_sq2(g);
    if (llvm::failed(B.transform_by_gridop(g_conj, TransformMode::Fallback, M00,
                                           M01, M10, M11, tol)))
      return llvm::failure();
    return llvm::success();
  }

  // Non-singular path: invert F algebraically once, then reuse the inverse
  // entries for both A (Direct mode) and B (Conjugate mode). 1 division + 4
  // multiplications instead of two independent inversions.
  Real inv_det = 1 / det;
  Real inv00 = M11 * inv_det;
  Real inv01 = -(M01 * inv_det);
  Real inv10 = -(M10 * inv_det);
  Real inv11 = M00 * inv_det;

  if (llvm::failed(A.transform_by_gridop_mat(TransformMode::Direct, M00, M01,
                                             M10, M11, inv00, inv01, inv10,
                                             inv11, tol, &g)))
    return llvm::failure();

  // For B we need the conjugated forward matrix, but the *inverse* is the
  // same up to the conjugation that is folded into the Conjugate path.
  ZOmega u0c = g.u0().conj_sq2();
  ZOmega u1c = g.u1().conj_sq2();
  Real M00c, M10c, M01c, M11c;
  to_real_imag(u0c, M00c, M10c);
  to_real_imag(u1c, M01c, M11c);
  if (llvm::failed(B.transform_by_gridop_mat(TransformMode::Conjugate, M00c,
                                             M01c, M10c, M11c, inv00, inv01,
                                             inv10, inv11, tol, &g)))
    return llvm::failure();

  return llvm::success();
}

//===----------------------------------------------------------------------===//
// reduction
//===----------------------------------------------------------------------===//

llvm::LogicalResult reduction(Ellipse &A, Ellipse &B, GridOp &opG_r,
                              const GridOp &new_opG) {
  if (llvm::failed(apply_grid_op(A, B, new_opG)))
    return llvm::failure();
  opG_r = new_opG * opG_r;
  return llvm::success();
}

//===----------------------------------------------------------------------===//
// shift_ellipses
//===----------------------------------------------------------------------===//

void shift_ellipses(Ellipse &A, Ellipse &B, const Integer &n) {
  // lambda^n acts as a scaling on a single diagonal entry; lambda^-n on the
  // other. The B side picks up the conjugation, which is the b -> -b flip
  // for odd n (the sqrt(2)-conjugate of lambda^n is (-1)^n * lambda^-n).
  ZSqrt2 lambda_n = pow(ZSqrt2::lambda(), n);
  ZSqrt2 lambda_inv_n = *inv(lambda_n); // lambda is a unit so inv always ok
  Real lambda_n_real = to_real(lambda_n);
  Real lambda_inv_n_real = to_real(lambda_inv_n);

  A.scale_a(lambda_inv_n_real);
  A.scale_d(lambda_n_real);
  B.scale_a(lambda_n_real);
  B.scale_d(lambda_inv_n_real);

  if (n.is_odd())
    B.flip_b();
}

//===----------------------------------------------------------------------===//
// step_lemma
//===----------------------------------------------------------------------===//

llvm::LogicalResult step_lemma(Ellipse &A, Ellipse &B, GridOp &opG_l,
                               GridOp &opG_r, bool &end) {
  SYNTH_OPEN_SUB("step_lemma");
  LLVM_DEBUG(cudaq::synth::dbgs() << "A=" << A << '\n');
  LLVM_DEBUG(cudaq::synth::dbgs() << "B=" << B << '\n');

  // Z: ensure beta >= 0 by negating the anti-diagonal.
  if (B.b() < 0) {
    static const GridOp OP_Z(ZOmega(0, 0, 0, 1), ZOmega(0, -1, 0, 0));
    if (llvm::failed(reduction(A, B, opG_r, OP_Z))) {
      SYNTH_CLOSE_FAILURE("reduction failed");
      return llvm::failure();
    }
    end = false;
    SYNTH_ACTION("Apply") << "Z\n";
    SYNTH_CLOSE_SUCCESS("applied Z");
    return llvm::success();
  }

  Real bias_A = bias(A);
  Real bias_B = bias(B);
  Real pair_bias_val = bias_B / bias_A;

  LLVM_DEBUG(cudaq::synth::dbgs()
             << "bias_A=" << bias_A << ", bias_B=" << bias_B
             << ", pair_bias=" << pair_bias_val << '\n');

  // X: ensure z + zeta >= 0 by swapping diagonals.
  if (bias_A * bias_B < 1) {
    static const GridOp OP_X(ZOmega(0, 1, 0, 0), ZOmega(0, 0, 0, 1));
    if (llvm::failed(reduction(A, B, opG_r, OP_X))) {
      SYNTH_CLOSE_FAILURE("reduction failed");
      return llvm::failure();
    }
    end = false;
    SYNTH_ACTION("Apply") << "X\n";
    SYNTH_CLOSE_SUCCESS("applied X");
    return llvm::success();
  }

  // Cached log(lambda) for the Sigma / S exponent computations. Function-
  // local static so it is computed at most once per process and only when
  // execution actually reaches this point.
  static const Real lambda_real = to_real(ZSqrt2::lambda());

  // S: extreme pair-bias values. The threshold constants are the
  // floating-point evaluations of the symbolic bounds from sec. A.1.
  if (pair_bias_val > 33.971 || pair_bias_val < 0.029437) {
    static const GridOp OP_S(ZOmega(-1, 0, 1, 1), ZOmega(1, -1, 1, 0));
    Integer n = round_to_integer(log(pair_bias_val) / log(lambda_real) / 8);
    if (llvm::failed(reduction(A, B, opG_r, pow(OP_S, n)))) {
      SYNTH_CLOSE_FAILURE("reduction failed");
      return llvm::failure();
    }
    end = false;
    SYNTH_ACTION("Apply") << "S^" << n << '\n';
    SYNTH_CLOSE_SUCCESS("applied S^" + n.to_string());
    return llvm::success();
  }

  // Done check is computed lazily -- only after Z/X/S have all missed, so
  // the typical iteration pays for one comparison rather than a full skew.
  Real skew = pair_skew(A, B);
  if (skew <= 15) {
    end = true;
    LLVM_DEBUG(cudaq::synth::dbgs()
               << "pair_skew=" << skew << " <= 15, done\n");
    SYNTH_CLOSE_SUCCESS("done");
    return llvm::success();
  }

  // Sigma: moderate bias values. The shift is built into both halves of the
  // running product (opG_l and opG_r) so the final accumulated operator
  // remains a valid special grid operator.
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
    SYNTH_ACTION("Apply") << "Sigma^" << n << '\n';
    SYNTH_CLOSE_SUCCESS("applied Sigma^" + n.to_string());
    return llvm::success();
  }

  // R: both per-side biases inside the moderate range; the rotation-like
  // operator from Figure 6 of the paper applies.
  if (0.24410 <= bias_A && bias_A <= 4.0968 && 0.24410 <= bias_B &&
      bias_B <= 4.0968) {
    static const GridOp OP_R(ZOmega(0, 0, 1, 0), ZOmega(1, 0, 0, 0));
    if (llvm::failed(reduction(A, B, opG_r, OP_R))) {
      SYNTH_CLOSE_FAILURE("reduction failed");
      return llvm::failure();
    }
    end = false;
    SYNTH_ACTION("Apply") << "R\n";
    SYNTH_CLOSE_SUCCESS("applied R");
    return llvm::success();
  }

  // K: A.b >= 0 and the A side is the wider one (bias(A) <= 1.6969).
  if (A.b() >= 0 && bias_A <= 1.6969) {
    static const GridOp OP_K(ZOmega(-1, -1, 0, 0), ZOmega(0, -1, 1, 0));
    if (llvm::failed(reduction(A, B, opG_r, OP_K))) {
      SYNTH_CLOSE_FAILURE("reduction failed");
      return llvm::failure();
    }
    end = false;
    SYNTH_ACTION("Apply") << "K\n";
    SYNTH_CLOSE_SUCCESS("applied K");
    return llvm::success();
  }

  // K_conj_sq2: the swapped-roles case where B is now the wider one.
  if (A.b() >= 0 && bias_B <= 1.6969) {
    static const GridOp OP_K_conj_sq2(ZOmega(1, -1, 0, 0),
                                      ZOmega(0, -1, -1, 0));
    if (llvm::failed(reduction(A, B, opG_r, OP_K_conj_sq2))) {
      SYNTH_CLOSE_FAILURE("reduction failed");
      return llvm::failure();
    }
    end = false;
    SYNTH_ACTION("Apply") << "K_conj\n";
    SYNTH_CLOSE_SUCCESS("applied K_conj");
    return llvm::success();
  }

  // A: parameterised shear (Lemma A.17). The exponent n is the floor of
  // sqrt of the smaller bias, clamped to 1 so the operator is non-trivial.
  if (A.b() >= 0) {
    Integer n = std::max(Integer(1), floor_sqrt(std::min(bias_A, bias_B)) / 2);
    GridOp OP_A_n(ZOmega(0, 0, 0, 1), ZOmega(0, 1, 0, 2 * n));
    if (llvm::failed(reduction(A, B, opG_r, OP_A_n))) {
      SYNTH_CLOSE_FAILURE("reduction failed");
      return llvm::failure();
    }
    end = false;
    SYNTH_ACTION("Apply") << "A(n=" << n << ")\n";
    SYNTH_CLOSE_SUCCESS("applied A(n=" + n.to_string() + ")");
    return llvm::success();
  }

  // B: catch-all (Lemma A.19) for the b < 0 <= beta case. Same n clamp as
  // the A operator above.
  Integer n = std::max(Integer(1), floor_sqrt(std::min(bias_A, bias_B) / 2));
  GridOp OP_B_n(ZOmega(0, 0, 0, 1), ZOmega(n, 1, -n, 0));
  if (llvm::failed(reduction(A, B, opG_r, OP_B_n))) {
    SYNTH_CLOSE_FAILURE("reduction failed");
    return llvm::failure();
  }
  end = false;
  SYNTH_ACTION("Apply") << "B(n=" << n << ")\n";
  SYNTH_CLOSE_SUCCESS("applied B(n=" + n.to_string() + ")");
  return llvm::success();
}

//===----------------------------------------------------------------------===//
// to_upright
//===----------------------------------------------------------------------===//

llvm::FailureOr<UprightResult> to_upright(const Ellipse &setA,
                                          const Ellipse &setB) {
  SYNTH_OPEN_SUB("to_upright");
  Ellipse A = setA;
  if (llvm::failed(A.normalize())) {
    SYNTH_CLOSE_FAILURE("normalize(A) failed");
    return llvm::failure();
  }
  Ellipse B = setB;
  if (llvm::failed(B.normalize())) {
    SYNTH_CLOSE_FAILURE("normalize(B) failed");
    return llvm::failure();
  }

  GridOp opG_l = GridOp::identity();
  GridOp opG_r = GridOp::identity();

  [[maybe_unused]] int32_t iterations = 0;
  bool done = false;
  while (!done) {
    if (llvm::failed(step_lemma(A, B, opG_l, opG_r, done))) {
      SYNTH_CLOSE_FAILURE("step_lemma failed at iteration " +
                          std::to_string(iterations));
      return llvm::failure();
    }
    ++iterations;
  }

  // opG was built against the *normalised* ellipse pair; apply it to fresh
  // copies of the *original* (un-normalised) inputs to compute the upright
  // bounding boxes. Going back through A / B would not work because
  // shift_ellipses() mutates them in ways that are not captured by the
  // D' = I^T D I congruence transform.
  GridOp opG = opG_l * opG_r;
  Ellipse A_upright = setA;
  Ellipse B_upright = setB;
  if (llvm::failed(apply_grid_op(A_upright, B_upright, opG))) {
    SYNTH_CLOSE_FAILURE("apply_grid_op (upright recompute) failed");
    return llvm::failure();
  }

  llvm::FailureOr<Rectangle> bboxA_or = bbox(A_upright);
  if (llvm::failed(bboxA_or)) {
    SYNTH_CLOSE_FAILURE("bbox(A_upright) failed");
    return llvm::failure();
  }
  llvm::FailureOr<Rectangle> bboxB_or = bbox(B_upright);
  if (llvm::failed(bboxB_or)) {
    SYNTH_CLOSE_FAILURE("bbox(B_upright) failed");
    return llvm::failure();
  }

  LLVM_DEBUG(cudaq::synth::dbgs()
             << "opG=" << opG << ", bboxA=" << *bboxA_or
             << ", bboxB=" << *bboxB_or
             << ", uprightness_A=" << (A_upright.area() / bboxA_or->area())
             << ", uprightness_B=" << (B_upright.area() / bboxB_or->area())
             << '\n');
  SYNTH_CLOSE_SUCCESS("converged after " + std::to_string(iterations) +
                      " step_lemma iterations");
  return UprightResult(opG, *bboxA_or, *bboxB_or);
}

} // namespace cudaq::synth
