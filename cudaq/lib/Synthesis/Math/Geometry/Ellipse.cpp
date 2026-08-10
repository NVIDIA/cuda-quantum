/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Geometry/Ellipse.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"

#include <cassert>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Ellipse::transform_by_gridop / transform_by_gridop_mat
//===----------------------------------------------------------------------===//

llvm::LogicalResult
Ellipse::transform_by_gridop(const GridOp &g_local, TransformMode mode,
                             const Real &preinv00, const Real &preinv01,
                             const Real &preinv10, const Real &preinv11,
                             const Real &tol) {
  auto mat = to_real_mat(g_local);
  return transform_by_gridop_mat(mode, mat[0][0], mat[0][1], mat[1][0],
                                 mat[1][1], preinv00, preinv01, preinv10,
                                 preinv11, tol, &g_local);
}

llvm::LogicalResult Ellipse::transform_by_gridop_mat(
    TransformMode mode, const Real &F00, const Real &F01, const Real &F10,
    const Real &F11, const Real &preinv00, const Real &preinv01,
    const Real &preinv10, const Real &preinv11, const Real &tol,
    const GridOp *fallback_g) {
  // Common path for both Fallback mode and the recovery branch of Conjugate
  // mode: take the exact inverse via GridOp::inv() and feed it to
  // apply_inverse_transform.
  auto apply_from_gridop_inv = [&](const GridOp &g) -> llvm::LogicalResult {
    llvm::FailureOr<GridOp> inv_or = inv(g);
    if (llvm::failed(inv_or))
      return llvm::failure();
    auto invm = to_real_mat(*inv_or);
    apply_inverse_transform(invm[0][0], invm[0][1], invm[1][0], invm[1][1], F00,
                            F01, F10, F11);
    return llvm::success();
  };

  switch (mode) {
  case TransformMode::Fallback:
    assert(fallback_g && "transform_by_gridop_mat: Fallback mode requires "
                         "a non-null fallback GridOp");
    return apply_from_gridop_inv(*fallback_g);

  case TransformMode::Direct:
    // The caller already has the inverse entries; pass them through without
    // recomputation.
    apply_inverse_transform(preinv00, preinv01, preinv10, preinv11, F00, F01,
                            F10, F11);
    return llvm::success();

  case TransformMode::Conjugate:
    // Algebraic 2x2 inverse via det(F). Fall back to the exact GridOp::inv()
    // path only when the determinant is below `tol`, since for very small
    // det the closed-form entries lose precision catastrophically.
    Real det = F00 * F11 - F01 * F10;
    if (abs(det) < tol) {
      assert(fallback_g && "transform_by_gridop_mat: Conjugate mode with "
                           "singular matrix requires a non-null fallback "
                           "GridOp");
      return apply_from_gridop_inv(*fallback_g);
    }
    Real inv_det = 1 / det;
    apply_inverse_transform(F11 * inv_det, -(F01 * inv_det), -(F10 * inv_det),
                            F00 * inv_det, F00, F01, F10, F11);
    return llvm::success();
  }
  // Unreachable; suppress compiler warning.
  return llvm::failure();
}

//===----------------------------------------------------------------------===//
// Ellipse::intersect
//===----------------------------------------------------------------------===//

std::optional<std::pair<Real, Real>> Ellipse::intersect(const DOmega &u0,
                                                        const DOmega &v) const {
  // Currently unused: the upstream code paths route line/ray intersection
  // through the specialised UnitDisk and EpsilonRegion overrides, not the
  // generic Ellipse. A working implementation would solve the quadratic
  // (v - p)^T D (v - p) <= 1 along u0 + t * v, falling back to a linear
  // solve when the quadratic coefficient is near zero. Returning nullopt
  // signals "no intersection" -- safe but conservative.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Ellipse::apply_inverse_transform
//===----------------------------------------------------------------------===//

void Ellipse::apply_inverse_transform(const Real &I00, const Real &I01,
                                      const Real &I10, const Real &I11,
                                      const Real &F00, const Real &F01,
                                      const Real &F10, const Real &F11) {
  // D' = I^T D I, computed as the two-stage product D' = I^T * (D * I) and
  // exploiting D' symmetry to skip computing the (1, 0) entry.
  //
  // The const-ref bindings capture the old values *before* writing to the
  // members. Callers may pass aliased references (e.g. preinv* pointing
  // into the same Ellipse), so the writes below must not be visible to the
  // read side of this computation.
  const Real &old_a = _a;
  const Real &old_b = _b;
  const Real &old_d = _d;

  Real tmp00 = I00 * old_a + I10 * old_b;
  Real tmp01 = I00 * old_b + I10 * old_d;
  Real tmp10 = I01 * old_a + I11 * old_b;
  Real tmp11 = I01 * old_b + I11 * old_d;

  Real new_a = tmp00 * I00 + tmp01 * I10;
  Real new_b = tmp00 * I01 + tmp01 * I11;
  Real new_d = tmp10 * I01 + tmp11 * I11;

  // New centre p' = F p, snapshotted for the same aliasing reason.
  const Real &old_px = _p[0];
  const Real &old_py = _p[1];
  Real new_px = F00 * old_px + F01 * old_py;
  Real new_py = F10 * old_px + F11 * old_py;

  // Move into the members: mpfr_swap is O(1), the corresponding mpfr_set
  // would be O(precision/64). Only three D entries because the off-diagonal
  // is stored once.
  _a = std::move(new_a);
  _b = std::move(new_b);
  _d = std::move(new_d);
  _p[0] = std::move(new_px);
  _p[1] = std::move(new_py);
}

} // namespace cudaq::synth
