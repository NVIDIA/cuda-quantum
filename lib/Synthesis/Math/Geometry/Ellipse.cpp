/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Geometry/Ellipse.h"
#include "Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Real.h"

#include <cassert>

namespace cudaq::synth {

LogicalResult
Ellipse::transform_by_gridop(const GridOp &g_local, TransformMode mode,
                             const Real &preinv00, const Real &preinv01,
                             const Real &preinv10, const Real &preinv11,
                             const Real &tol) {
  auto mat = to_real_mat(g_local);
  return transform_by_gridop_mat(mode, mat[0][0], mat[0][1], mat[1][0],
                                 mat[1][1], preinv00, preinv01, preinv10,
                                 preinv11, tol, &g_local);
}

LogicalResult Ellipse::transform_by_gridop_mat(
    TransformMode mode, const Real &F00, const Real &F01, const Real &F10,
    const Real &F11, const Real &preinv00, const Real &preinv01,
    const Real &preinv10, const Real &preinv11, const Real &tol,
    const GridOp *fallback_g) {
  // Computes the exact inverse via GridOp::inv() and applies the transform.
  // Used for both the Fallback mode and as a recovery path when the forward
  // matrix is near-singular in Conjugate mode.
  auto apply_from_gridop_inv = [&](const GridOp &g) -> LogicalResult {
    FailureOr<GridOp> inv_or = inv(g);
    if (failed(inv_or))
      return failure();
    auto invm = to_real_mat(*inv_or);
    apply_inverse_transform(invm[0][0], invm[0][1], invm[1][0], invm[1][1], F00,
                            F01, F10, F11);
    return success();
  };

  switch (mode) {
  case TransformMode::Fallback:
    assert(fallback_g && "transform_by_gridop_mat: Fallback mode requires "
                         "a non-null fallback GridOp");
    return apply_from_gridop_inv(*fallback_g);

  case TransformMode::Direct:
    // Hot path: inverse is precomputed — pass references directly, no copies.
    apply_inverse_transform(preinv00, preinv01, preinv10, preinv11, F00, F01,
                            F10, F11);
    return success();

  case TransformMode::Conjugate:
    // Compute I = F⁻¹ algebraically from the 2×2 determinant formula.
    Real det = F00 * F11 - F01 * F10;
    if (abs(det) < tol) {
      // Near-singular: fall back to the exact GridOp::inv() path.
      assert(fallback_g && "transform_by_gridop_mat: Conjugate mode with "
                           "singular matrix requires a non-null fallback "
                           "GridOp");
      return apply_from_gridop_inv(*fallback_g);
    }
    Real inv_det = 1 / det;
    apply_inverse_transform(F11 * inv_det, -(F01 * inv_det), -(F10 * inv_det),
                            F00 * inv_det, F00, F01, F10, F11);
    return success();
  }
  // Unreachable; suppress compiler warning.
  return failure();
}

std::optional<std::pair<Real, Real>> Ellipse::intersect(const DOmega &u0,
                                                        const DOmega &v) const {
  // static const Real tolerance(1e-30);
  // Real rel_x0 = u0[0] - `px`();
  // Real rel_y0 = u0[1] - `py`();
  // const Real &dx = v[0];
  // const Real &dy = v[1];

  // Real qa = eval_quadratic_form(dx, dy);
  // Real qb = 2 * (a() * rel_x0 * dx + b() * (rel_x0 * dy + rel_y0 * dx) +
  // d() * rel_y0 * dy);
  // Real qc = eval_quadratic_form(rel_x0, rel_y0) - 1;

  //// Degenerate: v is nearly in the null space of D (qa ≈ 0 → linear
  /// equation).
  // if (abs(qa) < tolerance) {
  // if (abs(qb) < tolerance)
  // return std::nullopt;
  //// Single intersection point: move to avoid an extra GMP copy.
  // Real t = -qc / qb;
  // auto t2 = t;
  // return std::make_pair(std::move(t), std::move(t2));
  //}

  // return solve_quadratic(qa, qb, qc);
}

void Ellipse::apply_inverse_transform(const Real &I00, const Real &I01,
                                      const Real &I10, const Real &I11,
                                      const Real &F00, const Real &F01,
                                      const Real &F10, const Real &F11) {
  // Compute D' = Iᵀ D I via the two-stage product:
  //   T = D · I,   D' = Iᵀ · T  (exploiting the symmetry of D').
  //
  // References to old values must be captured before any write to _a/_b/_d
  // since the callers may pass aliased references (e.g. `preinv`* pointing into
  // the same Ellipse).  The old_* bindings below snapshot the values.
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

  // Compute the new center p' = F p before writing to _p, since F entries
  // may alias _p in pathological call sites.
  const Real &old_px = _p[0];
  const Real &old_py = _p[1];
  Real new_px = F00 * old_px + F01 * old_py;
  Real new_py = F10 * old_px + F11 * old_py;

  // Move computed values into members: mpfr_swap is O(1) vs mpfr_set O(p/64).
  // _b covers both off-diagonal entries; no separate D[1][0] synchronisation
  // is needed because we store only three scalars.
  _a = std::move(new_a);
  _b = std::move(new_b);
  _d = std::move(new_d);
  _p[0] = std::move(new_px);
  _p[1] = std::move(new_py);
}

} // namespace cudaq::synth
