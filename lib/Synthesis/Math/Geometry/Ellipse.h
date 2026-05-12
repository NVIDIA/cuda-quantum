/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/ConvexSet.h"
#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Rectangle.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Support/Result.h"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace cudaq::synth {

/// Selects the path taken by Ellipse::transform_by_gridop_mat.
enum class TransformMode {
  /// Hot path: the caller has already precomputed the inverse entries
  /// (preinv*).  No division or GridOp::inv() call is performed.
  Direct,
  /// Conjugate path: the caller supplies only the forward matrix entries
  /// (F*).  The inverse is derived algebraically from det(F).  Falls
  /// through to the Fallback path if det(F) is near-singular.
  Conjugate,
  /// Fallback path: the exact inverse is obtained via GridOp::inv().
  /// Used when the forward matrix is singular or not analytically inverted.
  Fallback,
};

/// Ellipse: The set E = { u ∈ R² | (u-p)ᵀ D (u-p) ≤ 1 } where D is a
/// positive definite 2×2 matrix and p is the center.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 5.15.
///
/// D is stored as three independent scalars _a, _b, _d representing the
/// symmetric matrix [[a, b], [b, d]].  Storing only three scalars (rather
/// than a full 2×2 array) saves one GMP allocation per instance and
/// eliminates the need to synchronise D[0][1] with D[1][0] on every update.
///
/// Ellipses are central to the algorithm because:
///
/// (1) The ε-region R_ε (Section 7.1) is bounded by an ellipse (used
///     in the EpsilonRegion class for tighter intersection tests).
///
/// (2) The "to-upright" algorithm (Theorem 5.16) operates on pairs of
///     ellipses, using grid operators to make them "M-upright" (Definition
///     5.7, uprightness = area(E) / area(BBox(E))).
///
/// (3) The state (D, Δ) in the Step Lemma (Appendix A, Definition A.1)
///     consists of the D-matrices of a normalized ellipse pair.
///
/// Key derived quantities:
/// - det(D) = ad - b²  (must be > 0 for positive definiteness)
/// - area(E) = π / √det(D)
/// - uprightness = (π/4) · √(det(D) / (a·d))  (equation (30))
/// - skew = b²  (from equation (33): b² = π²/(16M²) - 1)
/// - bias = d/a  (related to the ratio of eigenvalues)
///
/// The normalize() method scales D so that det(D) = 1, matching the
/// normal form (31) used in the Step Lemma analysis.
///
/// Construction: use the static factory methods (create) rather than the
/// constructors directly. Factories return FailureOr<Ellipse> and enforce
/// the positive-definiteness invariant without exceptions.
class Ellipse : public ConvexSet {
private:
  // D-matrix stored as three scalars: D = [[_a, _b], [_b, _d]].
  // Symmetry is an invariant: D[0][1] == D[1][0] == _b always.
  Real _a, _b, _d;
  std::array<Real, 2> _p;

  // Private tag type for unchecked internal construction (known-valid inputs).
  struct UncheckedTag {};

  /// Unchecked construction from a full 2×2 D-matrix array (extracts a, b, d).
  Ellipse(UncheckedTag, const std::array<std::array<Real, 2>, 2> &D,
          const std::array<Real, 2> &p)
      : _a(D[0][0]), _b(D[0][1]), _d(D[1][1]), _p(p) {}

  /// Unchecked construction from scalar coefficients a, b, d and center.
  Ellipse(UncheckedTag, const Real &a, const Real &b, const Real &d,
          const Real &px, const Real &py)
      : _a(a), _b(b), _d(d), _p{{px, py}} {}

  /// Evaluates the quadratic form a·x² + 2b·x·y + d·y².
  Real eval_quadratic_form(const Real &x, const Real &y) const {
    return _a * x * x + 2 * _b * x * y + _d * y * y;
  }

  /// Applies the congruence transformation D' = Iᵀ D I to the D-matrix and
  /// the forward transformation p' = F p to the center.
  ///
  /// I is the inverse of the forward matrix F.  Both are passed as
  /// precomputed scalars to avoid any recomputation.
  void apply_inverse_transform(const Real &I00, const Real &I01,
                               const Real &I10, const Real &I11,
                               const Real &F00, const Real &F01,
                               const Real &F10, const Real &F11);

public:
  // -------------------------------------------------------------------------
  // Factory constructors (preferred public interface)
  // -------------------------------------------------------------------------

  /// Construct from a full 2×2 positive-definite matrix D and center p.
  ///
  /// Returns failure() if:
  ///   - D is not symmetric (D[0][1] ≠ D[1][0]), or
  ///   - D is not positive definite (det(D) ≤ 0, a ≤ 0, or d ≤ 0).
  static FailureOr<Ellipse> create(const std::array<std::array<Real, 2>, 2> &D,
                                   const std::array<Real, 2> &p) {
    if (D[0][1] != D[1][0])
      return failure();
    Real det = D[0][0] * D[1][1] - D[0][1] * D[1][0];
    if (det <= 0 || D[0][0] <= 0 || D[1][1] <= 0)
      return failure();
    return Ellipse(UncheckedTag{}, D, p);
  }

  /// Construct from scalar coefficients a, b, d (matrix [[a,b],[b,d]]) and
  /// center (px, py).  Returns failure() if the matrix is not positive
  /// definite (det(D) ≤ 0, a ≤ 0, or d ≤ 0).
  static FailureOr<Ellipse> create(const Real &a, const Real &b, const Real &d,
                                   const Real &px, const Real &py) {
    Real det = a * d - b * b;
    if (det <= 0 || a <= 0 || d <= 0)
      return failure();
    return Ellipse(UncheckedTag{}, a, b, d, px, py);
  }

  /// Convenience factory that asserts success — for call sites where the
  /// parameters are statically known to form a valid positive-definite matrix.
  ///
  /// Note: the assert is compiled out in release builds (NDEBUG).  Only use
  /// this overload when the parameters are verifiably correct.
  static Ellipse must_create(const Real &a, const Real &b, const Real &d,
                             const Real &px, const Real &py) {
    auto result = create(a, b, d, px, py);
    assert(succeeded(result) && "Ellipse::must_create: invalid parameters");
    return *result;
  }

  // -------------------------------------------------------------------------
  // Accessors
  // -------------------------------------------------------------------------

  /// Returns the (0,0) entry of D: the x² coefficient.
  const Real &a() const { return _a; }

  /// Returns the (0,1) == (1,0) entry of D: the xy cross-term coefficient.
  const Real &b() const { return _b; }

  /// Returns the (1,1) entry of D: the y² coefficient.
  const Real &d() const { return _d; }

  /// Returns the center of the ellipse.
  const std::array<Real, 2> &p() const { return _p; }

  /// Returns the x-coordinate of the center.
  const Real &px() const { return _p[0]; }

  /// Returns the y-coordinate of the center.
  const Real &py() const { return _p[1]; }

  /// Sets the center to (px, py).
  void set_p(const Real &px, const Real &py) {
    _p[0] = px;
    _p[1] = py;
  }

  /// Scales the (0,0) entry of D by factor.
  void scale_a(const Real &factor) { _a *= factor; }

  /// Scales the (1,1) entry of D by factor.
  void scale_d(const Real &factor) { _d *= factor; }

  /// Negates the off-diagonal entries of D (i.e. b ← −b).
  void flip_b() { _b = -_b; }

  // -------------------------------------------------------------------------
  // Geometric transform (mutating, returns LogicalResult)
  // -------------------------------------------------------------------------

  /// Applies the grid operator g_local to the ellipse (D ← Iᵀ D I, p ← F p).
  ///
  /// mode controls how the inverse is computed:
  ///   - Direct:    use the precomputed inverse entries (preinv*).
  ///   - Conjugate: derive the inverse from det(F); fall back to Fallback if
  ///                det(F) < tol.
  ///   - Fallback:  compute the inverse via GridOp::inv().
  ///
  /// The forward matrix F is extracted from g_local via to_real_mat.
  /// Returns failure() if mode == Fallback and GridOp::inv() fails.
  LogicalResult transform_by_gridop(const GridOp &g_local, TransformMode mode,
                                    const Real &preinv00, const Real &preinv01,
                                    const Real &preinv10, const Real &preinv11,
                                    const Real &tol);

  /// Applies a precomputed forward matrix F (given as scalars) to the ellipse.
  ///
  /// mode controls how the inverse is computed (see transform_by_gridop).
  /// fallback_g must be non-null when mode == Fallback or when mode ==
  /// Conjugate and the matrix is found to be near-singular.
  /// Returns failure() if GridOp::inv() fails in the Fallback path.
  LogicalResult transform_by_gridop_mat(TransformMode mode, const Real &F00,
                                        const Real &F01, const Real &F10,
                                        const Real &F11, const Real &preinv00,
                                        const Real &preinv01,
                                        const Real &preinv10,
                                        const Real &preinv11, const Real &tol,
                                        const GridOp *fallback_g = nullptr);

  // -------------------------------------------------------------------------
  // Derived quantities (see free functions below)
  // -------------------------------------------------------------------------

  /// Normalize the D-matrix so that det(D) = 1.
  ///
  /// Scales all three D entries by 1/√det(D).  Used to produce the normal
  /// form (31) required by the Step Lemma analysis.
  /// Returns failure() if the determinant invariant is violated (det ≤ 0).
  LogicalResult normalize() {
    Real det_val = _a * _d - _b * _b;
    if (det_val <= 0)
      return failure();
    // Multiply by 1/√det rather than dividing four times: one GMP division
    // instead of three.
    Real inv_sd = 1 / sqrt(det_val);
    _a *= inv_sd;
    _b *= inv_sd;
    _d *= inv_sd;
    return success();
  }

  bool contains(const DOmega &v) const override {
    exit(1);
    return false;
    //Real x = v[0] - px();
    //Real y = v[1] - py();
    //Real tmp = eval_quadratic_form(x, y);
    //return tmp <= 1.0;
  }

  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0,
            const DOmega &v) const override;

  /// Returns a compact human-readable string of the ellipse parameters.
  ///
  /// Intended for debug/trace logging only. Uses mpfr_snprintf to format
  /// MPFR values at full precision — no lossy double conversion.
  /// Pass as a macro argument so Quill's level check keeps this lazy.
  std::string to_string() const {
    char buf[2048];
    mpfr_snprintf(
        buf, sizeof(buf),
        "D=(a=%.40Rg,b=%.40Rg,d=%.40Rg),center=(%.40Rg,%.40Rg),area=%.40Rg",
        _a.get_mpfr(), _b.get_mpfr(), _d.get_mpfr(), _p[0].get_mpfr(),
        _p[1].get_mpfr(), area().get_mpfr());
    return buf;
  }

  Real area() const {
    Real det_r = _a * _d - _b * _b;
    return Real::pi() / sqrt(det_r);
  }
};

// ---------------------------------------------------------------------------
// Free functions: derived quantities that only need public accessors
// ---------------------------------------------------------------------------

/// Determinant of the D-matrix: det(D) = a·d − b².
///
/// Returns failure() if the result is ≤ 0 (invariant violation — should not
/// happen for a correctly constructed or transformed Ellipse).
inline FailureOr<Real> det(const Ellipse &E) {
  Real result = E.a() * E.d() - E.b() * E.b();
  if (result <= 0)
    return failure();
  return result;
}

/// Area of the ellipse: π / √det(D).
///
/// Propagates failure from det().
inline FailureOr<Real> area(const Ellipse &E) {
  auto det_or = det(E);
  if (failed(det_or))
    return failure();
  return Real::pi() / sqrt(*det_or);
}

/// Skew of the ellipse: b² (the squared off-diagonal entry of D).
///
/// Used by the Step Lemma as part of the pair invariant Skew(D,Δ) = b² + β².
/// Larger skew means the ellipse is less axis-aligned.
inline Real skew(const Ellipse &E) { return E.b() * E.b(); }

/// Bias of the ellipse: d/a (ratio of the diagonal entries of D).
///
/// Measures the asymmetry of the eigenvalues.  bias = 1 means the ellipse is
/// a circle (when b = 0).  Used by the Step Lemma to select the reduction
/// operator.
inline Real bias(const Ellipse &E) { return E.d() / E.a(); }

/// Axis-aligned bounding box of the ellipse: [px±w] × [py±h].
///
/// Derived from the formula w = √(d/det), h = √(a/det).
/// Returns failure() if the determinant invariant is violated.
inline FailureOr<Rectangle> bbox(const Ellipse &E) {
  FailureOr<Real> det_or = det(E);
  if (failed(det_or))
    return failure();
  const Real &det_val = *det_or;
  Real w = sqrt(E.d() / det_val);
  Real h = sqrt(E.a() / det_val);
  return Rectangle(E.px() - w, E.px() + w, E.py() - h, E.py() + h);
}

} // namespace cudaq::synth
