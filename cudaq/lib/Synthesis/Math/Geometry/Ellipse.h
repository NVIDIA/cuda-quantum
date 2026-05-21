/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "llvm/Support/LogicalResult.h"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// TransformMode
//===----------------------------------------------------------------------===//

/// Selects the inversion strategy used by `Ellipse::transform_by_gridop_mat`.
enum class TransformMode {
  /// Hot path: the caller already has the inverse matrix entries. No
  /// division or GridOp::inv() call is performed.
  Direct,
  /// Algebraic path: derive the inverse from det(F) via the 2x2 closed
  /// form. Falls through to Fallback if det(F) is near-singular.
  Conjugate,
  /// Slow / safe path: compute the exact inverse via GridOp::inv(). Used
  /// when the forward matrix is singular or not analytically inverted.
  Fallback,
};

//===----------------------------------------------------------------------===//
// Ellipse
//===----------------------------------------------------------------------===//

/// The set E = { u in R^2 | (u - p)^T D (u - p) <= 1 } for a positive
/// definite 2x2 matrix D and centre p.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 5.15.
///
/// Storage. D is stored as three scalars _a, _b, _d representing the
/// symmetric matrix [[a, b], [b, d]]. Holding three scalars rather than a
/// 2x2 array saves one GMP allocation per instance and removes the need to
/// keep D[0][1] and D[1][0] in sync after every update.
///
/// Why this matters elsewhere:
///   1. The epsilon-region R_epsilon is enclosed by an ellipse (used by the
///      EpsilonRegion class for tighter intersection tests).
///   2. The to-upright algorithm (Theorem 5.16) iterates on a *pair* of
///      ellipses, using grid operators to drive them to M-upright form
///      (Definition 5.7, uprightness = area(E) / area(bbox(E))).
///   3. The Step Lemma state (D, Delta) (Appendix A, Definition A.1) is
///      itself just the D-matrices of a normalised ellipse pair.
///
/// Derived quantities used downstream:
///   det(D)       = a*d - b^2     (must be > 0 for positive definiteness).
///   area(E)      = pi / sqrt(det(D)).
///   uprightness  = (pi/4) * sqrt(det(D) / (a*d))      (equation (30)).
///   skew         = b^2                                 (equation (33)).
///   bias         = d / a                               (eigenvalue ratio).
///
/// `normalize()` scales D so det(D) = 1, matching the normal form (31) used
/// in the Step Lemma analysis.
///
/// Construction. Prefer the `create` factories over the raw constructors:
/// they return llvm::FailureOr<Ellipse> and enforce positive definiteness
/// without throwing.
class Ellipse : public ConvexSet {
private:
  // D = [[_a, _b], [_b, _d]]; the off-diagonal is stored once.
  Real _a, _b, _d;
  std::array<Real, 2> _p;

  // Tag for the unchecked internal constructor used by the factories.
  struct UncheckedTag {};

  Ellipse(UncheckedTag, const std::array<std::array<Real, 2>, 2> &D,
          const std::array<Real, 2> &p)
      : _a(D[0][0]), _b(D[0][1]), _d(D[1][1]), _p(p) {}

  Ellipse(UncheckedTag, const Real &a, const Real &b, const Real &d,
          const Real &px, const Real &py)
      : _a(a), _b(b), _d(d), _p{{px, py}} {}

  /// Evaluate a*x^2 + 2*b*x*y + d*y^2.
  Real eval_quadratic_form(const Real &x, const Real &y) const {
    return _a * x * x + 2 * _b * x * y + _d * y * y;
  }

  /// Apply the congruence D' = I^T D I and the forward map p' = F p, where
  /// I is the inverse of F. Both matrices are passed as separate scalars so
  /// callers can supply precomputed inverses without re-deriving them.
  void apply_inverse_transform(const Real &I00, const Real &I01,
                               const Real &I10, const Real &I11,
                               const Real &F00, const Real &F01,
                               const Real &F10, const Real &F11);

public:
  // -- Factories --

  /// Build from a full 2x2 D-matrix and centre p. Returns failure() if D is
  /// asymmetric or fails positive-definiteness (det <= 0, a <= 0, or
  /// d <= 0).
  static llvm::FailureOr<Ellipse>
  create(const std::array<std::array<Real, 2>, 2> &D,
         const std::array<Real, 2> &p) {
    if (D[0][1] != D[1][0])
      return llvm::failure();
    Real det = D[0][0] * D[1][1] - D[0][1] * D[1][0];
    if (det <= 0 || D[0][0] <= 0 || D[1][1] <= 0)
      return llvm::failure();
    return Ellipse(UncheckedTag{}, D, p);
  }

  /// Build from scalar D entries and centre (px, py). Returns failure() if
  /// the matrix is not positive definite.
  static llvm::FailureOr<Ellipse> create(const Real &a, const Real &b,
                                         const Real &d, const Real &px,
                                         const Real &py) {
    Real det = a * d - b * b;
    if (det <= 0 || a <= 0 || d <= 0)
      return llvm::failure();
    return Ellipse(UncheckedTag{}, a, b, d, px, py);
  }

  /// Asserting factory for call sites where the parameters are statically
  /// known to be a valid positive-definite matrix. The assertion is
  /// compiled out under NDEBUG.
  static Ellipse must_create(const Real &a, const Real &b, const Real &d,
                             const Real &px, const Real &py) {
    llvm::FailureOr<Ellipse> result = create(a, b, d, px, py);
    assert(llvm::succeeded(result) &&
           "Ellipse::must_create: invalid parameters");
    return *result;
  }

  // -- Accessors --

  /// D(0, 0): x^2 coefficient.
  const Real &a() const { return _a; }

  /// D(0, 1) == D(1, 0): the off-diagonal entry (the xy cross-term divided
  /// by 2 in the quadratic-form expansion).
  const Real &b() const { return _b; }

  /// D(1, 1): y^2 coefficient.
  const Real &d() const { return _d; }

  const std::array<Real, 2> &p() const { return _p; }
  const Real &px() const { return _p[0]; }
  const Real &py() const { return _p[1]; }

  void set_p(const Real &px, const Real &py) {
    _p[0] = px;
    _p[1] = py;
  }

  /// Scale D(0, 0) by `factor`.
  void scale_a(const Real &factor) { _a *= factor; }

  /// Scale D(1, 1) by `factor`.
  void scale_d(const Real &factor) { _d *= factor; }

  /// Negate the off-diagonal entries of D (b -> -b).
  void flip_b() { _b = -_b; }

  // -- Mutating geometric transforms --

  /// Apply the grid operator `g_local`: D <- I^T D I, p <- F p, where F is
  /// extracted from g_local via `to_real_mat` and I is its inverse.
  ///
  /// `mode` selects how the inverse is obtained -- see `TransformMode`.
  /// Returns failure() if the Fallback path is taken and GridOp::inv()
  /// fails.
  llvm::LogicalResult
  transform_by_gridop(const GridOp &g_local, TransformMode mode,
                      const Real &preinv00, const Real &preinv01,
                      const Real &preinv10, const Real &preinv11,
                      const Real &tol);

  /// Same as `transform_by_gridop` but with the forward-matrix entries
  /// supplied directly by the caller. `fallback_g` must be non-null when
  /// `mode == Fallback`, or when `mode == Conjugate` and the matrix turns
  /// out to be near-singular at runtime.
  llvm::LogicalResult
  transform_by_gridop_mat(TransformMode mode, const Real &F00, const Real &F01,
                          const Real &F10, const Real &F11,
                          const Real &preinv00, const Real &preinv01,
                          const Real &preinv10, const Real &preinv11,
                          const Real &tol, const GridOp *fallback_g = nullptr);

  // -- Derived state --

  /// Normalise D so det(D) = 1 (the normal form used by the Step Lemma).
  /// Scales all three D entries by 1/sqrt(det) -- one GMP division instead
  /// of three. Returns failure() if det(D) <= 0.
  llvm::LogicalResult normalize() {
    Real det_val = _a * _d - _b * _b;
    if (det_val <= 0)
      return llvm::failure();
    Real inv_sd = 1 / sqrt(det_val);
    _a *= inv_sd;
    _b *= inv_sd;
    _d *= inv_sd;
    return llvm::success();
  }

  /// Membership test. Currently unimplemented (the upstream code paths use
  /// the bounding-ellipse representation only for the upright preprocessing
  /// stage, where the actual ConvexSet::contains called is the
  /// EpsilonRegion / UnitDisk specialisation). Aborts if reached so
  /// accidental callers fail loudly.
  bool contains(const DOmega &v) const override {
    exit(1);
    return false;
  }

  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const override;

  /// Render at full MPFR precision (no lossy double conversion) for use
  /// inside LLVM_DEBUG / CUDAQ_CUDAQ_SYNTH_OPEN_SUB diagnostic streams.
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

//===----------------------------------------------------------------------===//
// Free functions: derived quantities and the bounding box
//===----------------------------------------------------------------------===//

/// det(D) = a*d - b^2. Returns failure() if the result is non-positive
/// (the positive-definiteness invariant should keep that from happening for
/// a correctly constructed or transformed Ellipse).
inline llvm::FailureOr<Real> det(const Ellipse &E) {
  Real result = E.a() * E.d() - E.b() * E.b();
  if (result <= 0)
    return llvm::failure();
  return result;
}

/// area(E) = pi / sqrt(det(D)). Propagates failure from `det`.
inline llvm::FailureOr<Real> area(const Ellipse &E) {
  llvm::FailureOr<Real> det_or = det(E);
  if (llvm::failed(det_or))
    return llvm::failure();
  return Real::pi() / sqrt(*det_or);
}

/// skew(E) = b^2. Part of the pair invariant Skew(D, Delta) = b^2 + beta^2
/// driven down by the Step Lemma; larger skew means less axis-alignment.
inline Real skew(const Ellipse &E) { return E.b() * E.b(); }

/// bias(E) = d / a. Eigenvalue-ratio proxy: bias = 1 with b = 0 means a
/// circle. The Step Lemma uses bias to choose the reduction operator.
inline Real bias(const Ellipse &E) { return E.d() / E.a(); }

/// Axis-aligned bounding box of E: [px - w, px + w] x [py - h, py + h] with
/// w = sqrt(d / det) and h = sqrt(a / det). Propagates failure from `det`.
inline llvm::FailureOr<Rectangle> bbox(const Ellipse &E) {
  llvm::FailureOr<Real> det_or = det(E);
  if (llvm::failed(det_or))
    return llvm::failure();
  const Real &det_val = *det_or;
  Real w = sqrt(E.d() / det_val);
  Real h = sqrt(E.a() / det_val);
  return Rectangle(E.px() - w, E.px() + w, E.py() - h, E.py() + h);
}

} // namespace cudaq::synth
