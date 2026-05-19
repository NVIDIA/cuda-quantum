/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/ConvexSet.h"
#include "Math/Geometry/Ellipse.h"
#include "Support/LogMacros.h"
#include "cudaq/Synthesis/Math/Real.h"

#include <array>
#include <optional>
#include <utility>

namespace cudaq::synth {

/// UnitDisk: the closed unit disc D̄ = { u ∈ ℂ : |u| ≤ 1 }.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §7.2.
///
/// In Algorithm 7.6, the grid problem is formulated for pairs (u, u●) with
/// the constraint u†u + t†t = 1. Lemma 7.5 implies that a necessary condition
/// is |u| ≤ 1 and |u●| ≤ 1, i.e. both u and its √2-conjugate u● must lie in
/// the closed unit disc. This is where the set B = D̄ in the two-dimensional
/// grid problem comes from.
///
/// Two distinct views of the same object are available:
///   - as_ellipse(): the Ellipse representation (identity quadratic form,
///     center at the origin), used by to_upright() for the `preprocessing`
///     step.
///   - inside() / intersect(): `specialised` implementations that exploit the
///     circular symmetry directly, avoiding the general Ellipse machinery.
class UnitDisk : public ConvexSet {
public:
  UnitDisk() = default;

  /// The unit disc as an Ellipse object (for use with to_upright).
  ///
  /// The unit disc x² + y² ≤ 1 corresponds to the Ellipse with quadratic
  /// form matrix equal to the 2×2 identity (A = D = 1, B = 0) and center
  /// at the origin (`px` = `py` = 0).
  ///
  /// Implemented as a function-local static to avoid static initialization
  /// order issues (same pattern as ZSqrt2::lambda() and GridOp::identity()).
  static const Ellipse &as_ellipse() {
    static const Ellipse value = Ellipse::must_create(1.0, 0.0, 1.0, 0.0, 0.0);
    return value;
  }

  /// Exact membership test: u = (x, y) ∈ D̄ ⟺ x² + y² ≤ 1.
  ///
  /// A small tolerance is added to guard against rounding in Real arithmetic
  /// at the boundary. The threshold 1 + 1e-30 is cached as a static to
  /// avoid constructing a temporary Real on every call (which would incur a
  /// GMP allocation in the hot TDGP inner loop).
  bool contains(const DOmega &u) const override {
    return DSqrt2::from_domega(u.conj() * u) <= DSqrt2{1};
  }

  /// Intersect a ray u(t) = u0 + t·v with the unit disc.
  ///
  /// The condition ‖u(t)‖² ≤ 1 becomes a quadratic in t:
  ///   a t² + b t + c ≤ 0
  /// with
  ///   a = ‖v‖²  =  v·v,
  ///   b = 2(u0·v),
  ///   c = ‖u0‖² − 1.
  ///
  /// Returns the parameter interval [t₀, t₁] where the ray lies inside the
  /// disc, or std::nullopt if the ray misses it entirely.
  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const override {
    DOmega a = v.conj() * v;
    DOmega b = DOmega::from_int(2) * (u0.conj() * v);
    DOmega c = u0.conj() * u0 - DOmega::from_dsqrt2(DSqrt2{1});
    return solve_quadratic(a.real(), b.real(), c.real());
  }
};

} // namespace cudaq::synth
