/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/ConvexSet.h"
#include "Math/Geometry/Ellipse.h"
#include "cudaq/Synthesis/Math/Real.h"

#include <array>
#include <optional>
#include <utility>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// UnitDisk
//===----------------------------------------------------------------------===//

/// The closed unit disk { u in C : |u| <= 1 }.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, sec. 7.2. Algorithm 7.6 looks
/// for pairs (u, conj_sq2(u)) satisfying conj(u) * u + conj(t) * t = 1; Lemma
/// 7.5 forces |u| <= 1 and |conj_sq2(u)| <= 1, so the closed unit disk is the
/// `B` argument of the underlying 2D grid problem.
///
/// Two views of the same object are exposed: `as_ellipse()` produces the
/// generic Ellipse representation (used by to_upright()), while `contains()`
/// and `intersect()` are specialized implementations that use the disk's
/// circular symmetry directly.
class UnitDisk : public ConvexSet {
public:
  UnitDisk() = default;

  /// Unit disk in Ellipse form: identity quadratic form, center at the
  /// origin. Function-local static avoids the static-initialization-order
  /// pitfall (same pattern as ZSqrt2::lambda() and GridOp::identity()).
  static const Ellipse &as_ellipse() {
    static const Ellipse value = Ellipse::must_create(1.0, 0.0, 1.0, 0.0, 0.0);
    return value;
  }

  /// Exact membership: u is in the disk iff |u|^2 <= 1, evaluated as
  /// conj_sq2-compatible DSqrt2 arithmetic so the comparison is exact rather
  /// than MPFR-rounded.
  bool contains(const DOmega &u) const override {
    return DSqrt2::from_domega(u.conj() * u) <= DSqrt2{1};
  }

  /// Intersect the ray u(t) = u0 + t*v with the disk.
  ///
  /// |u(t)|^2 <= 1 becomes a*t^2 + b*t + c <= 0 with
  ///     a = conj(v) * v
  ///     b = 2 * (conj(u0) * v)
  ///     c = conj(u0) * u0 - 1
  /// Returns the parameter interval [t_0, t_1] inside the disk, or
  /// std::nullopt if the ray misses.
  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const override {
    DOmega a = v.conj() * v;
    DOmega b = DOmega::from_int(2) * (u0.conj() * v);
    DOmega c = u0.conj() * u0 - DOmega::from_dsqrt2(DSqrt2{1});
    return solve_quadratic(a.real(), b.real(), c.real());
  }
};

} // namespace cudaq::synth
