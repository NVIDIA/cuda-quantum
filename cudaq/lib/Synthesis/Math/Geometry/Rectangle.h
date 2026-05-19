/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/ConvexSet.h"
#include "Math/Geometry/Interval.h"
#include "cudaq/Synthesis/Math/Real.h"

#include <array>
#include <optional>
#include <string>
#include <utility>

namespace cudaq::synth {

/// Rectangle: An "upright rectangle" [x₀,x₁] × [y₀,y₁].
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §5.1.
///
/// The two-dimensional grid problem for upright rectangles A and B
/// reduces to two independent one-dimensional grid problems (Lemma 5.6):
///   u = α + βi  with α ∈ Aₓ, α● ∈ Bₓ and β ∈ Aᵧ, β● ∈ Bᵧ
/// (and similarly for the ω-offset case u = α + βi + ω).
///
/// Rectangles serve as bounding boxes for ellipses (`bbox` method) and
/// define the domain for the TDGP line-scanning phase.
///
class Rectangle : public ConvexSet {
private:
  Interval x;
  Interval y;

public:
  /// Constructs the rectangle [x1,x2] × [y1,y2].
  explicit Rectangle(const Real &x1, const Real &x2, const Real &y1,
                     const Real &y2)
      : x(x1, x2), y(y1, y2) {}

  /// Returns the x-axis interval [x₀, x₁].
  const Interval &I_x() const { return x; }

  /// Returns the y-axis interval [y₀, y₁].
  const Interval &I_y() const { return y; }

  /// Returns true iff v = (vₓ, vᵧ) lies inside the rectangle.
  bool contains(const DOmega &v) const override { return false; }

  /// Intersects the ray u(t) = u0 + t·v with the rectangle.
  ///
  /// Returns the interval [t_lo, t_hi] of parameter values for which u(t)
  /// lies inside the rectangle, or std::nullopt if the ray misses entirely.
  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const override;

  Real area() const { return (x.r() - x.l()) * (y.r() - y.l()); }

  /// Returns the rectangle as "[x0,x1] x [y0,y1]" using Interval::to_string()
  /// for each axis. Intended for logging and debugging.
  std::string to_string() const {
    return x.to_string() + " x " + y.to_string();
  }
};

} // namespace cudaq::synth
