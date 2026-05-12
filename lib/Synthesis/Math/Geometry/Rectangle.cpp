/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Geometry/Rectangle.h"
#include "cudaq/Synthesis/Math/Real.h"

#include <algorithm>

namespace cudaq::synth {

std::optional<std::pair<Real, Real>>
Rectangle::intersect(const DOmega &u0,
                     const DOmega &v) const {
  //static const Real tolerance(1e-30);

  //// We parameterize the ray as u(t) = u0 + t·v and want all t such that
  //// u(t) lies inside the axis-aligned rectangle:
  ////
  ////   x.l() ≤ u_x(t) ≤ x.r()
  ////   y.l() ≤ u_y(t) ≤ y.r()
  ////
  //// Each inequality gives an interval of t; the intersection of all these
  //// intervals is the set of t for which the ray is in the rectangle.
  //Real t_low = Real::neg_inf(); // current lower bound on t
  //Real t_high = Real::inf();    // current upper bound on t

  //// Update [t_low, t_high] given one coordinate constraint:
  ////
  ////   a ≤ p(t) = p0 + t·dp ≤ b.
  ////
  //// If dp ≈ 0, the coordinate is almost constant along the ray; then we just
  //// check whether that constant value lies within [a, b]. If not, there is no
  //// intersection at all.
  ////
  //// If dp ≠ 0, we solve:
  ////   a ≤ p0 + t·dp ≤ b  ⇒  two bounds on t:
  ////     t ≥ (a - p0)/dp,   t ≤ (b - p0)/dp   (up to swapping if dp < 0),
  //// and intersect this new [t_min, t_max] with the global [t_low, t_high].
  ////
  //// Returns false if the intersection becomes empty.
  //auto update_bounds = [&](const Real &p0, const Real &dp, const Real &a,
  //                         const Real &b) -> bool {
  //  // Nearly parallel to this axis: p(t) ≈ constant = p0.
  //  if (abs(dp) < tolerance)
  //    // If the constant value lies outside [a, b], the ray misses the
  //    // rectangle.
  //    return !(p0 < a - tolerance || p0 > b + tolerance);

  //  // General case: dp nonzero, compute interval of t for which a ≤ p(t) ≤ b.
  //  Real t1 = (a - p0) / dp;
  //  Real t2 = (b - p0) / dp;
  //  if (t1 > t2)
  //    std::swap(t1, t2); // ensure t1 is the lower bound, t2 the upper bound

  //  // Intersect with current global t interval.
  //  // Move t1/t2 since they are locals not used after this point.
  //  if (t1 > t_low)
  //    t_low = std::move(t1);
  //  if (t2 < t_high)
  //    t_high = std::move(t2);

  //  // If the interval collapses (no overlap), there is no intersection.
  //  return t_low <= t_high + tolerance;
  //};

  //// Combine x-interval and y-interval constraints.
  //if (!update_bounds(u0[0], v[0], x.l(), x.r()))
  //  return std::nullopt;
  //if (!update_bounds(u0[1], v[1], y.l(), y.r()))
  //  return std::nullopt;

  //if (t_low > t_high)
  //  return std::nullopt;

  //return std::make_pair(std::move(t_low), std::move(t_high));
  exit(1);
}

} // namespace cudaq::synth
