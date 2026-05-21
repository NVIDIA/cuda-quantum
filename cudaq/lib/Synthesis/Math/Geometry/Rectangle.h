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

//===----------------------------------------------------------------------===//
// Rectangle
//===----------------------------------------------------------------------===//

/// An axis-aligned ("upright") rectangle [x_0, x_1] x [y_0, y_1].
///
/// Reference: Ross & Selinger, arXiv:1403.2975, sec. 5.1. For upright A and
/// B the 2D grid problem decouples into two independent 1D ODGPs (Lemma 5.6):
///     u = alpha + beta*i, with alpha in A_x, conj_sq2(alpha) in B_x
///                              beta  in A_y, conj_sq2(beta)  in B_y
/// and analogously for the omega-offset case u = alpha + beta*i + omega.
///
/// Rectangles also serve as bounding boxes for ellipses (`bbox`) and as the
/// search domain for the TDGP line-scanning phase.
class Rectangle : public ConvexSet {
private:
  Interval x;
  Interval y;

public:
  explicit Rectangle(const Real &x1, const Real &x2, const Real &y1,
                     const Real &y2)
      : x(x1, x2), y(y1, y2) {}

  const Interval &I_x() const { return x; }
  const Interval &I_y() const { return y; }

  /// Membership in the rectangle. Currently unimplemented (the TDGP path
  /// uses the inherited ConvexSet interface but only the line-intersection
  /// branch is exercised); kept as a stub returning false.
  bool contains(const DOmega &v) const override { return false; }

  /// Intersect the ray u(t) = u0 + t*v with the rectangle, returning the
  /// parameter interval [t_lo, t_hi] inside the box, or nullopt for a miss.
  std::optional<std::pair<Real, Real>>
  intersect(const DOmega &u0, const DOmega &v) const override;

  Real area() const { return (x.r() - x.l()) * (y.r() - y.l()); }

  /// Compact "[x0, x1] x [y0, y1]" rendering for debug logging.
  std::string to_string() const {
    return x.to_string() + " x " + y.to_string();
  }
};

} // namespace cudaq::synth
