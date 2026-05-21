/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Math/Real.h"

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Interval
//===----------------------------------------------------------------------===//

/// A closed real interval [l, r].
///
/// Reference: Ross & Selinger, arXiv:1403.2975, sec. 4. Intervals are the
/// building blocks of the one-dimensional grid problem (Definition 4.3): the
/// ODGP solver finds all alpha in Z[`sqrt`(2)] with alpha in A = [x_0, x_1]
/// and conj_sq2(alpha) in B = [y_0, y_1].
///
/// Solvability bounds carried over from the paper:
///   width(A) * width(B) < 1            -- at most one solution (Lemma 4.4).
///   width(A) * width(B) >= (1+`sqrt`(2))^2 -- at least one solution.
///
/// Degenerate intervals (l > r, i.e. width() < 0) are permitted and represent
/// the empty set. The ODGP solver short-circuits to an empty result on these.
class Interval {
private:
  Real begin;
  Real end;

public:
  explicit Interval(const Real &a, const Real &b) : begin(a), end(b) {}

  const Real &l() const { return begin; }
  const Real &r() const { return end; }

  Real width() const { return end - begin; }

  /// True iff x lies in the closed interval [l, r].
  bool contains(const Real &x) const { return x >= begin && x <= end; }

  /// Scale: multiply both endpoints by `scale`. Negative scales swap the
  /// endpoints so the result remains a non-degenerate (l <= r) interval.
  Interval operator*(const Real &scale) const {
    if (scale >= 0)
      return Interval(begin * scale, end * scale);
    return Interval(end * scale, begin * scale);
  }

  /// Translate both endpoints by `shift`.
  Interval operator+(const Real &shift) const {
    return Interval(begin + shift, end + shift);
  }

  Interval operator-(const Real &shift) const {
    return Interval(begin - shift, end - shift);
  }

  std::string to_string() const {
    return "[" + begin.to_string() + ", " + end.to_string() + "]";
  }
};

//===----------------------------------------------------------------------===//
// Free functions on Interval
//===----------------------------------------------------------------------===//

/// Symmetric expansion: fatten([l, r], delta) = [l - delta, r + delta].
///
/// Used by the TDGP solver to pad an intersection interval by a small
/// multiple of its width, absorbing finite-precision rounding when checking
/// whether a grid point lies on the boundary.
inline Interval fatten(const Interval &I, const Real &amount) {
  return Interval(I.l() - amount, I.r() + amount);
}

} // namespace cudaq::synth
