/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Math/Real.h"

#include <cassert>

namespace cudaq::synth {

/// Interval: A closed real interval [l, r].
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §4.
///
/// Intervals are the fundamental building blocks for one-dimensional grid
/// problems (Definition 4.3). The ODGP solver finds all α ∈ Z[√2] with
/// α ∈ A = [x₀, x₁] and α● ∈ B = [y₀, y₁].
///
/// Key properties from the paper:
/// - If δΔ < 1 (where δ, Δ are interval widths), the grid problem has
///   at most one solution (Lemma 4.4).
/// - If δΔ ≥ (1+√2)², the grid problem has at least one solution.
/// - The fattened interval (free function fatten) is used in the TDGP solver
///   to handle the ω-offset variant from Lemma 5.5.
class Interval {
private:
  Real begin;
  Real end;

public:
  explicit Interval(const Real &a, const Real &b) : begin(a), end(b) {
    assert(!(a > b) && "Interval: a must be <= b");
  }

  const Real &l() const { return begin; }
  const Real &r() const { return end; }

  Real width() const { return end - begin; }

  /// Return true iff x lies in the closed interval [l, r].
  bool contains(const Real &x) const { return x >= begin && x <= end; }

  /// Scale: multiply both endpoints by scale, swapping if scale < 0.
  Interval operator*(const Real &scale) const {
    if (scale >= 0)
      return Interval(begin * scale, end * scale);
    return Interval(end * scale, begin * scale);
  }

  /// Shift: translate both endpoints by shift.
  Interval operator+(const Real &shift) const {
    return Interval(begin + shift, end + shift);
  }

  /// Shift: translate both endpoints by -shift.
  Interval operator-(const Real &shift) const {
    return Interval(begin - shift, end - shift);
  }

  std::string to_string() const {
    return "[" + begin.to_string() + ", " + end.to_string() + "]";
  }
};

// ---------------------------------------------------------------------------
// Free functions on Interval
// ---------------------------------------------------------------------------

/// Return a copy of I expanded symmetrically by amount on each side:
/// fatten([l, r], δ) = [l − δ, r + δ].
///
/// Used in the TDGP solver (tdgp.cpp) to pad the intersection interval by
/// a small multiple of its width, guarding against finite-precision rounding
/// when checking whether a grid point lies on the boundary.
inline Interval fatten(const Interval &I, const Real &amount) {
  return Interval(I.l() - amount, I.r() + amount);
}

} // namespace cudaq::synth
