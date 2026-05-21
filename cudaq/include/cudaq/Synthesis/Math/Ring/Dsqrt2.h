/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Math/Integer.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"

#include <cassert>

namespace cudaq::synth {

class DOmega;

//===----------------------------------------------------------------------===//
// DSqrt2
//===----------------------------------------------------------------------===//

/// Elements of the ring D[sqrt(2)] = Z[1/sqrt(2)]:
///
///     D[sqrt(2)] = { a + b*sqrt(2) | a, b in D }
///
/// where D = Z[1/2] is the ring of dyadic fractions.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// Representation. Stored as alpha / sqrt(2)^k with alpha in Z[sqrt(2)] and
/// k >= 0 (the "denominator exponent", Definition 3.4).
class DSqrt2 {
private:
  ZSqrt2 _alpha;
  Integer _k;

public:
  /// Construct alpha / sqrt(2)^k in D[sqrt(2)] from numerator and
  /// denominator exponent.
  explicit DSqrt2(const ZSqrt2 &alpha = ZSqrt2{0},
                  const Integer &k = 0) noexcept
      : _alpha(alpha), _k(k) {}

  /// Embed an integer val as val / sqrt(2)^0.
  explicit DSqrt2(const Integer &val) : _alpha(ZSqrt2{val}), _k(0) {}

  const ZSqrt2 &alpha() const noexcept { return _alpha; }
  const Integer &k() const noexcept { return _k; }

  /// In-place assignment that reuses the mpz_t buffers inside _alpha and
  /// _k. Saves the mpz_init / mpz_clear pair the temporary in
  /// `*this = DSqrt2(alpha, k)` would otherwise pay.
  void assign(const ZSqrt2 &alpha, const Integer &k) noexcept {
    _alpha.assign(alpha.a(), alpha.b());
    _k = k;
  }

  /// In-place assignment that swaps mpz_t ownership with the source values
  /// via ZSqrt2's move-assign and Integer's move-assign.
  void assign(ZSqrt2 &&alpha, Integer &&k) noexcept {
    _alpha = std::move(alpha);
    _k = std::move(k);
  }

  /// Embed a Z[sqrt(2)] element x as x / sqrt(2)^0.
  static DSqrt2 from_zsqrt2(const ZSqrt2 &x) { return DSqrt2(x, 0); }

  /// Project a D[omega] element to D[sqrt(2)]. Precondition: x must be
  /// real-valued, i.e. x.u().b() == 0 and x.u().a() == -x.u().c() (asserted
  /// in debug builds). Defined in domega.h once DOmega is complete.
  static DSqrt2 from_domega(const DOmega &x);

  /// Return 1 / sqrt(2)^k.
  static DSqrt2 power_of_inv_sqrt2(const Integer &k) {
    return DSqrt2(ZSqrt2{1}, k);
  }

  /// Equality modulo denominator exponent: both operands are normalised to
  /// max(_k, other._k) before the numerator comparison. Defined out of
  /// line because it depends on `with_denom_exp`.
  bool operator==(const DSqrt2 &other) const;

  /// Addition. Defined out of line for the same reason as `operator==`.
  DSqrt2 operator+(const DSqrt2 &other) const noexcept;

  DSqrt2 operator-(const DSqrt2 &other) const noexcept {
    return *this + (-other);
  }

  DSqrt2 operator-() const noexcept { return DSqrt2(-_alpha, _k); }

  /// Multiplication in D[sqrt(2)]:
  ///     (alpha / sqrt(2)^k) * (beta / sqrt(2)^m) = (alpha*beta) /
  ///     sqrt(2)^(k+m)
  DSqrt2 operator*(const DSqrt2 &other) const noexcept {
    return DSqrt2(_alpha * other._alpha, _k + other._k);
  }

  //===--------------------------------------------------------------------===//
  // Ring automorphisms
  //===--------------------------------------------------------------------===//

  /// sqrt(2)-conjugation. For odd k the sign flips
  /// ((-sqrt(2))^k = -sqrt(2)^k), so
  ///     conj_sq2(alpha / sqrt(2)^k) = -conj_sq2(alpha) / sqrt(2)^k   (k odd)
  ///                                 =  conj_sq2(alpha) / sqrt(2)^k   (k even)
  DSqrt2 conj_sq2() const {
    if (_k & 1)
      return DSqrt2(-_alpha.conj_sq2(), _k);
    return DSqrt2(_alpha.conj_sq2(), _k);
  }

  //===--------------------------------------------------------------------===//
  // Derived properties
  //===--------------------------------------------------------------------===//

  /// Parity of the constant coefficient of the Z[sqrt(2)] numerator. Always
  /// 0 or 1. Used by the ODGP parity-aware variant in odgp.cpp.
  int32_t parity() const noexcept { return _alpha.parity(); }

  /// Scaling factor sqrt(2)^k as a floating-point Real.
  Real scale() const { return pow_sqrt2(_k); }

  /// "(a, b)/sqrt2^k" rendering for debug logging.
  std::string to_string() const {
    return _alpha.to_string() + "/sqrt2^" + _k.to_string();
  }
};

//===----------------------------------------------------------------------===//
// Free functions on DSqrt2
//===----------------------------------------------------------------------===//

/// Multiply x by 1/sqrt(2) by absorbing the factor into the Z[sqrt(2)]
/// numerator. The transformation
///     (a + b*sqrt(2)) / sqrt(2) = b + (a/2)*sqrt(2)
/// requires `a` to be even; the precondition is asserted in debug builds.
inline DSqrt2 mul_by_inv_sqrt2(const DSqrt2 &x) {
  const ZSqrt2 &a = x.alpha();
  assert(!(a.a()).is_odd() && "mul_by_inv_sqrt2: alpha.a() not divisible by 2");
  return DSqrt2(ZSqrt2(a.b(), a.a() >> 1), x.k());
}

/// Multiply x by sqrt(2)^d while leaving the denominator exponent k
/// unchanged.
///
/// For d > 0 we shift each pair of sqrt(2) factors into an integer factor
/// of 2 (left-shift the coefficients by d/2) and, for an odd remainder,
/// multiply once more by ZSqrt2(0, 1) = sqrt(2).
///
/// For d < 0 the numerator must be divisible by sqrt(2)^|d| (asserted).
/// Even |d| corresponds to exact right-shifts of all coefficients; odd
/// |d| applies the mul_by_inv_sqrt2 transformation once and then shifts.
inline DSqrt2 mul_by_sqrt2_power(const DSqrt2 &x, const Integer &d) {
  if (d == 0)
    return x;
  if (d == -1)
    return mul_by_inv_sqrt2(x);

  const ZSqrt2 &a = x.alpha();
  if (d < 0) {
    Integer abs_d = -d;
    Integer d_div_2 = abs_d >> 1;
    Integer d_mod_2 = abs_d & 1;
    if (d_mod_2 == 0) {
      Integer bit = (Integer(1) << d_div_2) - 1;
      assert((a.a() & bit) == 0 && (a.b() & bit) == 0 &&
             "mul_by_sqrt2_power: numerator not divisible by 2^(|d|/2)");
      return DSqrt2(ZSqrt2(a.a() >> d_div_2, a.b() >> d_div_2), x.k());
    }
    // Odd |d|: combine one mul_by_inv_sqrt2 step with the shift. The
    // combined divisibility check covers both substeps in one assert.
    //   (a + b*sqrt(2)) / sqrt(2)^|d|
    //     = (b + (a/2)*sqrt(2)) / 2^d_div_2
    //     = (b/2^d_div_2,  a/2^(d_div_2+1))
    Integer bit_b = (Integer(1) << d_div_2) - 1;
    Integer bit_a = (Integer(1) << (d_div_2 + 1)) - 1;
    assert((a.a() & bit_a) == 0 && (a.b() & bit_b) == 0 &&
           "mul_by_sqrt2_power: numerator not divisible by sqrt(2)^|d|");
    return DSqrt2(ZSqrt2(a.b() >> d_div_2, a.a() >> (d_div_2 + 1)), x.k());
  }

  // d > 0: multiply numerator by 2^(d_div_2), then optionally by sqrt(2).
  Integer d_div_2 = d >> 1;
  Integer d_mod_2 = d & 1;
  ZSqrt2 new_alpha(a.a() << static_cast<int32_t>(d_div_2),
                   a.b() << static_cast<int32_t>(d_div_2));
  if (d_mod_2)
    new_alpha = new_alpha * ZSqrt2(0, 1); // multiply by sqrt(2) in Z[sqrt(2)]
  return DSqrt2(new_alpha, x.k());
}

/// Re-express x with denominator exponent `new_k`. Same value, different
/// representation. If new_k > x.k() the numerator is scaled up by
/// sqrt(2)^(new_k - x.k()), always exact. If new_k < x.k() the numerator
/// must be divisible by sqrt(2)^(x.k() - new_k); the precondition is
/// asserted.
inline DSqrt2 with_denom_exp(const DSqrt2 &x, const Integer &new_k) {
  ZSqrt2 new_alpha = mul_by_sqrt2_power(x, new_k - x.k()).alpha();
  return DSqrt2(new_alpha, new_k);
}

/// Convert x to a floating-point Real. Callers that process many elements
/// at the same k should compute pow_sqrt2(k) once and reuse it -- the
/// generic to_real does the pow on every call.
inline Real to_real(const DSqrt2 &x) {
  return to_real(x.alpha()) / pow_sqrt2(x.k());
}

/// Multiply x by sqrt(2)^d by decrementing the denominator exponent, with
/// the numerator unchanged: O(1) shortcut over `mul_by_sqrt2_power` when
/// the caller knows the result will be re-expressed with a lower k.
/// Precondition: d <= x.k() so the resulting exponent is non-negative.
inline DSqrt2 absorb_sqrt2_power(const DSqrt2 &x, const Integer &d) {
  assert(!(d > x.k()) && "absorb_sqrt2_power: d > x.k() would produce negative "
                         "denominator exponent");
  return DSqrt2(x.alpha(), x.k() - d);
}

//===----------------------------------------------------------------------===//
// Out-of-line DSqrt2 members (depend on free functions above)
//===----------------------------------------------------------------------===//

/// Ordering: align both operands to the same denominator exponent and
/// compare the Z[sqrt(2)] numerators exactly (no floating-point).
inline bool operator<(const DSqrt2 &lhs, const DSqrt2 &rhs) {
  Integer k = std::max(lhs.k(), rhs.k());
  return with_denom_exp(lhs, k).alpha() < with_denom_exp(rhs, k).alpha();
}

inline bool operator<=(const DSqrt2 &lhs, const DSqrt2 &rhs) {
  return !(rhs < lhs);
}

inline bool operator>(const DSqrt2 &lhs, const DSqrt2 &rhs) {
  return rhs < lhs;
}

inline bool operator>=(const DSqrt2 &lhs, const DSqrt2 &rhs) {
  return !(lhs < rhs);
}

/// Equality: two DSqrt2 values equal as elements of D[sqrt(2)] compare
/// equal even if stored with different denominator exponents.
inline bool DSqrt2::operator==(const DSqrt2 &other) const {
  if (_k == other._k)
    return _alpha == other._alpha; // fast path
  Integer k = std::max(_k, other._k);
  return with_denom_exp(*this, k).alpha() == with_denom_exp(other, k).alpha();
}

/// Addition: align denominator exponents to max(_k, other._k), then add
/// the Z[sqrt(2)] numerators.
inline DSqrt2 DSqrt2::operator+(const DSqrt2 &other) const noexcept {
  if (_k < other._k)
    return with_denom_exp(*this, other._k) + other;
  if (_k > other._k)
    return *this + with_denom_exp(other, _k);
  return DSqrt2(_alpha + other._alpha, _k);
}

} // namespace cudaq::synth
