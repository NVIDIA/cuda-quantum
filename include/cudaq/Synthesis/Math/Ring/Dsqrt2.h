/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
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

/// Element of the ring
///
///      D[√2] = Z[1/√2] = { a + b√2 | a, b ∈ D },
///
/// where D = Z[1/2] = { a/2^k | a ∈ Z, k ∈ N } is the ring of dyadic fractions.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// Represented internally as α / √2^k where α ∈ Z[√2] and k ≥ 0. The integer k
/// is the "denominator exponent" (Definition 3.4)
class DSqrt2 {
private:
  ZSqrt2 _alpha;
  Integer _k;

public:
  /// Construct from numerator α ∈ Z[√2] and denominator exponent k ≥ 0,
  /// representing the element α / √2^k ∈ D[√2].
  explicit DSqrt2(const ZSqrt2 &alpha = ZSqrt2{0},
                  const Integer &k = 0) noexcept
      : _alpha(alpha), _k(k) {}

  /// Construct from an integer, embedding val as val / √2^0 ∈ D[√2].
  explicit DSqrt2(const Integer &val) : _alpha(ZSqrt2{val}), _k(0) {}

  /// Access the numerator α ∈ Z[√2].
  const ZSqrt2 &alpha() const noexcept { return _alpha; }

  /// Access the denominator exponent k ≥ 0.
  const Integer &k() const noexcept { return _k; }

  /// Embed a Z[√2] element x as x / √2^0 ∈ D[√2].
  static DSqrt2 from_zsqrt2(const ZSqrt2 &x) { return DSqrt2(x, 0); }

  /// Convert a D[ω] element x to D[√2].
  /// Precondition: x must be real-valued, i.e. x.u() must satisfy
  /// x.u().b() == 0 and x.u().a() == -x.u().c() (asserted in debug builds).
  /// Defined in domega.h where DOmega is complete.
  static DSqrt2 from_domega(const DOmega &x);

  /// Return 1/√2^k ∈ D[√2], i.e., the element with numerator 1 and
  /// denominator exponent k.
  static DSqrt2 power_of_inv_sqrt2(const Integer &k) {
    return DSqrt2(ZSqrt2{1}, k);
  }

  /// Equality: normalizes both operands to max(_k, other._k) before comparing
  /// numerators. Defined out-of-line because it requires with_denom_exp.
  bool operator==(const DSqrt2 &other) const;

  /// Addition: aligns denominator exponents before adding numerators.
  /// Defined out-of-line because it requires with_denom_exp.
  DSqrt2 operator+(const DSqrt2 &other) const noexcept;

  DSqrt2 operator-(const DSqrt2 &other) const noexcept {
    return *this + (-other);
  }

  DSqrt2 operator-() const noexcept { return DSqrt2(-_alpha, _k); }

  /// Multiplication in D[√2]: (α/√2^k)·(β/√2^m) = (α·β)/√2^(k+m).
  DSqrt2 operator*(const DSqrt2 &other) const noexcept {
    return DSqrt2(_alpha * other._alpha, _k + other._k);
  }

  // ---------------------------------------------------------------------------
  // Ring `automorphism`s
  // ---------------------------------------------------------------------------

  /// √2-conjugation (-)●: sends √2 → -√2.
  /// For odd k, the sign flips: (-√2)^k = -√2^k, so (α/√2^k)● = -α●/√2^k.
  /// For even k: (α/√2^k)● = α●/√2^k.
  /// Used in the grid problem formulation (Paper Definition 3.2, 5.1).
  DSqrt2 conj_sq2() const {
    if (_k & 1)
      return DSqrt2(-_alpha.conj_sq2(), _k);
    return DSqrt2(_alpha.conj_sq2(), _k);
  }

  // ---------------------------------------------------------------------------
  // Derived properties
  // ---------------------------------------------------------------------------

  /// Parity of the constant coefficient of the Z[√2] numerator (a mod 2).
  /// Result is always 0 or 1. Used to determine grid-problem parity
  /// constraints (odgp.cpp).
  i32 parity() const noexcept { return _alpha.parity(); }

  /// Scaling factor √2^k as a floating-point Real.
  Real scale() const { return pow_sqrt2(_k); }

  /// Returns the element as "(a, b)/sqrt2^k" — the ZSqrt2 numerator
  /// coefficients and denominator exponent. Intended for logging and debugging.
  std::string to_string() const {
    return _alpha.to_string() + "/sqrt2^" + _k.to_string();
  }
};

// ---------------------------------------------------------------------------
// Free functions on DSqrt2
// ---------------------------------------------------------------------------

/// Return x · (1/√2) by absorbing the factor into the Z[√2] numerator.
///
/// Precondition: alpha.a() must be even (the constant coefficient of the
/// Z[√2] numerator must be divisible by 2). Asserted in debug builds.
///
/// The transformation follows from (a + b√2)/√2 = b + (a/2)√2:
///   (a, b) → (b, a/2),  with k unchanged.
inline DSqrt2 mul_by_inv_sqrt2(const DSqrt2 &x) {
  const ZSqrt2 &a = x.alpha();
  assert(!(a.a()).is_odd() && "mul_by_inv_sqrt2: alpha.a() not divisible by 2");
  return DSqrt2(ZSqrt2(a.b(), a.a() >> 1), x.k());
}

/// Return x · √2^d by modifying the Z[√2] numerator, leaving the denominator
/// exponent k unchanged.
///
/// For d > 0, the numerator coefficients are left-shifted by d/2 (each pair
/// of √2 factors combines into an integer factor of 2), then optionally
/// multiplied by ZSqrt2(0,1) = √2 for an odd remainder.
///
/// For d < 0, the numerator must be divisible by √2^|d|; this precondition
/// is asserted in debug builds. Even |d| corresponds to exact right-shifts;
/// odd |d| applies mul_by_inv_sqrt2 first then shifts.
///
/// For d = 0 returns x unchanged.
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
      // Divide both coefficients by 2^(d_div_2) (exact integer right-shifts).
      Integer bit = (Integer(1) << d_div_2) - 1;
      assert((a.a() & bit) == 0 && (a.b() & bit) == 0 &&
             "mul_by_sqrt2_power: numerator not divisible by 2^(|d|/2)");
      return DSqrt2(ZSqrt2(a.a() >> d_div_2, a.b() >> d_div_2), x.k());
    }
    // Odd |d|: apply one mul_by_inv_sqrt2 step then divide by 2^d_div_2.
    // (a + b√2)/√2^|d| = (b + (a/2)√2) / 2^d_div_2
    //                  = (b/2^d_div_2, a/2^(d_div_2+1))
    Integer bit_b = (Integer(1) << d_div_2) - 1;
    Integer bit_a = (Integer(1) << (d_div_2 + 1)) - 1;
    assert((a.a() & bit_a) == 0 && (a.b() & bit_b) == 0 &&
           "mul_by_sqrt2_power: numerator not divisible by √2^|d|");
    return DSqrt2(ZSqrt2(a.b() >> d_div_2, a.a() >> (d_div_2 + 1)), x.k());
  }

  // d > 0: multiply numerator by 2^(d_div_2), then optionally by √2.
  Integer d_div_2 = d >> 1;
  Integer d_mod_2 = d & 1;
  ZSqrt2 new_alpha(a.a() << static_cast<i32>(d_div_2),
                   a.b() << static_cast<i32>(d_div_2));
  if (d_mod_2)
    new_alpha = new_alpha * ZSqrt2(0, 1); // multiply by √2 in Z[√2]
  return DSqrt2(new_alpha, x.k());
}

/// Re-express x with denominator exponent new_k.
///
/// Returns the same mathematical value as x but with k == new_k.
///
/// If new_k > x.k(), the numerator is scaled up by √2^(new_k - x.k()),
/// which is always exact. If new_k < x.k(), the numerator must be divisible
/// by √2^(x.k() - new_k); this precondition is asserted in debug builds.
inline DSqrt2 with_denom_exp(const DSqrt2 &x, const Integer &new_k) {
  ZSqrt2 new_alpha = mul_by_sqrt2_power(x, new_k - x.k()).alpha();
  return DSqrt2(new_alpha, new_k);
}

/// Convert x = α/√2^k to a floating-point Real.
///
/// Delegates to to_real(ZSqrt2) for the numerator, then divides by the
/// floating-point scale √2^k. For callers processing many D[√2] elements
/// with the same denominator exponent, computing pow_sqrt2(k) once and
/// reusing it is more efficient than calling this function repeatedly.
inline Real to_real(const DSqrt2 &x) {
  return to_real(x.alpha()) / pow_sqrt2(x.k());
}

/// Return x · √2^d by decrementing the denominator exponent, leaving the Z[√2]
/// numerator unchanged.
///
/// This is a O(1) alternative to mul_by_sqrt2_power for the case where the
/// caller knows the result will be re-expressed with a lower k. The identity
/// α/√2^k · √2^d = α/√2^(k-d) is used directly.
///
/// Precondition: d ≤ x.k(), so that k - d ≥ 0. Asserted in debug builds.
inline DSqrt2 absorb_sqrt2_power(const DSqrt2 &x, const Integer &d) {
  assert(!(d > x.k()) && "absorb_sqrt2_power: d > x.k() would produce negative "
                         "denominator exponent");
  return DSqrt2(x.alpha(), x.k() - d);
}

// ---------------------------------------------------------------------------
// Out-of-line DSqrt2 member definitions (require free functions above)
// ---------------------------------------------------------------------------

/// Ordering helpers: align both operands to the same denominator exponent, then
/// compare the resulting Z[√2] numerators using ZSqrt2::operator<.
///
/// The comparison is exact (no floating-point): a/√2^k ≤ b/√2^m is equivalent
/// to a·√2^(max-k) ≤ b·√2^(max-m) as Z[√2] elements.

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

/// Equality: normalizes both operands to max(_k, other._k) before comparing
/// numerators. Two elements equal as D[√2] values but with different k compare
/// equal.
inline bool DSqrt2::operator==(const DSqrt2 &other) const {
  if (_k == other._k)
    return _alpha == other._alpha; // fast path: exponents already equal
  Integer k = std::max(_k, other._k);
  return with_denom_exp(*this, k).alpha() == with_denom_exp(other, k).alpha();
}

/// Addition: aligns denominator exponents to max(_k, other._k) before adding
/// the Z[√2] numerators.
inline DSqrt2 DSqrt2::operator+(const DSqrt2 &other) const noexcept {
  if (_k < other._k)
    return with_denom_exp(*this, other._k) + other;
  if (_k > other._k)
    return *this + with_denom_exp(other, _k);
  return DSqrt2(_alpha + other._alpha, _k);
}

} // namespace cudaq::synth
