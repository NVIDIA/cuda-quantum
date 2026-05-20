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
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Math/Ring/Zomega.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"

#include <algorithm>
#include <cassert>
#include <string>

namespace cudaq::synth {

/// Elements of the ring D[ω]:
///
///     D[ω] = Z[1/√2, i] = { aω³ + bω² + cω + d | a,b,c,d ∈ D }.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// Represented internally as u / √2^k where u ∈ Z[ω] and k ≥ 0 is the
/// denominator exponent.
class DOmega {
private:
  ZOmega _u;
  Integer _k;

public:
  /// Construct from numerator u ∈ Z[ω] and denominator exponent k ≥ 0,
  /// representing the element u / √2^k ∈ D[ω].
  explicit DOmega(const ZOmega &u = ZOmega(), const Integer &k = 0)
      : _u(u), _k(k) {}

  /// Access the numerator u ∈ Z[ω].
  const ZOmega &u() const { return _u; }

  /// Access the denominator exponent k ≥ 0.
  const Integer &k() const { return _k; }

  /// In-place assignment that reuses the existing `mpz_t` buffers inside
  /// `_u` and `_k`. Equivalent to `*this = DOmega(u, k)` but avoids the
  /// temporary's allocator traffic. `ZOmega::operator=` is `mpz_set`-based.
  void assign(const ZOmega &u, const Integer &k) {
    _u = u;
    _k = k;
  }

  /// In-place assignment that move-swaps `mpz_t` ownership with the source
  /// values.
  void assign(ZOmega &&u, Integer &&k) noexcept {
    _u = std::move(u);
    _k = std::move(k);
  }

  /// Embed an integer x as x / √2^0 ∈ D[ω].
  static DOmega from_int(const Integer &x) {
    return DOmega(ZOmega::from_int(x), 0);
  }

  /// Embed a Z[ω] element u as u / √2^0 ∈ D[ω].
  static DOmega from_zomega(const ZOmega &x) { return DOmega(x, 0); }

  /// Embed a D[√2] element x as the real-valued D[ω] element x.alpha()/√2^k.
  static DOmega from_dsqrt2(const DSqrt2 &x) {
    return DOmega(ZOmega::from_zsqrt2(x.alpha()), x.k());
  }

  /// Construct the D[ω] element (x + i·y) / √2^k from D[√2] components x, y
  /// and target denominator exponent k.
  ///
  /// Embeds x as the real part and y as the imaginary part (multiplied by
  /// ω² = i), then normalizes the result to the given k via with_denom_exp.
  static DOmega from_dsqrt2_vector(const DSqrt2 &x, const DSqrt2 &y,
                                   const Integer &k);

  /// Equality: normalizes both operands to max(_k, other._k) before comparing
  /// numerators, so two representations of the same value compare equal
  /// regardless of their denominator exponents.
  bool operator==(const DOmega &other) const;

  /// Addition in D[ω]: aligns denominator exponents before adding numerators.
  DOmega operator+(const DOmega &other) const;

  DOmega operator-(const DOmega &other) const { return *this + (-other); }

  DOmega operator-() const { return DOmega(-_u, _k); }

  /// Multiplication in D[ω]: multiplies numerators and adds denominator
  /// exponents, i.e. (u/√2^k)·(v/√2^m) = (u·v)/√2^(k+m).
  DOmega operator*(const DOmega &other) const {
    return DOmega(_u * other._u, _k + other._k);
  }

  // ---------------------------------------------------------------------------
  // Ring `automorphism`s
  // ---------------------------------------------------------------------------

  /// Complex conjugation (-)†: (u/√2^k)† = u†/√2^k.
  /// Maps ω → ω⁻¹ (Paper Definition 3.2).
  DOmega conj() const { return DOmega(_u.conj(), _k); }

  /// √2-conjugation (-)●: sends √2 → -√2 while fixing i.
  /// For odd k, the sign flips: (-√2)^k = -√2^k, so (u/√2^k)● = -u●/√2^k.
  /// For even k: (u/√2^k)● = u●/√2^k.
  /// Central to the grid problem formulation (Paper Definition 3.2, 5.1).
  DOmega conj_sq2() const {
    if (_k.is_odd())
      return DOmega(-_u.conj_sq2(), _k);
    return DOmega(_u.conj_sq2(), _k);
  }

  // ---------------------------------------------------------------------------
  // Derived properties
  // ---------------------------------------------------------------------------

  /// Scaling factor √2^k as a floating-point Real (for coordinate conversion).
  Real scale() const { return pow_sqrt2(_k); }

  /// Residue: 4-bit parity pattern (a mod 2, b mod 2, c mod 2, d mod 2) of
  /// the Z[ω] numerator. Used to determine divisibility by √2 and δ = 1 + ω
  /// in the denominator exponent reduction (Remark D.2, Lemma 7.3).
  i32 residue() const { return _u.residue(); }

  Real real() const { return _u.real() / scale(); }

  Real imag() const { return _u.imag() / scale(); }

  /// Returns the element as "(a, b, c, d)/sqrt2^k" — the ZOmega numerator
  /// coefficients and denominator exponent. Intended for logging and debugging.
  std::string to_string() const {
    return _u.to_string() + "/sqrt2^" + _k.to_string();
  }
};

// ---------------------------------------------------------------------------
// Free functions on DOmega
// ---------------------------------------------------------------------------

/// Return x · (1/√2).
///
/// Precondition: (u.b() + u.d()) and (u.c() + u.a()) must both be even.
/// This is the divisibility condition for the Z[ω] numerator to be
/// divisible by δ = 1 + ω (Remark D.2). Asserted in debug builds.
///
/// The transformation in the (a,b,c,d) basis follows from:
///   1/√2 = ω - ω³ = (ω³ acts as -√2/2 in the basis)
/// yielding new coefficients:
///   a' = (b-d)/2,  b' = (c+a)/2,  c' = (b+d)/2,  d' = (c-a)/2.
inline DOmega mul_by_inv_sqrt2(const DOmega &x) {
  const ZOmega &u = x.u();
  assert(!(u.b() + u.d()).is_odd() && !(u.c() + u.a()).is_odd() &&
         "mul_by_inv_sqrt2: numerator not divisible by δ = 1 + ω");
  ZOmega new_u((u.b() - u.d()) >> 1, (u.c() + u.a()) >> 1, (u.b() + u.d()) >> 1,
               (u.c() - u.a()) >> 1);
  return DOmega(new_u, x.k());
}

/// Return x · ω. Overloads mul_by_omega(ZOmega) for D[ω] elements.
///
/// Delegates to the ZOmega free function; the denominator exponent is
/// unchanged since ω is a unit in D[ω].
inline DOmega mul_by_omega(const DOmega &x) noexcept {
  return DOmega(mul_by_omega(x.u()), x.k());
}

/// Return x · ω⁻¹. Overloads mul_by_omega_inv(ZOmega) for D[ω] elements.
inline DOmega mul_by_omega_inv(const DOmega &x) noexcept {
  return DOmega(mul_by_omega_inv(x.u()), x.k());
}

/// Return x · ωⁿ (n reduced mod 8 internally).
/// Overloads mul_by_omega_power(ZOmega, int) for D[ω] elements.
inline DOmega mul_by_omega_power(const DOmega &x, i32 n) noexcept {
  return DOmega(mul_by_omega_power(x.u(), n), x.k());
}

/// Return x · √2^d.
///
/// For d > 0, the numerator coefficients are left-shifted by d/2 (for each
/// pair of √2 factors combining into an integer factor), and then optionally
/// multiplied by √2 in Z[ω] via the ZOmega element (-1,0,1,0) = √2.
///
/// For d < 0, the numerator must be divisible by √2^|d|; this precondition
/// is asserted in debug builds. Divisibility by even powers of √2 corresponds
/// to exact right-shifts of all coefficients; odd powers use the
/// mul_by_inv_sqrt2 transformation first.
///
/// For d = 0 returns x unchanged.
inline DOmega mul_by_sqrt2_power(const DOmega &x, const Integer &d) {
  if (d == 0)
    return x;
  if (d == -1)
    return mul_by_inv_sqrt2(x);

  const ZOmega &u = x.u();
  if (d < 0) {
    Integer abs_d = -d;
    Integer d_div_2 = abs_d >> 1;
    Integer d_mod_2 = abs_d & 1;
    if (d_mod_2 == 0) {
      // Divide all coefficients by 2^(d_div_2) (exact integer right-shift).
      Integer bit = (Integer(1) << d_div_2) - 1;
      assert((u.a() & bit) == 0 && (u.b() & bit) == 0 && (u.c() & bit) == 0 &&
             (u.d() & bit) == 0 &&
             "mul_by_sqrt2_power: numerator not divisible by 2^(|d|/2)");
      ZOmega new_u(u.a() >> d_div_2, u.b() >> d_div_2, u.c() >> d_div_2,
                   u.d() >> d_div_2);
      return DOmega(new_u, x.k());
    }
    // Odd |d|: apply one mul_by_inv_sqrt2 step then divide by 2^d_div_2.
    // The combined divisibility condition covers both steps.
    Integer bit = (Integer(1) << (d_div_2 + 1)) - 1;
    assert(((u.b() - u.d()) & bit) == 0 && ((u.c() + u.a()) & bit) == 0 &&
           ((u.b() + u.d()) & bit) == 0 && ((u.c() - u.a()) & bit) == 0 &&
           "mul_by_sqrt2_power: numerator not divisible by √2^|d|");
    ZOmega new_u(
        (u.b() - u.d()) >> (d_div_2 + 1), (u.c() + u.a()) >> (d_div_2 + 1),
        (u.b() + u.d()) >> (d_div_2 + 1), (u.c() - u.a()) >> (d_div_2 + 1));
    return DOmega(new_u, x.k());
  }

  // d > 0: multiply numerator by 2^(d_div_2), then optionally by √2.
  Integer d_div_2 = d >> 1;
  Integer d_mod_2 = d & 1;
  ZOmega new_u(
      u.a() << static_cast<int>(d_div_2), u.b() << static_cast<int>(d_div_2),
      u.c() << static_cast<int>(d_div_2), u.d() << static_cast<int>(d_div_2));
  if (d_mod_2)
    new_u = new_u * ZOmega(-1, 0, 1, 0); // multiply by √2 in Z[ω]
  return DOmega(new_u, x.k());
}

/// Re-express x with denominator exponent new_k.
///
/// Returns the same mathematical value as x but with k == new_k.
///
/// If new_k > x.k(), the numerator is scaled up by √2^(new_k - x.k()),
/// which is always exact. If new_k < x.k(), the numerator must be divisible
/// by √2^(x.k() - new_k); this precondition is asserted in debug builds.
inline DOmega with_denom_exp(const DOmega &x, const Integer &new_k) {
  ZOmega new_u = mul_by_sqrt2_power(x, new_k - x.k()).u();
  return DOmega(new_u, new_k);
}

/// Return the same value as x with the least denominator exponent, i.e., the
/// least k' ≥ 0 such that x = u'/√2^k' for u' ∈ Z[ω].
///
/// The algorithm follows Remark D.2 of Ross & Selinger (arXiv:1403.2975):
///
/// Step 1 — Remove common integer factors from the numerator:
///   Let reduce_k = `min(ntz(a), ntz(b), ntz(c), ntz(d))`, the 2-`adic`
///   valuation of the GCD of the four Z[ω] coefficients.
///   Each factor of 2 in all coefficients simultaneously cancels √2² = 2
///   from the denominator, so k decreases by 2·reduce_k.
///
/// Step 2 — Check δ = (1+ω) divisibility:
///   If after removing integer factors the residue satisfies (c+a) ≡ 0 and
///   (b+d) ≡ 0 (mod 2^(reduce_k+1)), then the numerator is additionally
///   divisible by δ, reducing k by 1.
inline DOmega to_lde(const DOmega &x) {
  const Integer &k = x.k();
  const ZOmega &u = x.u();

  Integer k_a = (u.a() == 0) ? k : ntz(u.a());
  Integer k_b = (u.b() == 0) ? k : ntz(u.b());
  Integer k_c = (u.c() == 0) ? k : ntz(u.c());
  Integer k_d = (u.d() == 0) ? k : ntz(u.d());

  Integer reduce_k = std::min(std::min(k_a, k_b), std::min(k_c, k_d));

  // Each factor of 2 shared by all coefficients cancels √2² = 2 from the
  // denominator, so the exponent decreases by 2 per factor.
  Integer new_k = k - reduce_k * 2;

  // Check δ-divisibility (Remark D.2): (c+a) and (b+d) must both vanish
  // mod 2^(reduce_k+1). Use Integer arithmetic to avoid overflow for large k.
  Integer bit = (Integer(1) << (reduce_k + 1)) - 1;
  if (((u.c() + u.a()) & bit) == 0 && ((u.b() + u.d()) & bit) == 0)
    new_k -= 1;
  return with_denom_exp(x, std::max(Integer(0), new_k));
}

/// Convert x = u/√2^k to floating-point (real, imaginary) coordinates using
/// pre-computed scaling factors.
///
/// The formulas follow Lemma 5.5 of Ross & Selinger:
///   Re(u/√2^k) = (d + (c-a)·√2/2) · inv_scale
///   Im(u/√2^k) = (b + (c+a)·√2/2) · inv_scale
///
/// where (a,b,c,d) are the Z[ω] coefficients of u in the basis
/// aω³ + bω² + cω + d.
///
/// The caller is expected to precompute and reuse inv_scale = 1/√2^k and
/// sqrt2_over_2 = √2/2 across multiple elements sharing the same k, in
/// order to amortize the cost of MPFR floating-point operations.
inline void coords_into(const DOmega &x, const Real &inv_scale,
                        const Real &sqrt2_over_2, Real &out_real,
                        Real &out_imag) noexcept {
  const ZOmega &u = x.u();
  const Real real_numer = Real(u.d()) + (Real(u.c() - u.a()) * sqrt2_over_2);
  const Real imag_numer = Real(u.b()) + (Real(u.c() + u.a()) * sqrt2_over_2);
  out_real = real_numer * inv_scale;
  out_imag = imag_numer * inv_scale;
}

// ---------------------------------------------------------------------------
// Out-of-line DOmega member definitions (require free functions above)
// ---------------------------------------------------------------------------

/// Equality: normalizes both operands to the same denominator exponent (the
/// maximum of the two) before comparing numerators. Two elements that are
/// equal as D[ω] values but stored with different k will compare equal.
inline bool DOmega::operator==(const DOmega &other) const {
  if (_k == other._k)
    return _u == other._u; // fast path: exponents already equal
  Integer k = std::max(_k, other._k);
  return with_denom_exp(*this, k).u() == with_denom_exp(other, k).u();
}

/// Addition: aligns denominator exponents to max(_k, other._k) before adding
/// the Z[ω] numerators.
inline DOmega DOmega::operator+(const DOmega &other) const {
  if (_k < other._k)
    return with_denom_exp(*this, other._k) + other;
  if (_k > other._k)
    return *this + with_denom_exp(other, _k);
  return DOmega(_u + other._u, _k);
}

/// Construct the D[ω] element (x + i·y) / √2^k from D[√2] components x, y
/// and target denominator exponent k.
///
/// The imaginary component is embedded by multiplying the D[ω] representation
/// of y by ω² = i ∈ Z[ω], represented as the ZOmega element (0,1,0,0).
inline DOmega DOmega::from_dsqrt2_vector(const DSqrt2 &x, const DSqrt2 &y,
                                         const Integer &k) {
  DOmega dx = DOmega::from_dsqrt2(x);
  // ω² = i, so multiplying y's embedding by from_zomega(ω²) scales it to the
  // imaginary axis.
  DOmega dy = DOmega::from_dsqrt2(y) * DOmega::from_zomega(ZOmega(0, 1, 0, 0));
  return with_denom_exp(dx + dy, k);
}

// DSqrt2::from_domega is defined here because it requires DOmega to be
// complete (dsqrt2.h forward-declares DOmega but cannot include domega.h).
inline DSqrt2 DSqrt2::from_domega(const DOmega &x) {
  return DSqrt2(ZSqrt2::from_zomega(x.u()), x.k());
}

} // namespace cudaq::synth
