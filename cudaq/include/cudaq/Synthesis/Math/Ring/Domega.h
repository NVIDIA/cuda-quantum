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
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Math/Ring/Zomega.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"

#include <algorithm>
#include <cassert>
#include <string>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// DOmega
//===----------------------------------------------------------------------===//

/// Elements of the ring
///
///     D[omega] = Z[1/sqrt(2), i] = { a*omega^3 + b*omega^2 + c*omega + d
///                                    | a, b, c, d in D }
///
/// where D = Z[1/2] is the ring of dyadic fractions.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// Representation. Stored as u / sqrt(2)^k with u in Z[omega] and k >= 0
/// (the "denominator exponent").
class DOmega {
private:
  ZOmega _u;
  Integer _k;

public:
  /// Construct u / sqrt(2)^k.
  explicit DOmega(const ZOmega &u = ZOmega(), const Integer &k = 0)
      : _u(u), _k(k) {}

  const ZOmega &u() const { return _u; }
  const Integer &k() const { return _k; }

  /// In-place assignment that reuses the mpz_t buffers inside _u and _k.
  /// Saves the temporary's allocator traffic that `*this = DOmega(u, k)`
  /// would otherwise pay.
  void assign(const ZOmega &u, const Integer &k) {
    _u = u;
    _k = k;
  }

  /// In-place assignment that move-swaps mpz_t ownership with the source
  /// values.
  void assign(ZOmega &&u, Integer &&k) noexcept {
    _u = std::move(u);
    _k = std::move(k);
  }

  /// Embed an integer x as x / sqrt(2)^0.
  static DOmega from_int(const Integer &x) {
    return DOmega(ZOmega::from_int(x), 0);
  }

  /// Embed a Z[omega] element u as u / sqrt(2)^0.
  static DOmega from_zomega(const ZOmega &x) { return DOmega(x, 0); }

  /// Embed a real-valued D[omega] element from its D[sqrt(2)] form.
  static DOmega from_dsqrt2(const DSqrt2 &x) {
    return DOmega(ZOmega::from_zsqrt2(x.alpha()), x.k());
  }

  /// Build the D[omega] element (x + i*y) / sqrt(2)^k from D[sqrt(2)]
  /// components x and y. The imaginary part rides on omega^2 = i and the
  /// final value is renormalised to the requested k via with_denom_exp.
  static DOmega from_dsqrt2_vector(const DSqrt2 &x, const DSqrt2 &y,
                                   const Integer &k);

  /// Equality modulo denominator exponent: both operands are normalised
  /// to max(_k, other._k) before the numerator comparison.
  bool operator==(const DOmega &other) const;

  /// Addition: aligns denominator exponents before adding numerators.
  DOmega operator+(const DOmega &other) const;

  DOmega operator-(const DOmega &other) const { return *this + (-other); }

  DOmega operator-() const { return DOmega(-_u, _k); }

  /// Multiplication:
  ///   (u / sqrt(2)^k) * (v / sqrt(2)^m) = (u*v) / sqrt(2)^(k+m)
  DOmega operator*(const DOmega &other) const {
    return DOmega(_u * other._u, _k + other._k);
  }

  //===--------------------------------------------------------------------===//
  // Ring automorphisms
  //===--------------------------------------------------------------------===//

  /// Complex conjugation: (u / sqrt(2)^k)^* = conj(u) / sqrt(2)^k. Maps
  /// omega to omega^-1 (Paper Definition 3.2).
  DOmega conj() const { return DOmega(_u.conj(), _k); }

  /// sqrt(2)-conjugation. For odd k the sign flips
  /// ((-sqrt(2))^k = -sqrt(2)^k), so
  ///     conj_sq2(u / sqrt(2)^k) = -conj_sq2(u) / sqrt(2)^k   (k odd)
  ///                             =  conj_sq2(u) / sqrt(2)^k   (k even)
  /// Central to the grid problem formulation (Paper Definitions 3.2, 5.1).
  DOmega conj_sq2() const {
    if (_k.is_odd())
      return DOmega(-_u.conj_sq2(), _k);
    return DOmega(_u.conj_sq2(), _k);
  }

  //===--------------------------------------------------------------------===//
  // Derived properties
  //===--------------------------------------------------------------------===//

  /// Floating-point scaling factor sqrt(2)^k.
  Real scale() const { return pow_sqrt2(_k); }

  /// Residue: 4-bit parity pattern of the Z[omega] numerator (Remark D.2,
  /// Lemma 7.3). Drives the denominator-exponent reduction.
  i32 residue() const { return _u.residue(); }

  Real real() const { return _u.real() / scale(); }
  Real imag() const { return _u.imag() / scale(); }

  /// "(a, b, c, d)/sqrt2^k" rendering for debug logging.
  std::string to_string() const {
    return _u.to_string() + "/sqrt2^" + _k.to_string();
  }
};

//===----------------------------------------------------------------------===//
// Free functions on DOmega
//===----------------------------------------------------------------------===//

/// Multiply x by 1/sqrt(2).
///
/// Precondition: (u.b() + u.d()) and (u.c() + u.a()) must both be even.
/// This is the divisibility-by-delta = 1 + omega condition (Remark D.2);
/// without it the result would not lie in Z[omega] / sqrt(2)^k.
///
/// In the (a, b, c, d) basis, 1/sqrt(2) acts as
///     a' = (b - d)/2,  b' = (c + a)/2,  c' = (b + d)/2,  d' = (c - a)/2.
inline DOmega mul_by_inv_sqrt2(const DOmega &x) {
  const ZOmega &u = x.u();
  assert(!(u.b() + u.d()).is_odd() && !(u.c() + u.a()).is_odd() &&
         "mul_by_inv_sqrt2: numerator not divisible by delta = 1 + omega");
  ZOmega new_u((u.b() - u.d()) >> 1, (u.c() + u.a()) >> 1, (u.b() + u.d()) >> 1,
               (u.c() - u.a()) >> 1);
  return DOmega(new_u, x.k());
}

/// Multiply x by omega. Overload of mul_by_omega(ZOmega) for D[omega] --
/// the denominator exponent is unchanged because omega is a unit in
/// D[omega].
inline DOmega mul_by_omega(const DOmega &x) noexcept {
  return DOmega(mul_by_omega(x.u()), x.k());
}

inline DOmega mul_by_omega_inv(const DOmega &x) noexcept {
  return DOmega(mul_by_omega_inv(x.u()), x.k());
}

inline DOmega mul_by_omega_power(const DOmega &x, i32 n) noexcept {
  return DOmega(mul_by_omega_power(x.u(), n), x.k());
}

/// Multiply x by sqrt(2)^d.
///
/// For d > 0, every pair of sqrt(2) factors combines into a factor of 2,
/// so each coefficient is left-shifted by d/2; an odd remainder picks up
/// one extra multiplication by sqrt(2) in Z[omega] (the ZOmega element
/// (-1, 0, 1, 0)).
///
/// For d < 0, the numerator must be divisible by sqrt(2)^|d| (asserted).
/// Even |d| corresponds to exact right-shifts; odd |d| applies the
/// mul_by_inv_sqrt2 transformation first and then shifts.
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
      Integer bit = (Integer(1) << d_div_2) - 1;
      assert((u.a() & bit) == 0 && (u.b() & bit) == 0 && (u.c() & bit) == 0 &&
             (u.d() & bit) == 0 &&
             "mul_by_sqrt2_power: numerator not divisible by 2^(|d|/2)");
      ZOmega new_u(u.a() >> d_div_2, u.b() >> d_div_2, u.c() >> d_div_2,
                   u.d() >> d_div_2);
      return DOmega(new_u, x.k());
    }
    // Odd |d|: combine one mul_by_inv_sqrt2 step with the shift; one
    // assert covers the divisibility for both substeps.
    Integer bit = (Integer(1) << (d_div_2 + 1)) - 1;
    assert(((u.b() - u.d()) & bit) == 0 && ((u.c() + u.a()) & bit) == 0 &&
           ((u.b() + u.d()) & bit) == 0 && ((u.c() - u.a()) & bit) == 0 &&
           "mul_by_sqrt2_power: numerator not divisible by sqrt(2)^|d|");
    ZOmega new_u(
        (u.b() - u.d()) >> (d_div_2 + 1), (u.c() + u.a()) >> (d_div_2 + 1),
        (u.b() + u.d()) >> (d_div_2 + 1), (u.c() - u.a()) >> (d_div_2 + 1));
    return DOmega(new_u, x.k());
  }

  // d > 0: multiply numerator by 2^(d_div_2), then optionally by sqrt(2).
  Integer d_div_2 = d >> 1;
  Integer d_mod_2 = d & 1;
  ZOmega new_u(
      u.a() << static_cast<int>(d_div_2), u.b() << static_cast<int>(d_div_2),
      u.c() << static_cast<int>(d_div_2), u.d() << static_cast<int>(d_div_2));
  if (d_mod_2)
    new_u = new_u * ZOmega(-1, 0, 1, 0); // multiply by sqrt(2) in Z[omega]
  return DOmega(new_u, x.k());
}

/// Re-express x with denominator exponent `new_k`. Same value, different
/// representation. Scaling up is always exact; scaling down requires the
/// numerator to be divisible by sqrt(2)^(x.k() - new_k) and is asserted.
inline DOmega with_denom_exp(const DOmega &x, const Integer &new_k) {
  ZOmega new_u = mul_by_sqrt2_power(x, new_k - x.k()).u();
  return DOmega(new_u, new_k);
}

/// Reduce x to its least denominator exponent: smallest k' >= 0 such that
/// x can be written as u' / sqrt(2)^k' with u' in Z[omega].
///
/// Algorithm (Remark D.2 of Ross & Selinger):
///   1. Strip the common 2-adic valuation of the four Z[omega] coefficients
///      (reduce_k = min of ntz over a, b, c, d). Each shared factor of 2
///      cancels sqrt(2)^2 from the denominator, so k decreases by
///      2*reduce_k.
///   2. Test delta = (1 + omega) divisibility: if (c + a) == 0 and
///      (b + d) == 0 modulo 2^(reduce_k + 1), the numerator is also
///      divisible by delta, so k decreases by an additional 1.
inline DOmega to_lde(const DOmega &x) {
  const Integer &k = x.k();
  const ZOmega &u = x.u();

  Integer k_a = (u.a() == 0) ? k : ntz(u.a());
  Integer k_b = (u.b() == 0) ? k : ntz(u.b());
  Integer k_c = (u.c() == 0) ? k : ntz(u.c());
  Integer k_d = (u.d() == 0) ? k : ntz(u.d());

  Integer reduce_k = std::min(std::min(k_a, k_b), std::min(k_c, k_d));

  Integer new_k = k - reduce_k * 2;

  // delta-divisibility (Remark D.2). All checks done in Integer arithmetic
  // to remain exact for large k.
  Integer bit = (Integer(1) << (reduce_k + 1)) - 1;
  if (((u.c() + u.a()) & bit) == 0 && ((u.b() + u.d()) & bit) == 0)
    new_k -= 1;
  return with_denom_exp(x, std::max(Integer(0), new_k));
}

/// Convert x = u / sqrt(2)^k into floating-point (real, imaginary) parts,
/// reusing caller-provided scaling factors.
///
/// Formulas (Lemma 5.5):
///   Re(u / sqrt(2)^k) = (d + (c - a) * sqrt(2)/2) * inv_scale
///   Im(u / sqrt(2)^k) = (b + (c + a) * sqrt(2)/2) * inv_scale
///
/// Callers that process many elements with the same k should compute
/// inv_scale = 1 / sqrt(2)^k and sqrt2_over_2 = sqrt(2) / 2 once and pass
/// them in; the MPFR cost is then amortised over all elements.
inline void coords_into(const DOmega &x, const Real &inv_scale,
                        const Real &sqrt2_over_2, Real &out_real,
                        Real &out_imag) noexcept {
  const ZOmega &u = x.u();
  const Real real_numer = Real(u.d()) + (Real(u.c() - u.a()) * sqrt2_over_2);
  const Real imag_numer = Real(u.b()) + (Real(u.c() + u.a()) * sqrt2_over_2);
  out_real = real_numer * inv_scale;
  out_imag = imag_numer * inv_scale;
}

//===----------------------------------------------------------------------===//
// Out-of-line DOmega members (depend on free functions above)
//===----------------------------------------------------------------------===//

inline bool DOmega::operator==(const DOmega &other) const {
  if (_k == other._k)
    return _u == other._u; // fast path: exponents already match
  Integer k = std::max(_k, other._k);
  return with_denom_exp(*this, k).u() == with_denom_exp(other, k).u();
}

inline DOmega DOmega::operator+(const DOmega &other) const {
  if (_k < other._k)
    return with_denom_exp(*this, other._k) + other;
  if (_k > other._k)
    return *this + with_denom_exp(other, _k);
  return DOmega(_u + other._u, _k);
}

inline DOmega DOmega::from_dsqrt2_vector(const DSqrt2 &x, const DSqrt2 &y,
                                         const Integer &k) {
  DOmega dx = DOmega::from_dsqrt2(x);
  // omega^2 = i; multiplying y's embedding by from_zomega(omega^2) rotates
  // it onto the imaginary axis.
  DOmega dy = DOmega::from_dsqrt2(y) * DOmega::from_zomega(ZOmega(0, 1, 0, 0));
  return with_denom_exp(dx + dy, k);
}

// DSqrt2::from_domega is here because it needs DOmega to be complete
// (dsqrt2.h only forward-declares DOmega).
inline DSqrt2 DSqrt2::from_domega(const DOmega &x) {
  return DSqrt2(ZSqrt2::from_zomega(x.u()), x.k());
}

} // namespace cudaq::synth
