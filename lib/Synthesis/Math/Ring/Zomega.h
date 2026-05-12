/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <complex>
#include <string>

#include "Math/Integer.h"
#include "Math/Ring/Zsqrt2.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Support/Result.h"

namespace cudaq::synth {

/// Elements of the ring Z[ω] where ω = e^(iπ/4) = (1+i)/√2.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// Z[ω] = { aω³ + bω² + cω + d | a, b, c, d ∈ Z } is the ring of
/// cyclotomic integers of degree 8. It is a subring of the complex numbers
/// and a Euclidean domain.
///
/// Representation: stored as (a, b, c, d) representing aω³ + bω² + cω + d.
/// The real and imaginary parts are:
///   Re(u) = d + (c - a)√2/2
///   Im(u) = b + (c + a)√2/2
/// (See Lemma 5.5 and the proof of Lemma 5.6.)
///
/// Key operations from the paper:
///
/// - Complex conjugation (-)† maps (a,b,c,d) → (-c,-b,-a,d)
///   (Definition 3.2). Implemented as conj().
///
/// - √2-conjugation (-)● maps (a,b,c,d) → (-a,b,-c,d)
///   (Definition 3.2). Implemented as conj_sq2().
///   This sends √2 → -√2 while fixing i.
///
/// - The full norm N(u) = u†u · (u●)†(u●) is always a non-negative
///   integer (used for Euclidean division).
///
/// - Multiplication by ω is a cyclic permutation of coefficients
///   (exploited in mul_by_omega_power for efficient gate operations).
///
/// - The residue mod 2 (Definition D.1, Remark D.2) determines
///   divisibility by √2 and δ = 1 + ω, essential for the denominator
///   exponent reduction in the synthesis algorithm (Lemma 7.3).
///
/// Z[ω] ⊂ C is dense in C (Definition 3.1), and the complex grid
/// Grid(B) = { u ∈ Z[ω] | u● ∈ B } is discrete for bounded B
/// (Definition 5.1).
class ZOmega {
private:
  Integer _a, _b, _c, _d; // aω³ + bω² + cω + d

public:
  /// Construct aω³ + bω² + cω + d from integer coefficients.
  ///
  /// The first coefficient is the ω³ term, not the constant term.
  /// To embed an integer n, use ZOmega::from_int(n), which produces
  /// 0·ω³ + 0·ω² + 0·ω + n. A bare ZOmega(n) creates n·ω³, which is
  /// almost certainly not what is intended; hence the constructor is
  /// explicit to prevent silent implicit conversions from integers.
  explicit ZOmega(const Integer &a = 0, const Integer &b = 0,
                  const Integer &c = 0, const Integer &d = 0)
      : _a(a), _b(b), _c(c), _d(d) {}

  const Integer &a() const noexcept { return _a; }
  const Integer &b() const noexcept { return _b; }
  const Integer &c() const noexcept { return _c; }
  const Integer &d() const noexcept { return _d; }

  /// Embed integer x as 0·ω³ + 0·ω² + 0·ω + x ∈ Z[ω].
  static ZOmega from_int(const Integer &x) { return ZOmega(0, 0, 0, x); }

  /// Embed a + b√2 ∈ Z[√2] into Z[ω] via √2 = ω + ω⁻¹, yielding
  /// (-b)ω³ + 0·ω² + b·ω + a.
  ///
  /// The inverse (projection back) is ZSqrt2::from_zomega, defined in
  /// this header after ZOmega is complete (see below).
  static ZOmega from_zsqrt2(const ZSqrt2 &x) {
    return ZOmega(-x.b(), 0, x.b(), x.a());
  }

  bool operator==(const ZOmega &other) const {
    return _a == other._a && _b == other._b && _c == other._c && _d == other._d;
  }

  /// Equality with a ZSqrt2 element: checks directly that *this lies in the
  /// image of the ZSqrt2 → ZOmega embedding (_b == 0, _a == -_c) without
  /// constructing a temporary ZOmega.
  bool operator==(const ZSqrt2 &other) const {
    return _b == 0 && _a == -other.b() && _c == other.b() && _d == other.a();
  }

  ZOmega operator+(const ZOmega &other) const {
    return ZOmega(_a + other._a, _b + other._b, _c + other._c, _d + other._d);
  }

  ZOmega operator-(const ZOmega &other) const {
    return ZOmega(_a - other._a, _b - other._b, _c - other._c, _d - other._d);
  }

  ZOmega operator-() const { return ZOmega(-_a, -_b, -_c, -_d); }

  /// Ring multiplication in Z[ω] ≅ Z[x]/(x⁴+1).
  ///
  /// Uses ω⁴ = -1: polynomial multiplication mod x⁴+1, coefficient order
  /// (a=x³, b=x², c=x, d=1). Costs 16 GMP multiplications + 7 additions.
  ZOmega operator*(const ZOmega &other) const {
    const Integer &a0 = _a, &b0 = _b, &c0 = _c, &d0 = _d;
    const Integer &a1 = other._a, &b1 = other._b, &c1 = other._c,
                  &d1 = other._d;

    const Integer r0 = d0 * d1;
    const Integer r1 = d0 * c1 + c0 * d1;
    const Integer r2 = d0 * b1 + c0 * c1 + b0 * d1;
    const Integer r3 = d0 * a1 + c0 * b1 + b0 * c1 + a0 * d1;
    const Integer r4 = c0 * a1 + b0 * b1 + a0 * c1;
    const Integer r5 = b0 * a1 + a0 * b1;
    const Integer r6 = a0 * a1;

    return ZOmega(r3, r2 - r6, r1 - r5, r0 - r4);
  }

  /// Scalar multiplication: scale all coefficients by x.
  ZOmega operator*(const Integer &x) const {
    return ZOmega(_a * x, _b * x, _c * x, _d * x);
  }

  // ---------------------------------------------------------------------------
  // Ring automorphisms
  // ---------------------------------------------------------------------------

  /// Complex conjugation (-)†: (aω³+bω²+cω+d)† = -cω³-bω²-aω+d.
  /// Paper Definition 3.2. Maps ω → ω⁻¹.
  ZOmega conj() const { return ZOmega(-_c, -_b, -_a, _d); }

  /// √2-conjugation (-)●: (aω³+bω²+cω+d)● = -aω³+bω²-cω+d.
  /// Paper Definition 3.2. Maps √2 → -√2 while fixing i.
  /// Central to the grid problem formulation (Definition 5.1):
  /// Grid(B) = { u ∈ Z[ω] | u● ∈ B }.
  ZOmega conj_sq2() const { return ZOmega(-_a, _b, -_c, _d); }

  // ---------------------------------------------------------------------------
  // Derived properties
  // ---------------------------------------------------------------------------

  /// Full norm N(u) = u†u · (u●)†(u●), always a non-negative integer
  /// (Remark 3.3). Used as the Euclidean function for divmod in Z[ω].
  /// Costs 6 GMP multiplications.
  Integer norm() const {
    Integer sum_squares = _a * _a + _b * _b + _c * _c + _d * _d;
    Integer cross_term = _a * _b + _b * _c + _c * _d - _d * _a;
    return sum_squares * sum_squares - 2 * cross_term * cross_term;
  }

  /// 4-bit parity pattern (a%2, b%2, c%2, d%2) packed into a native int.
  ///
  /// Result is always in [0, 15]. Determines divisibility by δ = 1+ω:
  /// an element is divisible by δ iff a+b+c+d is even (Definition D.1,
  /// Remark D.2, Lemma 7.3). Uses Integer::is_odd() — a single GMP bit
  /// test per field, no heap allocation.
  i32 residue() const {
    return (i32(_a.is_odd()) << 3) | (i32(_b.is_odd()) << 2) |
           (i32(_c.is_odd()) << 1) | i32(_d.is_odd());
  }

  /// Returns the element as "(a, b, c, d)" — the four integer coefficients
  /// of a*ω³ + b*ω² + c*ω + d. Intended for logging and debugging.
  std::string to_string() const {
    return "(" + _a.to_string() + ", " + _b.to_string() + ", " +
           _c.to_string() + ", " + _d.to_string() + ")";
  }

  Real real() const { return Real(d()) + Real::sqrt2() * Real(c() - a()) / 2; }

  Real imag() const { return Real(b()) + Real::sqrt2() * Real(c() + a()) / 2; }
};

// ---------------------------------------------------------------------------
// ZSqrt2::from_zomega — defined here because ZOmega must be complete.
// Declared in zsqrt2.h; the definition must follow ZOmega's definition to
// break the mutual-dependency cycle between the two ring types.
// ---------------------------------------------------------------------------

/// Project x ∈ Z[ω] to Z[√2], asserting that x actually lies in the
/// Z[√2] subring (i.e. x.b() == 0 and x.a() == -x.c()).
inline ZSqrt2 ZSqrt2::from_zomega(const ZOmega &x) {
  assert(x.b() == 0 && x.a() == -x.c());
  return ZSqrt2(x.d(), x.c());
}

// ---------------------------------------------------------------------------
// Free functions on ZOmega
// ---------------------------------------------------------------------------

/// Return x · ω = e^(iπ/4) · x.
///
/// Multiplication by ω is a cyclic left-shift of the (a,b,c,d) coefficients
/// with a sign flip on the wrapped coefficient:
///   (a,b,c,d) → (b,c,d,-a).
/// This is equivalent to ω⁴ = -1 combined with the basis representation.
inline ZOmega mul_by_omega(const ZOmega &x) {
  return ZOmega(x.b(), x.c(), x.d(), -x.a());
}

/// Return x · ω⁻¹ = e^(-iπ/4) · x.
///
/// Inverse cyclic shift with sign flip:
///   (a,b,c,d) → (-d,a,b,c).
inline ZOmega mul_by_omega_inv(const ZOmega &x) {
  return ZOmega(-x.d(), x.a(), x.b(), x.c());
}

/// Return x · ωⁿ for any integer n (reduced modulo 8 internally).
///
/// Uses a direct switch rather than repeated mul_by_omega calls to avoid
/// n intermediate allocations. n is reduced mod 8 using a bitmask since
/// ω⁸ = 1.
inline ZOmega mul_by_omega_power(const ZOmega &x, i32 n) {
  n &= 0b111;
  switch (n) {
  case 0:
    return x;
  case 1:
    return mul_by_omega(x); // (b,c,d,-a)
  case 2:
    return ZOmega(x.c(), x.d(), -x.a(), -x.b());
  case 3:
    return ZOmega(x.d(), -x.a(), -x.b(), -x.c());
  case 4:
    return ZOmega(-x.a(), -x.b(), -x.c(), -x.d());
  case 5:
    return ZOmega(-x.b(), -x.c(), -x.d(), x.a());
  case 6:
    return ZOmega(-x.c(), -x.d(), x.a(), x.b());
  case 7: // same as mul_by_omega_inv
    return ZOmega(-x.d(), x.a(), x.b(), x.c());
  default:
    return x;
  }
}

/// Multiplicative inverse of x, valid only for units (elements with norm 1).
///
/// Returns failure() if x is not a unit. In Z[ω] the units are the 8th roots
/// of unity {±1, ±ω, ±ω², ±ω³}; the inverse formula follows from
///   x⁻¹ = x● · (x†·(x†)●) / N(x)
/// which reduces to conj_sq2 * c * c.conj_sq2() when N(x) = 1.
/// conj() is cached to avoid constructing it twice.
inline FailureOr<ZOmega> inv(const ZOmega &x) {
  if (x.norm() == 1) {
    ZOmega c = x.conj();
    return x.conj_sq2() * c * c.conj_sq2();
  }
  return failure();
}

/// Decompose x into floating-point (real, imaginary) parts.
///
/// Implements Lemma 5.5:
///   Re(u) = d + (c − a) · √2/2
///   Im(u) = b + (c + a) · √2/2
///
/// Callers that process many elements with the same denominator exponent
/// should prefer coords_into (defined in domega.h) which amortises the
/// √2/2 computation across calls.
inline void to_real_imag(const ZOmega &x, Real &out_real, Real &out_imag) {
  out_real = Real(x.d()) + Real::sqrt2() * Real(x.c() - x.a()) / 2;
  out_imag = Real(x.b()) + Real::sqrt2() * Real(x.c() + x.a()) / 2;
}

/// Decompose x into floating-point (real, imaginary) parts.
///
/// Implements Lemma 5.5:
///   Re(u) = d + (c − a) · √2/2
///
/// Callers that process many elements with the same denominator exponent
/// should prefer coords_into (defined in domega.h) which amortises the
/// √2/2 computation across calls.
inline void to_real(const ZOmega &x, Real &out_real) {
  out_real = Real(x.d()) + Real::sqrt2() * Real(x.c() - x.a()) / 2;
}

/// Convert x to floating-point complex form.
///
/// Delegates to to_real_imag to avoid duplicating the floating-point
/// arithmetic. Prefer to_real_imag when the real and imaginary parts are
/// needed separately (e.g. for matrix construction in GridOp::to_mat).
inline std::complex<Real> to_complex(const ZOmega &x) {
  Real r, i;
  to_real_imag(x, r, i);
  return {r, i};
}

/// Division with remainder in Z[ω] (Euclidean domain).
///
/// Computes (q, r) such that x = y·q + r and N(r) < N(y). The quotient is
/// obtained by rounding the exact rational quotient to the nearest lattice
/// point.
///
/// The adjoint product x · y† · (y†)● · y● is computed with y.conj()
/// cached to avoid constructing it twice (it is needed both directly and
/// as the base for conj_sq2()).
inline std::pair<ZOmega, ZOmega> divmod(const ZOmega &x, const ZOmega &y) {
  ZOmega yc = y.conj();
  ZOmega p = x * yc * yc.conj_sq2() * y.conj_sq2();
  Integer k = y.norm();
  ZOmega q(rounddiv(p.a(), k), rounddiv(p.b(), k), rounddiv(p.c(), k),
           rounddiv(p.d(), k));
  ZOmega r = x - y * q;
  return {q, r};
}

/// GCD in Z[ω] via the Euclidean algorithm.
///
/// Z[ω] is a Euclidean domain with norm N, so the algorithm terminates.
/// The result is unique only up to multiplication by units.
inline ZOmega gcd(ZOmega a, ZOmega b) {
  const ZOmega zero(0, 0, 0, 0);
  while (!(b == zero)) {
    auto r = divmod(a, b).second;
    a = b;
    b = r;
  }
  return a;
}

} // namespace cudaq::synth
