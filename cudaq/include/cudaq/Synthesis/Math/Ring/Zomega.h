/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <complex>
#include <string>

#include "cudaq/Synthesis/Math/Integer.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"
#include "llvm/Support/LogicalResult.h"

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// ZOmega
//===----------------------------------------------------------------------===//

/// Elements of the ring Z[omega] where omega = e^(i*pi/4) = (1+i)/sqrt(2).
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// Z[omega] = { a*omega^3 + b*omega^2 + c*omega + d | a, b, c, d in Z } is
/// the ring of cyclotomic integers of degree 8: a subring of C and a
/// Euclidean domain.
///
/// Representation. Stored as the four-tuple (a, b, c, d). The real and
/// imaginary parts decompose as (Lemma 5.5)
///     Re(u) = d + (c - a) * sqrt(2)/2
///     Im(u) = b + (c + a) * sqrt(2)/2
///
/// Key operations used downstream:
///   - Complex conjugation (Definition 3.2):
///       conj((a, b, c, d)) = (-c, -b, -a, d)
///   - sqrt(2)-conjugation (Definition 3.2), sends sqrt(2) to -sqrt(2)
///     while fixing i:
///       conj_sq2((a, b, c, d)) = (-a, b, -c, d)
///   - Full norm N(u) = conj(u)*u * conj_sq2(conj(u))*conj_sq2(u) is a
///     non-negative integer (Euclidean function for divmod).
///   - Multiplication by omega is a cyclic permutation of the coefficients,
///     exploited by mul_by_omega_power for cheap gate operations.
///   - The 4-bit residue mod 2 (Definition D.1, Remark D.2) detects
///     divisibility by sqrt(2) and by delta = 1 + omega, which drives the
///     denominator-exponent reduction in Lemma 7.3.
///
/// Z[omega] is dense in C, and the complex grid
/// Grid(B) = { u in Z[omega] | conj_sq2(u) in B } is discrete for bounded
/// B (Definition 5.1).
class ZOmega {
private:
  Integer _a, _b, _c, _d; // a*omega^3 + b*omega^2 + c*omega + d

public:
  /// Construct a*omega^3 + b*omega^2 + c*omega + d.
  ///
  /// The first coefficient is the omega^3 term, not the constant term. To
  /// embed a plain integer n, use ZOmega::from_int(n) -- a bare
  /// ZOmega(n) creates n*omega^3, which is almost never what you want.
  /// The constructor is explicit to prevent silent Integer -> ZOmega
  /// conversions.
  explicit ZOmega(const Integer &a = 0, const Integer &b = 0,
                  const Integer &c = 0, const Integer &d = 0)
      : _a(a), _b(b), _c(c), _d(d) {}

  const Integer &a() const noexcept { return _a; }
  const Integer &b() const noexcept { return _b; }
  const Integer &c() const noexcept { return _c; }
  const Integer &d() const noexcept { return _d; }

  /// Embed integer x as 0*omega^3 + 0*omega^2 + 0*omega + x.
  static ZOmega from_int(const Integer &x) { return ZOmega(0, 0, 0, x); }

  /// Embed a + b*sqrt(2) in Z[sqrt(2)] into Z[omega] via the identity
  /// sqrt(2) = omega + omega^-1, yielding (-b)*omega^3 + 0*omega^2 +
  /// b*omega + a. The inverse projection is ZSqrt2::from_zomega, defined
  /// after ZOmega is complete to break the mutual-dependency cycle.
  static ZOmega from_zsqrt2(const ZSqrt2 &x) {
    return ZOmega(-x.b(), 0, x.b(), x.a());
  }

  bool operator==(const ZOmega &other) const {
    return _a == other._a && _b == other._b && _c == other._c && _d == other._d;
  }

  /// Direct equality test against a ZSqrt2 element: checks the
  /// embedding-image conditions (b == 0, a == -c) without constructing a
  /// temporary ZOmega.
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

  /// Ring multiplication in Z[omega] ~= Z[x] / (x^4 + 1).
  ///
  /// Uses omega^4 = -1: polynomial multiplication modulo x^4 + 1, in
  /// coefficient order (a = x^3, b = x^2, c = x, d = 1). Cost: 16 GMP
  /// multiplications and 7 additions.
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

  //===--------------------------------------------------------------------===//
  // Ring automorphisms
  //===--------------------------------------------------------------------===//

  /// Complex conjugation (Paper Definition 3.2). Maps omega to omega^-1:
  ///     conj(a*omega^3 + b*omega^2 + c*omega + d)
  ///       = -c*omega^3 - b*omega^2 - a*omega + d
  ZOmega conj() const { return ZOmega(-_c, -_b, -_a, _d); }

  /// sqrt(2)-conjugation (Paper Definition 3.2). Maps sqrt(2) to -sqrt(2)
  /// while fixing i:
  ///     conj_sq2(a*omega^3 + b*omega^2 + c*omega + d)
  ///       = -a*omega^3 + b*omega^2 - c*omega + d
  /// Central to the grid problem formulation
  ///     Grid(B) = { u in Z[omega] | conj_sq2(u) in B }   (Definition 5.1).
  ZOmega conj_sq2() const { return ZOmega(-_a, _b, -_c, _d); }

  //===--------------------------------------------------------------------===//
  // Derived properties
  //===--------------------------------------------------------------------===//

  /// Full norm N(u) = conj(u) * u * conj_sq2(conj(u)) * conj_sq2(u), always
  /// a non-negative integer (Remark 3.3). Serves as the Euclidean function
  /// for divmod. Costs 6 GMP multiplications via the squared-sum + cross-
  /// term identity below.
  Integer norm() const {
    Integer sum_squares = _a * _a + _b * _b + _c * _c + _d * _d;
    Integer cross_term = _a * _b + _b * _c + _c * _d - _d * _a;
    return sum_squares * sum_squares - 2 * cross_term * cross_term;
  }

  /// 4-bit parity pattern (a%2, b%2, c%2, d%2) packed into a native int
  /// (range [0, 15]). Drives divisibility by delta = 1 + omega: an element
  /// is divisible by delta iff a + b + c + d is even (Definition D.1,
  /// Remark D.2, Lemma 7.3). Uses Integer::is_odd which is a single GMP
  /// bit test per field -- no heap allocation.
  int32_t residue() const {
    return (int32_t(_a.is_odd()) << 3) | (int32_t(_b.is_odd()) << 2) |
           (int32_t(_c.is_odd()) << 1) | int32_t(_d.is_odd());
  }

  /// "(a, b, c, d)" rendering for debug logging.
  std::string to_string() const {
    return "(" + _a.to_string() + ", " + _b.to_string() + ", " +
           _c.to_string() + ", " + _d.to_string() + ")";
  }

  Real real() const { return Real(d()) + Real::sqrt2() * Real(c() - a()) / 2; }

  Real imag() const { return Real(b()) + Real::sqrt2() * Real(c() + a()) / 2; }
};

//===----------------------------------------------------------------------===//
// ZSqrt2::from_zomega -- out of line because ZOmega must be complete
//===----------------------------------------------------------------------===//

/// Project x in Z[omega] onto Z[sqrt(2)], asserting that x actually lies in
/// the subring (x.b() == 0 and x.a() == -x.c()).
inline ZSqrt2 ZSqrt2::from_zomega(const ZOmega &x) {
  assert(x.b() == 0 && x.a() == -x.c());
  return ZSqrt2(x.d(), x.c());
}

//===----------------------------------------------------------------------===//
// Free functions on ZOmega
//===----------------------------------------------------------------------===//

/// Multiplication by omega = e^(i*pi/4). In the coefficient basis this is a
/// cyclic left-shift with a sign flip on the wrapped coefficient:
///     (a, b, c, d) -> (b, c, d, -a)
/// which is exactly the action implied by omega^4 = -1.
inline ZOmega mul_by_omega(const ZOmega &x) {
  return ZOmega(x.b(), x.c(), x.d(), -x.a());
}

/// Multiplication by omega^-1 = e^(-i*pi/4):
///     (a, b, c, d) -> (-d, a, b, c)
inline ZOmega mul_by_omega_inv(const ZOmega &x) {
  return ZOmega(-x.d(), x.a(), x.b(), x.c());
}

/// Multiplication by omega^n. Direct switch over n mod 8 (omega^8 = 1) to
/// avoid building up |n| intermediate allocations from repeated single-step
/// applications.
inline ZOmega mul_by_omega_power(const ZOmega &x, int32_t n) {
  n &= 0b111;
  switch (n) {
  case 0:
    return x;
  case 1:
    return mul_by_omega(x); // (b, c, d, -a)
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
  case 7: // == mul_by_omega_inv(x)
    return ZOmega(-x.d(), x.a(), x.b(), x.c());
  default:
    return x;
  }
}

/// Multiplicative inverse, defined only for units. The units of Z[omega]
/// are the eighth roots of unity { +/-1, +/-omega, +/-omega^2, +/-omega^3 }.
///
/// For a unit x (N(x) = 1) the inverse follows from
///     inv(x) = conj_sq2(x) * conj(x) * conj_sq2(conj(x))
/// which reduces to the closed form below. We cache conj(x) so it is only
/// constructed once.
inline llvm::FailureOr<ZOmega> inv(const ZOmega &x) {
  if (x.norm() == 1) {
    ZOmega c = x.conj();
    return x.conj_sq2() * c * c.conj_sq2();
  }
  return llvm::failure();
}

/// Decompose x into floating-point (real, imaginary) coordinates via
/// Lemma 5.5:
///     Re(u) = d + (c - a) * sqrt(2)/2
///     Im(u) = b + (c + a) * sqrt(2)/2
///
/// Callers that process many elements with the same denominator exponent
/// should prefer `coords_into` (defined in domega.h), which amortises the
/// sqrt(2)/2 computation across calls.
inline void to_real_imag(const ZOmega &x, Real &out_real, Real &out_imag) {
  out_real = Real(x.d()) + Real::sqrt2() * Real(x.c() - x.a()) / 2;
  out_imag = Real(x.b()) + Real::sqrt2() * Real(x.c() + x.a()) / 2;
}

/// Same as `to_real_imag` but only the real part is needed. Skips one
/// Real allocation. Same amortisation note applies: prefer `coords_into`
/// when batching.
inline void to_real(const ZOmega &x, Real &out_real) {
  out_real = Real(x.d()) + Real::sqrt2() * Real(x.c() - x.a()) / 2;
}

/// Convert x to floating-point complex form. Delegates to `to_real_imag`.
/// Prefer the two-output variant when you only need the real or only the
/// imaginary part, or when you have multiple elements to convert
/// (e.g. matrix construction in GridOp::to_mat).
inline std::complex<Real> to_complex(const ZOmega &x) {
  Real r, i;
  to_real_imag(x, r, i);
  return {r, i};
}

/// Division with remainder in Z[omega]. Returns (q, r) with x = y*q + r and
/// N(r) < N(y). The quotient is obtained by rounding the exact rational
/// quotient to the nearest lattice point.
///
/// y.conj() is cached because it is needed both directly and as the base
/// for conj_sq2 -- one fewer ZOmega construction in the hot path.
inline std::pair<ZOmega, ZOmega> divmod(const ZOmega &x, const ZOmega &y) {
  ZOmega yc = y.conj();
  ZOmega p = x * yc * yc.conj_sq2() * y.conj_sq2();
  Integer k = y.norm();
  ZOmega q(rounddiv(p.a(), k), rounddiv(p.b(), k), rounddiv(p.c(), k),
           rounddiv(p.d(), k));
  ZOmega r = x - y * q;
  return {q, r};
}

/// GCD via the Euclidean algorithm. Z[omega] is a Euclidean domain with
/// norm N, so this terminates. The result is unique only up to
/// multiplication by units.
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
