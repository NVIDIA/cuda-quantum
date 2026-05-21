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
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <utility>

namespace cudaq::synth {

class ZOmega;

//===----------------------------------------------------------------------===//
// ZSqrt2
//===----------------------------------------------------------------------===//

/// Elements of the ring Z[sqrt(2)] = { a + b*sqrt(2) | a, b in Z }.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// Z[sqrt(2)] is a Euclidean domain, so unique factorization, GCD, and
/// division with remainder are all well-defined.
///
/// Key facts used downstream:
///   - sqrt(2)-conjugation `conj_sq2`: a + b*sqrt(2) -> a - b*sqrt(2)
///     (Definition 3.2).
///   - Norm N(alpha) = alpha * conj_sq2(alpha) = a^2 - 2*b^2 is an integer
///     (Remark 3.3); N is multiplicative.
///   - lambda = 1 + sqrt(2) is the fundamental unit: inv(lambda) =
///     -conj_sq2(lambda) = -1 + sqrt(2) (Remark 3.6), and every unit has
///     the form +/-lambda^m (Lemma C.2).
///   - Grid separation: for distinct alpha, beta in Z[sqrt(2)],
///     |alpha - beta| * |conj_sq2(alpha) - conj_sq2(beta)| >= 1
///     (Remark 3.3), so the real grid for any bounded interval is discrete.
///   - Z[sqrt(2)] is dense in R, which guarantees solutions to grid problems
///     when the intervals are wide enough.
class ZSqrt2 {
private:
  Integer _a;
  Integer _b;

public:
  /// Construct a + b*sqrt(2) from integer coefficients.
  ///
  /// Unlike ZOmega (where ZOmega(n) creates n*omega^3), ZSqrt2(n) correctly
  /// embeds the integer n as n + 0*sqrt(2). The constructor is still
  /// explicit for symmetry with the other ring types and to prevent
  /// accidental Integer -> ZSqrt2 conversions.
  explicit ZSqrt2(const Integer &a = 0, const Integer &b = 0) : _a(a), _b(b) {}

  const Integer &a() const { return _a; }
  const Integer &b() const { return _b; }

  /// In-place assignment that reuses the existing mpz_t buffers via
  /// Integer::operator= (mpz_set-based). Saves the mpz_init / mpz_clear
  /// pair that `*this = ZSqrt2(a, b)` would pay for the temporary.
  void assign(const Integer &a, const Integer &b) {
    _a = a;
    _b = b;
  }

  /// In-place assignment that swaps mpz_t ownership with the source
  /// Integers via Integer::operator=(Integer&&) (mpz_swap-based).
  void assign(Integer &&a, Integer &&b) noexcept {
    _a = std::move(a);
    _b = std::move(b);
  }

  /// Project a ZOmega element onto Z[sqrt(2)], asserting the input lies in
  /// the subring. Defined in zomega.h once ZOmega is complete (mutual
  /// dependency).
  static ZSqrt2 from_zomega(const ZOmega &x);

  /// Fundamental unit lambda = 1 + sqrt(2).
  ///
  /// lambda is the smallest unit greater than 1 (Remark 3.6); every unit
  /// of Z[sqrt(2)] has the form +/-lambda^m (m in Z, Lemma C.2). The
  /// sqrt(2)-conjugate is conj_sq2(lambda) = 1 - sqrt(2), and lambda *
  /// conj_sq2(lambda) = -1, so inv(lambda) = -conj_sq2(lambda) = -1 +
  /// sqrt(2). lambda drives the shift operators in the Step Lemma
  /// (Definition A.6) and sets the rescaling factor of the ODGP.
  ///
  /// Function-local static dodges static-initialization-order issues, same
  /// pattern as Real::pi() and Real::sqrt2().
  static const ZSqrt2 &lambda() {
    static const ZSqrt2 value(1, 1);
    return value;
  }

  bool operator==(const ZSqrt2 &other) const {
    return _a == other._a && _b == other._b;
  }

  bool operator!=(const ZSqrt2 &other) const { return !(*this == other); }

  /// Compare by real value: a + b*sqrt(2) < c + d*sqrt(2) as real numbers.
  /// Reduces to comparing (a-c)^2 vs 2*(b-d)^2 with a sign analysis, all in
  /// exact integer arithmetic (no floating point).
  bool operator<(const ZSqrt2 &other) const {
    Integer diff_a = _a - other._a;
    Integer diff_b = _b - other._b;

    if (diff_b == 0)
      return diff_a < 0;

    if (diff_b > 0) {
      if (diff_a >= 0)
        return false;
      Integer lhs = diff_a * diff_a;
      Integer rhs = diff_b * diff_b << 1;
      return lhs > rhs;
    }

    if (diff_a < 0)
      return true;
    Integer lhs = diff_a * diff_a;
    Integer rhs = diff_b * diff_b << 1;
    return lhs < rhs;
  }

  bool operator<=(const ZSqrt2 &other) const { return !(other < *this); }
  bool operator>(const ZSqrt2 &other) const { return other < *this; }
  bool operator>=(const ZSqrt2 &other) const { return !(*this < other); }

  ZSqrt2 operator+(const ZSqrt2 &other) const {
    return ZSqrt2(_a + other._a, _b + other._b);
  }

  ZSqrt2 operator-(const ZSqrt2 &other) const {
    return ZSqrt2(_a - other._a, _b - other._b);
  }

  ZSqrt2 operator-() const { return ZSqrt2(-_a, -_b); }

  /// Ring multiplication:
  ///     (a + b*sqrt(2)) * (c + d*sqrt(2)) = (a*c + 2*b*d) + (a*d + b*c)*sqrt(2)
  ZSqrt2 operator*(const ZSqrt2 &other) const {
    return ZSqrt2(_a * other._a + (_b * other._b << 1),
                  _a * other._b + _b * other._a);
  }

  // -- Ring automorphisms --

  /// sqrt(2)-conjugation: (a + b*sqrt(2))* = a - b*sqrt(2). Maps sqrt(2)
  /// to -sqrt(2) (Paper Definition 3.2). Used in grid problems and the
  /// Diophantine solver.
  ZSqrt2 conj_sq2() const { return ZSqrt2(_a, -_b); }

  // -- Derived properties --

  /// Parity of the constant coefficient: 0 or 1.
  i32 parity() const { return i32(_a.is_odd()); }

  /// "(a, b)" rendering for debug logging.
  std::string to_string() const {
    return "(" + _a.to_string() + ", " + _b.to_string() + ")";
  }

  /// Algebraic norm N(alpha) = alpha * conj_sq2(alpha) = a^2 - 2*b^2.
  ///
  /// Reference: Paper sec. 3 Remark 3.3; Appendix C, passim. Used for unit
  /// detection (|N| = 1), primality testing (Lemma C.4), and the
  /// Diophantine equation reduction (Proposition C.24).
  Integer norm() const { return _a * _a - (_b * _b << 1); }
};

//===----------------------------------------------------------------------===//
// Free functions on ZSqrt2
//===----------------------------------------------------------------------===//

/// Convert x to a floating-point Real. Multi-precision conversion: callers
/// that process many elements with the same denominator exponent should
/// prefer `to_real(DSqrt2)`, which amortizes the scale computation.
inline Real to_real(const ZSqrt2 &x) {
  return Real(x.a()) + Real::sqrt2() * Real(x.b());
}

/// Multiplicative inverse, defined only for units. The units of Z[sqrt(2)]
/// are { +/-lambda^m | m in Z } where lambda = 1 + sqrt(2) (Lemma C.2).
/// Returns failure() if x is not a unit.
inline llvm::FailureOr<ZSqrt2> inv(const ZSqrt2 &x) {
  Integer n = x.norm();
  // N = 1: x * x* = 1, so inv(x) = x*.
  // N = -1: x * (-x*) = 1, so inv(x) = -x*.
  if (n == 1)
    return x.conj_sq2();
  if (n == -1)
    return -x.conj_sq2();
  return llvm::failure();
}

/// Integer power x^exp via binary exponentiation. Negative exponents
/// require x to be a unit; the precondition is asserted.
inline ZSqrt2 pow(const ZSqrt2 &x, Integer exp) {
  if (exp < 0) {
    llvm::FailureOr<ZSqrt2> inv_or = inv(x);
    assert(llvm::succeeded(inv_or) && "ZSqrt2::pow: element is not a unit");
    return pow(*inv_or, -exp);
  }

  ZSqrt2 result(1, 0);
  ZSqrt2 base = x;
  while (exp > 0) {
    if (exp.is_odd())
      result = result * base;
    base = base * base;
    exp >>= 1;
  }
  return result;
}

/// Square root in Z[sqrt(2)] if x is a perfect square, otherwise failure().
///
/// Candidate roots are recovered from floor(sqrt) of the rational quantities
/// (x.a() +/- r)/2 and (x.a() +/- r)/4 with r = floor(sqrt(N(x))); the
/// candidates are then squared and compared exactly.
inline llvm::FailureOr<ZSqrt2> sqrt(const ZSqrt2 &x) {
  Integer n = x.norm();
  if (n < 0 || x.a() < 0)
    return llvm::failure();

  Integer r = floorsqrt(n);
  Integer a1 = floorsqrt(floordiv(x.a() + r, 2));
  Integer b1 = floorsqrt(floordiv(x.a() - r, 4));
  Integer a2 = floorsqrt(floordiv(x.a() - r, 2));
  Integer b2 = floorsqrt(floordiv(x.a() + r, 4));

  bool positive = (sign(x.a()) * sign(x.b())) >= 0;
  ZSqrt2 w1 = positive ? ZSqrt2(a1, b1) : ZSqrt2(a1, -b1);
  ZSqrt2 w2 = positive ? ZSqrt2(a2, b2) : ZSqrt2(a2, -b2);

  if (x == w1 * w1)
    return w1;
  if (x == w2 * w2)
    return w2;
  return llvm::failure();
}

/// Euclidean division: returns (q, r) with x = y*q + r and N(r) < N(y).
inline std::pair<ZSqrt2, ZSqrt2> divmod(const ZSqrt2 &x, const ZSqrt2 &y) {
  ZSqrt2 p = x * y.conj_sq2();
  Integer k = y.norm();
  ZSqrt2 q(rounddiv(p.a(), k), rounddiv(p.b(), k));
  ZSqrt2 r = x - y * q;
  return {q, r};
}

/// Quotient of Euclidean division.
inline ZSqrt2 operator/(const ZSqrt2 &x, const ZSqrt2 &y) {
  return divmod(x, y).first;
}

/// Remainder of Euclidean division.
inline ZSqrt2 operator%(const ZSqrt2 &x, const ZSqrt2 &y) {
  return divmod(x, y).second;
}

/// GCD via the Euclidean algorithm. The result is unique only up to
/// multiplication by units (+/-lambda^m).
inline ZSqrt2 gcd(ZSqrt2 a, ZSqrt2 b) {
  const ZSqrt2 zero(0, 0);
  while (!(b == zero)) {
    auto r = divmod(a, b).second;
    a = b;
    b = r;
  }
  return a;
}

/// True iff a and b are associates -- equivalently, a = u*b for some unit
/// u in Z[sqrt(2)]^x.
inline bool are_associates(const ZSqrt2 &a, const ZSqrt2 &b) {
  const ZSqrt2 zero(0, 0);
  return (a % b == zero) && (b % a == zero);
}

} // namespace cudaq::synth
