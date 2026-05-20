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

/// Elements of the ring Z[√2] = { a + b√2 | a, b ∈ Z }.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, Definition 3.1.
///
/// This is the ring of "quadratic integers with `radicand` 2". It is a
/// Euclidean domain, so unique factorization, GCD, and division with remainder
/// are well-defined.
///
/// Key algebraic operations from the paper:
///
/// - The √2-conjugation `automorphism` (-)^● maps a + b√2 → a - b√2
///   (Definition 3.2). Implemented as conj_sq2().
///
/// - The norm N(α) = α · α^● = a² - 2b² is an integer (Remark 3.3).
///   This satisfies N(αβ) = N(α)N(β).
///
/// - The element λ = 1 + √2 is a fundamental unit with λ^(-1) = -λ^●
///   = -1 + √2 (Remark 3.6). All units are ±λ^m (Lemma C.2).
///
/// - Grid constraint separation: for distinct α, β ∈ Z[√2],
///   |α - β| · |α^● - β^●| ≥ 1 (Remark 3.3). This guarantees that
///   the real grid for a bounded interval is discrete.
///
/// - Z[√2] is dense in R (Definition 3.1), which ensures the existence
///   of solutions to grid problems for sufficiently large intervals.
///
class ZSqrt2 {
private:
  Integer _a;
  Integer _b;

public:
  /// Construct a + b√2 from integer coefficients.
  ///
  /// Unlike ZOmega (where ZOmega(n) creates n·ω³), ZSqrt2(n) correctly
  /// embeds the integer n as n + 0·√2. The constructor is nonetheless
  /// explicit for consistency with the other ring types (DOmega, DSqrt2,
  /// ZOmega) and to prevent accidental implicit Integer→ZSqrt2 conversions.
  explicit ZSqrt2(const Integer &a = 0, const Integer &b = 0) : _a(a), _b(b) {}

  const Integer &a() const { return _a; }
  const Integer &b() const { return _b; }

  /// In-place assignment that reuses the existing `mpz_t` buffers via
  /// `Integer::operator=` (which is `mpz_set`-based). Equivalent to
  /// `*this = ZSqrt2(a, b)` but avoids the `mpz_init` / `mpz_clear` pair the
  /// temporary would otherwise pay.
  void assign(const Integer &a, const Integer &b) {
    _a = a;
    _b = b;
  }

  /// In-place assignment that swaps `mpz_t` ownership with the source
  /// Integers via `Integer::operator=(Integer &&)` (mpz_swap-based).
  void assign(Integer &&a, Integer &&b) noexcept {
    _a = std::move(a);
    _b = std::move(b);
  }

  /// Project a ZOmega element onto Z[√2], asserting it lies in the `subring`.
  /// Defined in zomega.h after ZOmega is complete (mutual dependency).
  static ZSqrt2 from_zomega(const ZOmega &x);

  /// The fundamental unit λ = 1 + √2 of Z[√2].
  ///
  /// λ is the smallest unit greater than 1 (Remark 3.6); all units of Z[√2]
  /// are of the form ±λ^m (m ∈ Z, Lemma C.2). Its conjugate λ● = 1 - √2
  /// satisfies λ · λ● = -1 (norm = -1), and λ^(-1) = -λ● = -1 + √2.
  ///
  /// λ generates the shift operators used in the Step Lemma (Appendix A,
  /// Definition A.6) and governs the scale of grid intervals in the ODGP.
  ///
  /// Implemented as a function-local static to avoid static initialization
  /// order issues (same pattern as Real::pi() and Real::sqrt2()).
  static const ZSqrt2 &lambda() {
    static const ZSqrt2 value(1, 1);
    return value;
  }

  bool operator==(const ZSqrt2 &other) const {
    return _a == other._a && _b == other._b;
  }

  bool operator!=(const ZSqrt2 &other) const { return !(*this == other); }

  /// Compare by real value: a + b√2 < c + d√2 as real numbers.
  ///
  /// Reduces to comparing (a-c)² vs 2(b-d)² with a sign analysis to
  /// determine the direction, avoiding floating-point evaluation.
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

  /// Ring multiplication: (a + b√2)(c + d√2) = (ac + 2`bd`) + (ad + `bc`)√2.
  ZSqrt2 operator*(const ZSqrt2 &other) const {
    return ZSqrt2(_a * other._a + (_b * other._b << 1),
                  _a * other._b + _b * other._a);
  }

  // ---------------------------------------------------------------------------
  // Ring `automorphism`s
  // ---------------------------------------------------------------------------

  /// √2-conjugation (-)●: (a + b√2)● = a - b√2.
  /// Paper Definition 3.2: `automorphism` mapping √2 → -√2.
  /// Used in grid problems (Definition 4.1) and the Diophantine solver.
  ZSqrt2 conj_sq2() const { return ZSqrt2(_a, -_b); }

  // ---------------------------------------------------------------------------
  // Derived properties
  // ---------------------------------------------------------------------------

  /// Parity of the constant coefficient a mod 2, as a native int.
  /// Result is always 0 or 1.
  i32 parity() const { return i32(_a.is_odd()); }

  /// Returns the element as "(a, b)" where a and b are the integer coefficients
  /// of a + b*`sqrt`(2). Intended for logging and debugging.
  std::string to_string() const {
    return "(" + _a.to_string() + ", " + _b.to_string() + ")";
  }

  /// Algebraic norm N(α) = α · α● = a² - 2b² ∈ Z.
  ///
  /// Paper §3, Remark 3.3; Appendix C, passim.
  /// Used for: unit detection (|N| = 1), `primality` testing (Lemma C.4),
  /// and the Diophantine equation reduction (Proposition C.24).
  Integer norm() const { return _a * _a - (_b * _b << 1); }
};

// ---------------------------------------------------------------------------
// Free functions on ZSqrt2
// ---------------------------------------------------------------------------

/// Convert x to a floating-point Real: a + b√2 → Real(a) + Real(√2)·Real(b).
///
/// The result is a multi-precision float evaluated at runtime precision.
/// Callers processing many elements with the same denominator exponent should
/// prefer to_real(DSqrt2) which amortizes the scale computation.
inline Real to_real(const ZSqrt2 &x) {
  return Real(x.a()) + Real::sqrt2() * Real(x.b());
}

/// Multiplicative inverse of x, valid only for units (elements with |norm|=1).
///
/// For Z[√2], the units are {±λ^m | m ∈ Z} where λ = 1 + √2 (Lemma C.2).
/// Returns failure() if x is not a unit.
/// N = 1:  x⁻¹ = x● (since x·x● = N = 1).
/// N = -1: x⁻¹ = -x● (since x·(-x●) = -N = 1).
inline llvm::FailureOr<ZSqrt2> inv(const ZSqrt2 &x) {
  Integer n = x.norm();
  if (n == 1)
    return x.conj_sq2();
  if (n == -1)
    return -x.conj_sq2();
  return llvm::failure();
}

/// Integer power x^exp.
///
/// Uses binary exponentiation (O(log |exp|) multiplications). Negative
/// exponents require x to be a unit; callers must ensure this precondition.
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

/// Return the square root of x in Z[√2] if x is a perfect square, otherwise
/// failure().
///
/// The algorithm finds candidate square roots w₁, w₂ ∈ Z[√2] from the
/// floor-square-roots of the rational candidates (x.a() ± r)/2 and
/// (x.a() ± r)/4 where r = ⌊√N(x)⌋, then verifies by squaring.
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

/// Euclidean division with remainder in Z[√2].
///
/// Z[√2] is a Euclidean domain with norm N(α) = |a² - 2b²|, so this
/// always yields (q, r) with x = y·q + r and N(r) < N(y).
inline std::pair<ZSqrt2, ZSqrt2> divmod(const ZSqrt2 &x, const ZSqrt2 &y) {
  ZSqrt2 p = x * y.conj_sq2();
  Integer k = y.norm();
  ZSqrt2 q(rounddiv(p.a(), k), rounddiv(p.b(), k));
  ZSqrt2 r = x - y * q;
  return {q, r};
}

/// Quotient of Euclidean division. Callers needing both quotient and
/// remainder should use `divmod` directly.
inline ZSqrt2 operator/(const ZSqrt2 &x, const ZSqrt2 &y) {
  return divmod(x, y).first;
}

/// Remainder of Euclidean division. Callers needing both quotient and
/// remainder should use `divmod` directly.
inline ZSqrt2 operator%(const ZSqrt2 &x, const ZSqrt2 &y) {
  return divmod(x, y).second;
}

/// GCD in Z[√2] via the Euclidean algorithm.
///
/// The result is unique only up to multiplication by units (±λ^m).
inline ZSqrt2 gcd(ZSqrt2 a, ZSqrt2 b) {
  const ZSqrt2 zero(0, 0);
  while (!(b == zero)) {
    auto r = divmod(a, b).second;
    a = b;
    b = r;
  }
  return a;
}

/// Return true iff a and b are associates (a | b and b | a in Z[√2]),
/// i.e. a = u·b for some unit u ∈ Z[√2]×.
inline bool are_associates(const ZSqrt2 &a, const ZSqrt2 &b) {
  const ZSqrt2 zero(0, 0);
  return (a % b == zero) && (b % a == zero);
}

} // namespace cudaq::synth
