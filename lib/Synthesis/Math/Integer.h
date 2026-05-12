/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cassert>
#include <gmp.h>
#include <iostream>
#include <string>

#include "Math/Types.h"

/// integer.h
///
/// GMP-based arbitrary precision integer implementation (wraps mpz_t).
///
/// The gridsynth algorithm (Ross & Selinger, arXiv:1403.2975) requires
/// arbitrary-precision integers for:
/// - Ring element coefficients in Z[√2] and Z[ω] (§3, Definition 3.1)
/// - Norm computations N(α) = α·α● which can be quadratic in the
///   coefficient sizes (Remark 3.3)
/// - Modular arithmetic in the Diophantine solver: Tonelli-Shanks for
///   √(-1) mod p, Miller-Rabin primality testing, Pollard-Brent
///   factoring (Appendix C)
/// - Euclidean division in Z[√2] for GCD computations (Lemma C.4)
///
/// The class provides a value-semantic wrapper around GMP's mpz_t with
/// operator overloads that mirror native integer types.
///
/// Integer type conventions:
///   - Public API uses i32 / i64 (from types.h) for all fixed-width integers.
///   - GMP API callsites cast to long / unsigned long as required by the
///     library ABI; these casts are always lossless on LP64 (Linux).

namespace cudaq::synth {
/// Integer: Arbitrary-precision integer backed by GMP's mpz_t.
class Integer {
private:
  mpz_t value_;

public:
  // Constructors
  Integer() { mpz_init(value_); }

  Integer(i32 val) { mpz_init_set_si(value_, static_cast<long>(val)); }

  Integer(i64 val) { mpz_init_set_si(value_, static_cast<long>(val)); }

  // Copy constructor
  Integer(const Integer &other) { mpz_init_set(value_, other.value_); }

  // Move constructor
  Integer(Integer &&other) noexcept {
    mpz_init(value_);
    mpz_swap(value_, other.value_);
  }

  // Destructor
  ~Integer() { mpz_clear(value_); }

  bool is_odd() const { return mpz_odd_p(value_); }

  // Assignment operators
  Integer &operator=(const Integer &other) {
    if (this != &other)
      mpz_set(value_, other.value_);
    return *this;
  }

  Integer &operator=(Integer &&other) noexcept {
    if (this != &other)
      mpz_swap(value_, other.value_);
    return *this;
  }

  Integer &operator=(i32 val) {
    mpz_set_si(value_, static_cast<long>(val));
    return *this;
  }

  Integer &operator=(i64 val) {
    mpz_set_si(value_, static_cast<long>(val));
    return *this;
  }

  // Conversion operators
  explicit operator i32() const { return static_cast<i32>(mpz_get_si(value_)); }

  explicit operator i64() const { return static_cast<i64>(mpz_get_si(value_)); }

  explicit operator double() const { return mpz_get_d(value_); }

  explicit operator size_t() const {
    return static_cast<size_t>(mpz_get_ui(value_));
  }

  explicit operator bool() const { return mpz_cmp_si(value_, 0) != 0; }

  // Access to internal representation for efficient operations
  mpz_t &get_mpz_t() { return value_; }
  const mpz_t &get_mpz_t() const { return value_; }

  // Arithmetic operators
  Integer operator+(const Integer &other) const {
    Integer result;
    mpz_add(result.value_, value_, other.value_);
    return result;
  }

  Integer operator-(const Integer &other) const {
    Integer result;
    mpz_sub(result.value_, value_, other.value_);
    return result;
  }

  Integer operator*(const Integer &other) const {
    Integer result;
    mpz_mul(result.value_, value_, other.value_);
    return result;
  }

  Integer operator/(const Integer &other) const {
    Integer result;
    mpz_tdiv_q(result.value_, value_, other.value_);
    return result;
  }

  Integer operator%(const Integer &other) const {
    Integer result;
    mpz_tdiv_r(result.value_, value_, other.value_);
    return result;
  }

  // Compound assignment operators
  Integer &operator+=(const Integer &other) {
    mpz_add(value_, value_, other.value_);
    return *this;
  }

  Integer &operator-=(const Integer &other) {
    mpz_sub(value_, value_, other.value_);
    return *this;
  }

  Integer &operator*=(const Integer &other) {
    mpz_mul(value_, value_, other.value_);
    return *this;
  }

  Integer &operator/=(const Integer &other) {
    mpz_tdiv_q(value_, value_, other.value_);
    return *this;
  }

  Integer &operator%=(const Integer &other) {
    mpz_tdiv_r(value_, value_, other.value_);
    return *this;
  }

  // Compound assignment with i64 (avoid temporary Integer construction)
  Integer &operator+=(i64 rhs) {
    if (rhs >= 0)
      mpz_add_ui(value_, value_, static_cast<unsigned long>(rhs));
    else
      mpz_sub_ui(value_, value_, static_cast<unsigned long>(-rhs));
    return *this;
  }

  Integer &operator-=(i64 rhs) {
    if (rhs >= 0)
      mpz_sub_ui(value_, value_, static_cast<unsigned long>(rhs));
    else
      mpz_add_ui(value_, value_, static_cast<unsigned long>(-rhs));
    return *this;
  }

  Integer &operator*=(i64 rhs) {
    mpz_mul_si(value_, value_, static_cast<long>(rhs));
    return *this;
  }

  Integer &operator/=(i64 rhs) {
    assert(rhs != 0 && "Integer::operator/=: division by zero");
    bool neg = rhs < 0;
    unsigned long mag = static_cast<unsigned long>(neg ? -rhs : rhs);
    mpz_tdiv_q_ui(value_, value_, mag);
    if (neg)
      mpz_neg(value_, value_);
    return *this;
  }

  Integer &operator%=(i64 rhs) {
    assert(rhs != 0 && "Integer::operator%=: modulo by zero");
    bool neg = rhs < 0; // Sign of remainder follows dividend per C++ semantics,
                        // so ignore sign of rhs
    unsigned long mag = static_cast<unsigned long>(neg ? -rhs : rhs);
    mpz_t r;
    mpz_init(r);
    mpz_tdiv_r_ui(r, value_, mag);
    mpz_set(value_, r); // store remainder
    mpz_clear(r);
    return *this;
  }

  // Increment and decrement operators
  Integer &operator++() {
    mpz_add_ui(value_, value_, 1);
    return *this;
  }

  Integer operator++(int) {
    Integer temp(*this);
    mpz_add_ui(value_, value_, 1);
    return temp;
  }

  // Comparison operators
  bool operator==(const Integer &other) const {
    return mpz_cmp(value_, other.value_) == 0;
  }

  bool operator!=(const Integer &other) const {
    return mpz_cmp(value_, other.value_) != 0;
  }

  bool operator<(const Integer &other) const {
    return mpz_cmp(value_, other.value_) < 0;
  }

  bool operator<=(const Integer &other) const {
    return mpz_cmp(value_, other.value_) <= 0;
  }

  bool operator>(const Integer &other) const {
    return mpz_cmp(value_, other.value_) > 0;
  }

  bool operator>=(const Integer &other) const {
    return mpz_cmp(value_, other.value_) >= 0;
  }

  // Bitwise operators
  Integer operator<<(i32 shift) const {
    Integer result;
    mpz_mul_2exp(result.value_, value_, static_cast<mp_bitcnt_t>(shift));
    return result;
  }

  Integer operator>>(i32 shift) const {
    Integer result;
    mpz_tdiv_q_2exp(result.value_, value_, static_cast<mp_bitcnt_t>(shift));
    return result;
  }

  Integer operator<<(const Integer &shift) const {
    Integer result;
    mpz_mul_2exp(result.value_, value_,
                 static_cast<mp_bitcnt_t>(mpz_get_ui(shift.value_)));
    return result;
  }

  Integer operator>>(const Integer &shift) const {
    Integer result;
    mpz_tdiv_q_2exp(result.value_, value_,
                    static_cast<mp_bitcnt_t>(mpz_get_ui(shift.value_)));
    return result;
  }

  Integer operator&(const Integer &other) const {
    Integer result;
    mpz_and(result.value_, value_, other.value_);
    return result;
  }

  Integer operator|(const Integer &other) const {
    Integer result;
    mpz_ior(result.value_, value_, other.value_);
    return result;
  }

  Integer operator^(const Integer &other) const {
    Integer result;
    mpz_xor(result.value_, value_, other.value_);
    return result;
  }

  // Compound bitwise assignment operators
  Integer &operator<<=(i32 shift) {
    mpz_mul_2exp(value_, value_, static_cast<mp_bitcnt_t>(shift));
    return *this;
  }

  Integer &operator>>=(i32 shift) {
    mpz_tdiv_q_2exp(value_, value_, static_cast<mp_bitcnt_t>(shift));
    return *this;
  }

  Integer &operator&=(const Integer &other) {
    mpz_and(value_, value_, other.value_);
    return *this;
  }

  // Unary operators
  Integer operator-() const {
    Integer result;
    mpz_neg(result.value_, value_);
    return result;
  }

  bool operator!() const { return mpz_cmp_si(value_, 0) == 0; }

  std::string to_string() const {
    char *str = mpz_get_str(nullptr, 10, value_);
    std::string result(str);
    free(str);
    return result;
  }

  friend std::ostream &operator<<(std::ostream &os, const Integer &val) {
    return os << val.to_string();
  }
};

/// Number of trailing zeros in binary representation.
///
/// Uses GMP's mpz_scan1 which finds the first 1 bit. For a positive integer,
/// this gives the number of trailing zeros. Returns 0 for zero input.
inline Integer ntz(const Integer &n) {
  if (n == Integer(0))
    return Integer(0);

  // mpz_scan1 returns the index of the first 1 bit (0-indexed from LSB)
  mp_bitcnt_t trailing_zeros = mpz_scan1(n.get_mpz_t(), 0);
  return Integer(static_cast<i64>(trailing_zeros));
}

/// Returns the sign of x as -1, 0, or +1.
inline i32 sign(const Integer &x) {
  return static_cast<i32>(mpz_sgn(x.get_mpz_t()));
}

/// Floor of square root for non-negative integers.
inline Integer floorsqrt(const Integer &x) {
  assert(!(x < 0) && "floorsqrt: negative input");
  if (x == Integer(0))
    return Integer(0);

  Integer result;
  mpz_sqrt(result.get_mpz_t(), x.get_mpz_t());
  return result;
}

/// Python-style floor division (truncate toward -infinity).
inline Integer floordiv(const Integer &x, const Integer &y) {
  Integer result;
  mpz_fdiv_q(result.get_mpz_t(), x.get_mpz_t(), y.get_mpz_t());
  return result;
}

/// floordiv overload for i32 divisors: avoids constructing a temporary
/// Integer for the common case.  When y is a positive power of 2,
/// mpz_fdiv_q_2exp is used — it is O(1) (arithmetic right-shift on GMP
/// limbs) vs. O(n limbs) for general division.
inline Integer floordiv(const Integer &x, i32 y) {
  Integer result;
  if (y > 0 && (y & (y - 1)) == 0) {
    // Positive power of 2: floor(x / 2^k) is a single limb shift.
    const unsigned k =
        static_cast<unsigned>(__builtin_ctz(static_cast<unsigned>(y)));
    mpz_fdiv_q_2exp(result.get_mpz_t(), x.get_mpz_t(), k);
  } else {
    // No mpz_fdiv_q_si in GMP; construct a single-limb divisor and call
    // mpz_fdiv_q.  Aliasing result as both src and dst is unsafe, so we
    // use a small named temp.
    Integer divisor(y);
    mpz_fdiv_q(result.get_mpz_t(), x.get_mpz_t(), divisor.get_mpz_t());
  }
  return result;
}

/// Round-to-nearest integer division.
inline Integer rounddiv(const Integer &x, const Integer &y) {
  if (y > Integer(0))
    return floordiv(x + floordiv(y, 2), y);
  return floordiv(x - floordiv(-y, 2), y);
}

/// Greatest common divisor using GMP's mpz_gcd.
inline Integer gcd(Integer a, Integer b) {
  // GMP gcd takes absolute values internally, but we do it explicitly
  if (a < Integer(0))
    a = -a;
  if (b < Integer(0))
    b = -b;

  Integer result;
  mpz_gcd(result.get_mpz_t(), a.get_mpz_t(), b.get_mpz_t());
  return result;
}

/// Probable-primality test using GMP's mpz_probab_prime_p.
///
/// Returns true if n is probably or definitely prime, false if n is
/// definitely composite or not a valid candidate (n < 2). If n < 0,
/// tests |n| so that is_probably_prime(-7) is true.
///
/// @param n The integer to test (uses |n| if n < 0).
/// @param reps Number of Miller–Rabin rounds; higher reduces false positives.
///             GMP uses this for the "probably prime" case.
/// @return true if mpz_probab_prime_p returns 1 or 2, false if 0 or |n| < 2.
inline bool is_probably_prime(const Integer &n, i32 reps = 4) {
  if (n == Integer(0) || n == Integer(1))
    return false;
  Integer m = n;
  if (m < Integer(0))
    m = -m;
  if (m < Integer(2))
    return false;
  i32 result = static_cast<i32>(mpz_probab_prime_p(m.get_mpz_t(), reps));
  return result != 0; // 2 = definitely prime, 1 = probably prime
}

// Optimized mixed operations with i64 (avoid temporary Integer construction)

inline Integer operator+(const Integer &lhs, i64 rhs) {
  Integer result(lhs);
  result += rhs;
  return result;
}
inline Integer operator+(i64 lhs, const Integer &rhs) {
  Integer result(rhs);
  result += lhs;
  return result;
}
inline Integer operator-(const Integer &lhs, i64 rhs) {
  Integer result(lhs);
  result -= rhs;
  return result;
}
inline Integer operator-(i64 lhs, const Integer &rhs) {
  Integer result(lhs);
  result -= static_cast<i64>(rhs.operator i64());
  return result;
}
inline Integer operator*(const Integer &lhs, i64 rhs) {
  Integer result(lhs);
  result *= rhs;
  return result;
}
inline Integer operator*(i64 lhs, const Integer &rhs) {
  Integer result(rhs);
  result *= lhs;
  return result;
}
inline Integer operator/(const Integer &lhs, i64 rhs) {
  Integer result(lhs);
  result /= rhs;
  return result;
}
inline Integer operator/(i64 lhs, const Integer &rhs) {
  Integer result(lhs);
  // Division by large rhs still needs full mpz operation
  mpz_tdiv_q(result.get_mpz_t(), result.get_mpz_t(), rhs.get_mpz_t());
  return result;
}
inline Integer operator%(const Integer &lhs, i64 rhs) {
  Integer result(lhs);
  result %= rhs;
  return result;
}
inline Integer operator%(i64 lhs, const Integer &rhs) {
  Integer result(lhs);
  mpz_tdiv_r(result.get_mpz_t(), result.get_mpz_t(), rhs.get_mpz_t());
  return result;
}
inline Integer operator<<(i64 lhs, const Integer &rhs) {
  Integer temp(lhs);
  return temp << static_cast<i32>(rhs); // potential narrowing
}

// Comparison operators with i64 — use mpz_cmp_si directly.
// On 64-bit Linux long == i64 (both 64-bit), so the cast is lossless.
inline bool operator==(const Integer &lhs, i64 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) == 0;
}
inline bool operator!=(const Integer &lhs, i64 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) != 0;
}
inline bool operator<(const Integer &lhs, i64 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) < 0;
}
inline bool operator<=(const Integer &lhs, i64 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) <= 0;
}
inline bool operator>(const Integer &lhs, i64 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) > 0;
}
inline bool operator>=(const Integer &lhs, i64 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) >= 0;
}

inline bool operator==(i64 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) == 0;
}
inline bool operator!=(i64 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) != 0;
}
inline bool operator<(i64 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) > 0;
}
inline bool operator<=(i64 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) >= 0;
}
inline bool operator>(i64 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) < 0;
}
inline bool operator>=(i64 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) <= 0;
}

// Bitwise operators with i64
inline Integer operator&(const Integer &lhs, i64 rhs) {
  return lhs & Integer(rhs);
}
inline Integer operator|(const Integer &lhs, i64 rhs) {
  return lhs | Integer(rhs);
}
inline Integer operator^(const Integer &lhs, i64 rhs) {
  return lhs ^ Integer(rhs);
}

inline Integer operator&(i64 lhs, const Integer &rhs) {
  return Integer(lhs) & rhs;
}
inline Integer operator|(i64 lhs, const Integer &rhs) {
  return Integer(lhs) | rhs;
}
inline Integer operator^(i64 lhs, const Integer &rhs) {
  return Integer(lhs) ^ rhs;
}

// Mixed operations with i32 (delegate to i64 overloads)
inline Integer operator+(const Integer &lhs, i32 rhs) {
  return lhs + static_cast<i64>(rhs);
}
inline Integer operator+(i32 lhs, const Integer &rhs) {
  return static_cast<i64>(lhs) + rhs;
}
inline Integer operator-(const Integer &lhs, i32 rhs) {
  return lhs - static_cast<i64>(rhs);
}
inline Integer operator-(i32 lhs, const Integer &rhs) {
  return static_cast<i64>(lhs) - rhs;
}
inline Integer operator*(const Integer &lhs, i32 rhs) {
  return lhs * static_cast<i64>(rhs);
}
inline Integer operator*(i32 lhs, const Integer &rhs) {
  return static_cast<i64>(lhs) * rhs;
}
inline Integer operator/(const Integer &lhs, i32 rhs) {
  return lhs / static_cast<i64>(rhs);
}
inline Integer operator/(i32 lhs, const Integer &rhs) {
  return static_cast<i64>(lhs) / rhs;
}
inline Integer operator%(const Integer &lhs, i32 rhs) {
  return lhs % static_cast<i64>(rhs);
}
inline Integer operator%(i32 lhs, const Integer &rhs) {
  return static_cast<i64>(lhs) % rhs;
}

// Comparison operators with i32 — use mpz_cmp_si directly to avoid
// constructing a temporary Integer(rhs) (mpz_init + mpz_set_si + mpz_clear).
inline bool operator==(const Integer &lhs, i32 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) == 0;
}
inline bool operator!=(const Integer &lhs, i32 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) != 0;
}
inline bool operator<(const Integer &lhs, i32 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) < 0;
}
inline bool operator<=(const Integer &lhs, i32 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) <= 0;
}
inline bool operator>(const Integer &lhs, i32 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) > 0;
}
inline bool operator>=(const Integer &lhs, i32 rhs) {
  return mpz_cmp_si(lhs.get_mpz_t(), static_cast<long>(rhs)) >= 0;
}

inline bool operator==(i32 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) == 0;
}
inline bool operator!=(i32 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) != 0;
}
inline bool operator<(i32 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) > 0;
}
inline bool operator<=(i32 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) >= 0;
}
inline bool operator>(i32 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) < 0;
}
inline bool operator>=(i32 lhs, const Integer &rhs) {
  return mpz_cmp_si(rhs.get_mpz_t(), static_cast<long>(lhs)) <= 0;
}

// Bitwise operators with i32
inline Integer operator&(const Integer &lhs, i32 rhs) {
  return lhs & Integer(rhs);
}
inline Integer operator|(const Integer &lhs, i32 rhs) {
  return lhs | Integer(rhs);
}
inline Integer operator^(const Integer &lhs, i32 rhs) {
  return lhs ^ Integer(rhs);
}

} // namespace cudaq::synth
