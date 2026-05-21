/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Synthesis/Math/Integer.h"

// Enable mpfr_set_sj / mpfr_get_sj (intmax_t variants). Needed so that i64
// bridges into MPFR without truncation on platforms where `long` is 32-bit
// (e.g. LLP64).
#ifndef MPFR_USE_INTMAX_T
#define MPFR_USE_INTMAX_T
#endif

#include <algorithm>
#include <cctype>
#include <climits>
#include <cmath>
#include <cstring>
#include <mpfr.h>
#include <optional>
#include <string>
#include <utility>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Real
//===----------------------------------------------------------------------===//

/// MPFR-backed arbitrary-precision floating-point number.
class Real {
private:
  mpfr_t value_;
  static mpfr_prec_t default_precision_;

  // Internal tag to construct at a specific precision without going through
  // the default-precision path twice.
  struct direct_init_tag {};
  Real(direct_init_tag, mpfr_prec_t prec) {
    mpfr_init2(value_, prec);
    mpfr_set_zero(value_, 1);
  }

public:
  // -- Precision management --

  static void set_default_precision(mpfr_prec_t prec) {
    default_precision_ = prec;
  }
  static mpfr_prec_t get_default_precision() { return default_precision_; }

  /// Build a Real with a non-default precision.
  static Real with_precision(mpfr_prec_t precision, double val = 0.0) {
    Real result(direct_init_tag{}, precision);
    if (val != 0.0)
      mpfr_set_d(result.value_, val, MPFR_RNDN);
    return result;
  }

  // -- Construction --

  Real() {
    mpfr_init2(value_, default_precision_);
    mpfr_set_zero(value_, 1);
  }

  Real(i32 val) {
    mpfr_init2(value_, default_precision_);
    mpfr_set_si(value_, static_cast<long>(val), MPFR_RNDN);
  }

  Real(i64 val) {
    mpfr_init2(value_, default_precision_);
    mpfr_set_sj(value_, static_cast<intmax_t>(val), MPFR_RNDN);
  }

  Real(float val) {
    mpfr_init2(value_, default_precision_);
    mpfr_set_flt(value_, val, MPFR_RNDN);
  }

  Real(double val) {
    mpfr_init2(value_, default_precision_);
    mpfr_set_d(value_, val, MPFR_RNDN);
  }

  Real(const Integer &val) {
    mpfr_init2(value_, default_precision_);
    mpfr_set_z(value_, val.get_mpz_t(), MPFR_RNDN);
  }

  Real(long double val) {
    mpfr_init2(value_, default_precision_);
    mpfr_set_ld(value_, val, MPFR_RNDN);
  }

  Real(const std::string &str) {
    mpfr_init2(value_, default_precision_);
    mpfr_set_str(value_, str.c_str(), 10, MPFR_RNDN);
  }

  Real(const Real &other) {
    mpfr_init2(value_, mpfr_get_prec(other.value_));
    mpfr_set(value_, other.value_, MPFR_RNDN);
  }

  /// Move: steal the mpfr_t struct directly. The source is left in a
  /// "moved-from" state where `_mpfr_d == nullptr`; the destructor guards
  /// against double-free with this sentinel.
  Real(Real &&other) noexcept {
    std::memcpy(&value_, &other.value_, sizeof(mpfr_t));
    other.value_[0]._mpfr_d = nullptr;
  }

  ~Real() {
    if (value_[0]._mpfr_d != nullptr)
      mpfr_clear(value_);
  }

  // -- Assignment --

  /// Copy assignment. Reinitialises the destination only when precisions
  /// disagree; the common same-precision case stays at one mpfr_set.
  Real &operator=(const Real &other) {
    if (this != &other) {
      mpfr_prec_t tp = mpfr_get_prec(value_);
      mpfr_prec_t op = mpfr_get_prec(other.value_);
      if (tp != op) {
        mpfr_clear(value_);
        mpfr_init2(value_, op);
      }
      mpfr_set(value_, other.value_, MPFR_RNDN);
    }
    return *this;
  }

  Real &operator=(Real &&other) noexcept {
    if (this != &other)
      mpfr_swap(value_, other.value_);
    return *this;
  }

  Real &operator=(i32 val) {
    mpfr_set_si(value_, static_cast<long>(val), MPFR_RNDN);
    return *this;
  }

  Real &operator=(i64 val) {
    mpfr_set_sj(value_, static_cast<intmax_t>(val), MPFR_RNDN);
    return *this;
  }

  Real &operator=(float val) {
    mpfr_set_flt(value_, val, MPFR_RNDN);
    return *this;
  }

  Real &operator=(double val) {
    mpfr_set_d(value_, val, MPFR_RNDN);
    return *this;
  }

  Real &operator=(long double val) {
    mpfr_set_ld(value_, val, MPFR_RNDN);
    return *this;
  }

  // -- Conversions --

  explicit operator i32() const {
    return static_cast<i32>(mpfr_get_si(value_, MPFR_RNDN));
  }

  explicit operator i64() const {
    return static_cast<i64>(mpfr_get_sj(value_, MPFR_RNDN));
  }

  explicit operator float() const { return mpfr_get_flt(value_, MPFR_RNDN); }

  [[nodiscard]] explicit operator double() const noexcept {
    return mpfr_get_d(value_, MPFR_RNDN);
  }

  explicit operator long double() const {
    return mpfr_get_ld(value_, MPFR_RNDN);
  }

  [[nodiscard]] explicit operator bool() const noexcept {
    return !mpfr_zero_p(value_);
  }

  [[nodiscard]] double to_double() const noexcept {
    return mpfr_get_d(value_, MPFR_RNDN);
  }

  // -- Arithmetic --

  Real operator+(const Real &other) const {
    mpfr_prec_t prec =
        std::max(mpfr_get_prec(value_), mpfr_get_prec(other.value_));
    Real result(direct_init_tag{}, prec);
    mpfr_add(result.value_, value_, other.value_, MPFR_RNDN);
    return result;
  }

  Real operator-(const Real &other) const {
    mpfr_prec_t prec =
        std::max(mpfr_get_prec(value_), mpfr_get_prec(other.value_));
    Real result(direct_init_tag{}, prec);
    mpfr_sub(result.value_, value_, other.value_, MPFR_RNDN);
    return result;
  }

  Real operator*(const Real &other) const {
    mpfr_prec_t prec =
        std::max(mpfr_get_prec(value_), mpfr_get_prec(other.value_));
    Real result(direct_init_tag{}, prec);
    mpfr_mul(result.value_, value_, other.value_, MPFR_RNDN);
    return result;
  }

  Real operator/(const Real &other) const {
    mpfr_prec_t prec =
        std::max(mpfr_get_prec(value_), mpfr_get_prec(other.value_));
    Real result(direct_init_tag{}, prec);
    mpfr_div(result.value_, value_, other.value_, MPFR_RNDN);
    return result;
  }

  // -- Compound assignment (Real, double, i32) --

  Real &operator+=(const Real &other) {
    mpfr_add(value_, value_, other.value_, MPFR_RNDN);
    return *this;
  }

  Real &operator-=(const Real &other) {
    mpfr_sub(value_, value_, other.value_, MPFR_RNDN);
    return *this;
  }

  Real &operator*=(const Real &other) {
    mpfr_mul(value_, value_, other.value_, MPFR_RNDN);
    return *this;
  }

  Real &operator/=(const Real &other) {
    mpfr_div(value_, value_, other.value_, MPFR_RNDN);
    return *this;
  }

  // double overloads avoid constructing a temporary Real for the rhs.
  Real &operator+=(double rhs) {
    mpfr_add_d(value_, value_, rhs, MPFR_RNDN);
    return *this;
  }
  Real &operator-=(double rhs) {
    mpfr_sub_d(value_, value_, rhs, MPFR_RNDN);
    return *this;
  }
  Real &operator*=(double rhs) {
    mpfr_mul_d(value_, value_, rhs, MPFR_RNDN);
    return *this;
  }
  Real &operator/=(double rhs) {
    mpfr_div_d(value_, value_, rhs, MPFR_RNDN);
    return *this;
  }

  // i32 overloads bind directly to mpfr_*_si helpers.
  Real &operator+=(i32 rhs) {
    mpfr_add_si(value_, value_, static_cast<long>(rhs), MPFR_RNDN);
    return *this;
  }
  Real &operator-=(i32 rhs) {
    mpfr_sub_si(value_, value_, static_cast<long>(rhs), MPFR_RNDN);
    return *this;
  }
  Real &operator*=(i32 rhs) {
    mpfr_mul_si(value_, value_, static_cast<long>(rhs), MPFR_RNDN);
    return *this;
  }
  Real &operator/=(i32 rhs) {
    mpfr_div_si(value_, value_, static_cast<long>(rhs), MPFR_RNDN);
    return *this;
  }

  // -- Increment / decrement --

  Real &operator++() {
    mpfr_add_ui(value_, value_, 1, MPFR_RNDN);
    return *this;
  }

  Real operator++(int) {
    Real temp(*this);
    mpfr_add_ui(value_, value_, 1, MPFR_RNDN);
    return temp;
  }

  Real &operator--() {
    mpfr_sub_ui(value_, value_, 1, MPFR_RNDN);
    return *this;
  }

  Real operator--(int) {
    Real temp(*this);
    mpfr_sub_ui(value_, value_, 1, MPFR_RNDN);
    return temp;
  }

  // -- Comparison --

  bool operator==(const Real &other) const {
    return mpfr_equal_p(value_, other.value_) != 0;
  }
  bool operator!=(const Real &other) const {
    return mpfr_equal_p(value_, other.value_) == 0;
  }
  bool operator<(const Real &other) const {
    return mpfr_less_p(value_, other.value_) != 0;
  }
  bool operator<=(const Real &other) const {
    return mpfr_lessequal_p(value_, other.value_) != 0;
  }
  bool operator>(const Real &other) const {
    return mpfr_greater_p(value_, other.value_) != 0;
  }
  bool operator>=(const Real &other) const {
    return mpfr_greaterequal_p(value_, other.value_) != 0;
  }

  // -- Unary --

  Real operator-() const {
    Real result(direct_init_tag{}, mpfr_get_prec(value_));
    mpfr_neg(result.value_, value_, MPFR_RNDN);
    return result;
  }

  Real operator+() const { return *this; }

  bool operator!() const { return mpfr_zero_p(value_) != 0; }

  // -- Classification helpers --

  bool is_nan() const { return mpfr_nan_p(value_) != 0; }
  bool is_inf() const { return mpfr_inf_p(value_) != 0; }
  bool is_zero() const { return mpfr_zero_p(value_) != 0; }
  bool is_finite() const { return mpfr_number_p(value_) != 0; }

  mpfr_prec_t precision() const { return mpfr_get_prec(value_); }
  void set_precision(mpfr_prec_t prec) {
    mpfr_prec_round(value_, prec, MPFR_RNDN);
  }

  std::string to_string(i32 digits = 40) const;

  /// Raw access to the underlying mpfr_t for callers that drop into MPFR
  /// APIs directly.
  const mpfr_t &get_mpfr() const { return value_; }
  mpfr_t &get_mpfr() { return value_; }

  // -- Mathematical constants --
  //
  // Function-local statics dodge static-initialization-order issues and
  // capture the default precision in force at first use.

  static const Real &sqrt2() {
    static const Real value = []() {
      Real v;
      mpfr_sqrt_ui(v.get_mpfr(), 2, MPFR_RNDN);
      return v;
    }();
    return value;
  }
  static const Real &pi() {
    static const Real value = []() {
      Real v;
      mpfr_const_pi(v.get_mpfr(), MPFR_RNDN);
      return v;
    }();
    return value;
  }
  // Infinity is precision-independent; use mpfr_set_inf directly rather
  // than going through a double conversion.
  static const Real &inf() {
    static const Real value = []() {
      Real v;
      mpfr_set_inf(v.get_mpfr(), 1);
      return v;
    }();
    return value;
  }
  static const Real &neg_inf() {
    static const Real value = []() {
      Real v;
      mpfr_set_inf(v.get_mpfr(), -1);
      return v;
    }();
    return value;
  }
};

// `Real::default_precision_` is defined in Math/Real.cpp.

//===----------------------------------------------------------------------===//
// Standard math functions
//===----------------------------------------------------------------------===//

inline Real abs(const Real &x) {
  Real result = Real::with_precision(x.precision());
  mpfr_abs(result.get_mpfr(), x.get_mpfr(), MPFR_RNDN);
  return result;
}
inline Real sqrt(const Real &x) {
  Real result = Real::with_precision(x.precision());
  mpfr_sqrt(result.get_mpfr(), x.get_mpfr(), MPFR_RNDN);
  return result;
}
inline Real log(const Real &x) {
  Real result = Real::with_precision(x.precision());
  mpfr_log(result.get_mpfr(), x.get_mpfr(), MPFR_RNDN);
  return result;
}
inline Real sin(const Real &x) {
  Real result = Real::with_precision(x.precision());
  mpfr_sin(result.get_mpfr(), x.get_mpfr(), MPFR_RNDN);
  return result;
}
inline Real cos(const Real &x) {
  Real result = Real::with_precision(x.precision());
  mpfr_cos(result.get_mpfr(), x.get_mpfr(), MPFR_RNDN);
  return result;
}

/// ADL-friendly swap. Uses mpfr_swap which is O(1).
inline void swap(Real &a, Real &b) noexcept {
  mpfr_swap(a.get_mpfr(), b.get_mpfr());
}

//===----------------------------------------------------------------------===//
// Real -> Integer rounding helpers
//===----------------------------------------------------------------------===//

inline Integer floor_to_integer(const Real &x) {
  Integer result;
  mpfr_get_z(result.get_mpz_t(), x.get_mpfr(), MPFR_RNDD);
  return result;
}

inline Integer ceil_to_integer(const Real &x) {
  Integer result;
  mpfr_get_z(result.get_mpz_t(), x.get_mpfr(), MPFR_RNDU);
  return result;
}

inline Integer round_to_integer(const Real &x) {
  Integer result;
  mpfr_get_z(result.get_mpz_t(), x.get_mpfr(), MPFR_RNDN);
  return result;
}

/// Compute sqrt(2)^k at arbitrary precision.
///
/// 2^(|k|/2) is realised in O(1) by `mpfr_mul_2ui`, which shifts the MPFR
/// exponent field directly rather than performing a full multiplication.
/// In practice |k|/2 always fits in `unsigned long` (it is bounded by the
/// working precision in bits).
inline Real pow_sqrt2(const Integer &k) {
  if (k == 0)
    return Real(1);

  bool negative = k < 0;
  Integer abs_k = negative ? -k : k;

  unsigned long half = mpz_get_ui((abs_k >> 1).get_mpz_t());
  bool odd = (abs_k & Integer(1)) != 0;

  Real result(1);
  if (half > 0)
    mpfr_mul_2ui(result.get_mpfr(), result.get_mpfr(), half, MPFR_RNDN);
  if (odd)
    result *= Real::sqrt2();

  if (negative) {
    Real inv = Real::with_precision(result.precision());
    mpfr_ui_div(inv.get_mpfr(), 1, result.get_mpfr(), MPFR_RNDN);
    return inv;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Mixed comparisons / arithmetic with `double` (no temporary Real)
//===----------------------------------------------------------------------===//

inline bool operator==(const Real &lhs, double rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) && mpfr_cmp_d(lhs.get_mpfr(), rhs) == 0;
}
inline bool operator!=(const Real &lhs, double rhs) noexcept {
  return mpfr_nan_p(lhs.get_mpfr()) || mpfr_cmp_d(lhs.get_mpfr(), rhs) != 0;
}
inline bool operator<(const Real &lhs, double rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) && mpfr_cmp_d(lhs.get_mpfr(), rhs) < 0;
}
inline bool operator<=(const Real &lhs, double rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) && mpfr_cmp_d(lhs.get_mpfr(), rhs) <= 0;
}
inline bool operator>(const Real &lhs, double rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) && mpfr_cmp_d(lhs.get_mpfr(), rhs) > 0;
}
inline bool operator>=(const Real &lhs, double rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) && mpfr_cmp_d(lhs.get_mpfr(), rhs) >= 0;
}

inline bool operator==(double lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) && mpfr_cmp_d(rhs.get_mpfr(), lhs) == 0;
}
inline bool operator!=(double lhs, const Real &rhs) noexcept {
  return mpfr_nan_p(rhs.get_mpfr()) || mpfr_cmp_d(rhs.get_mpfr(), lhs) != 0;
}
inline bool operator<(double lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) && mpfr_cmp_d(rhs.get_mpfr(), lhs) > 0;
}
inline bool operator<=(double lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) && mpfr_cmp_d(rhs.get_mpfr(), lhs) >= 0;
}
inline bool operator>(double lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) && mpfr_cmp_d(rhs.get_mpfr(), lhs) < 0;
}
inline bool operator>=(double lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) && mpfr_cmp_d(rhs.get_mpfr(), lhs) <= 0;
}

inline Real operator+(const Real &lhs, double rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_add_d(result.get_mpfr(), lhs.get_mpfr(), rhs, MPFR_RNDN);
  return result;
}
inline Real operator+(double lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_add_d(result.get_mpfr(), rhs.get_mpfr(), lhs, MPFR_RNDN);
  return result;
}
inline Real operator-(const Real &lhs, double rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_sub_d(result.get_mpfr(), lhs.get_mpfr(), rhs, MPFR_RNDN);
  return result;
}
inline Real operator-(double lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_d_sub(result.get_mpfr(), lhs, rhs.get_mpfr(), MPFR_RNDN);
  return result;
}
inline Real operator*(const Real &lhs, double rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_mul_d(result.get_mpfr(), lhs.get_mpfr(), rhs, MPFR_RNDN);
  return result;
}
inline Real operator*(double lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_mul_d(result.get_mpfr(), rhs.get_mpfr(), lhs, MPFR_RNDN);
  return result;
}
inline Real operator/(const Real &lhs, double rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_div_d(result.get_mpfr(), lhs.get_mpfr(), rhs, MPFR_RNDN);
  return result;
}
inline Real operator/(double lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_d_div(result.get_mpfr(), lhs, rhs.get_mpfr(), MPFR_RNDN);
  return result;
}

//===----------------------------------------------------------------------===//
// Mixed arithmetic / comparison with `i32` (no temporary Real)
//===----------------------------------------------------------------------===//

inline Real operator+(const Real &lhs, i32 rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_add_si(result.get_mpfr(), lhs.get_mpfr(), static_cast<long>(rhs),
              MPFR_RNDN);
  return result;
}
inline Real operator+(i32 lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_add_si(result.get_mpfr(), rhs.get_mpfr(), static_cast<long>(lhs),
              MPFR_RNDN);
  return result;
}
inline Real operator-(const Real &lhs, i32 rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_sub_si(result.get_mpfr(), lhs.get_mpfr(), static_cast<long>(rhs),
              MPFR_RNDN);
  return result;
}
inline Real operator-(i32 lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_si_sub(result.get_mpfr(), static_cast<long>(lhs), rhs.get_mpfr(),
              MPFR_RNDN);
  return result;
}
inline Real operator*(const Real &lhs, i32 rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_mul_si(result.get_mpfr(), lhs.get_mpfr(), static_cast<long>(rhs),
              MPFR_RNDN);
  return result;
}
inline Real operator*(i32 lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_mul_si(result.get_mpfr(), rhs.get_mpfr(), static_cast<long>(lhs),
              MPFR_RNDN);
  return result;
}
inline Real operator/(const Real &lhs, i32 rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_div_si(result.get_mpfr(), lhs.get_mpfr(), static_cast<long>(rhs),
              MPFR_RNDN);
  return result;
}
inline Real operator/(i32 lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_si_div(result.get_mpfr(), static_cast<long>(lhs), rhs.get_mpfr(),
              MPFR_RNDN);
  return result;
}

inline bool operator==(const Real &lhs, i32 rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) &&
         mpfr_cmp_si(lhs.get_mpfr(), static_cast<long>(rhs)) == 0;
}
inline bool operator!=(const Real &lhs, i32 rhs) noexcept {
  return mpfr_nan_p(lhs.get_mpfr()) ||
         mpfr_cmp_si(lhs.get_mpfr(), static_cast<long>(rhs)) != 0;
}
inline bool operator<(const Real &lhs, i32 rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) &&
         mpfr_cmp_si(lhs.get_mpfr(), static_cast<long>(rhs)) < 0;
}
inline bool operator<=(const Real &lhs, i32 rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) &&
         mpfr_cmp_si(lhs.get_mpfr(), static_cast<long>(rhs)) <= 0;
}
inline bool operator>(const Real &lhs, i32 rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) &&
         mpfr_cmp_si(lhs.get_mpfr(), static_cast<long>(rhs)) > 0;
}
inline bool operator>=(const Real &lhs, i32 rhs) noexcept {
  return !mpfr_nan_p(lhs.get_mpfr()) &&
         mpfr_cmp_si(lhs.get_mpfr(), static_cast<long>(rhs)) >= 0;
}

inline bool operator==(i32 lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) &&
         mpfr_cmp_si(rhs.get_mpfr(), static_cast<long>(lhs)) == 0;
}
inline bool operator!=(i32 lhs, const Real &rhs) noexcept {
  return mpfr_nan_p(rhs.get_mpfr()) ||
         mpfr_cmp_si(rhs.get_mpfr(), static_cast<long>(lhs)) != 0;
}
inline bool operator<(i32 lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) &&
         mpfr_cmp_si(rhs.get_mpfr(), static_cast<long>(lhs)) > 0;
}
inline bool operator<=(i32 lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) &&
         mpfr_cmp_si(rhs.get_mpfr(), static_cast<long>(lhs)) >= 0;
}
inline bool operator>(i32 lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) &&
         mpfr_cmp_si(rhs.get_mpfr(), static_cast<long>(lhs)) < 0;
}
inline bool operator>=(i32 lhs, const Real &rhs) noexcept {
  return !mpfr_nan_p(rhs.get_mpfr()) &&
         mpfr_cmp_si(rhs.get_mpfr(), static_cast<long>(lhs)) <= 0;
}

//===----------------------------------------------------------------------===//
// Mixed arithmetic / comparison with `Integer` (no temporary Real)
//===----------------------------------------------------------------------===//

inline Real operator+(const Integer &lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_add_z(result.get_mpfr(), rhs.get_mpfr(), lhs.get_mpz_t(), MPFR_RNDN);
  return result;
}
inline Real operator-(const Integer &lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_z_sub(result.get_mpfr(), lhs.get_mpz_t(), rhs.get_mpfr(), MPFR_RNDN);
  return result;
}
inline Real operator*(const Integer &lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  mpfr_mul_z(result.get_mpfr(), rhs.get_mpfr(), lhs.get_mpz_t(), MPFR_RNDN);
  return result;
}
inline Real operator/(const Integer &lhs, const Real &rhs) {
  Real result = Real::with_precision(rhs.precision());
  Real temp(lhs);
  mpfr_div(result.get_mpfr(), temp.get_mpfr(), rhs.get_mpfr(), MPFR_RNDN);
  return result;
}

inline Real operator+(const Real &lhs, const Integer &rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_add_z(result.get_mpfr(), lhs.get_mpfr(), rhs.get_mpz_t(), MPFR_RNDN);
  return result;
}
inline Real operator-(const Real &lhs, const Integer &rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_sub_z(result.get_mpfr(), lhs.get_mpfr(), rhs.get_mpz_t(), MPFR_RNDN);
  return result;
}
inline Real operator*(const Real &lhs, const Integer &rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_mul_z(result.get_mpfr(), lhs.get_mpfr(), rhs.get_mpz_t(), MPFR_RNDN);
  return result;
}
inline Real operator/(const Real &lhs, const Integer &rhs) {
  Real result = Real::with_precision(lhs.precision());
  mpfr_div_z(result.get_mpfr(), lhs.get_mpfr(), rhs.get_mpz_t(), MPFR_RNDN);
  return result;
}

// Integer <-> Real comparisons go through Real(lhs) to keep the
// implementations short; the temporary is unavoidable since MPFR's compare-
// against-mpz API does not provide every relation.
inline bool operator==(const Integer &lhs, const Real &rhs) {
  return Real(lhs) == rhs;
}
inline bool operator!=(const Integer &lhs, const Real &rhs) {
  return Real(lhs) != rhs;
}
inline bool operator<(const Integer &lhs, const Real &rhs) {
  return Real(lhs) < rhs;
}
inline bool operator<=(const Integer &lhs, const Real &rhs) {
  return Real(lhs) <= rhs;
}
inline bool operator>(const Integer &lhs, const Real &rhs) {
  return Real(lhs) > rhs;
}
inline bool operator>=(const Integer &lhs, const Real &rhs) {
  return Real(lhs) >= rhs;
}

inline bool operator==(const Real &lhs, const Integer &rhs) {
  return lhs == Real(rhs);
}
inline bool operator!=(const Real &lhs, const Integer &rhs) {
  return lhs != Real(rhs);
}
inline bool operator<(const Real &lhs, const Integer &rhs) {
  return lhs < Real(rhs);
}
inline bool operator<=(const Real &lhs, const Integer &rhs) {
  return lhs <= Real(rhs);
}
inline bool operator>(const Real &lhs, const Integer &rhs) {
  return lhs > Real(rhs);
}
inline bool operator>=(const Real &lhs, const Integer &rhs) {
  return lhs >= Real(rhs);
}

//===----------------------------------------------------------------------===//
// Quadratic solver
//===----------------------------------------------------------------------===//

/// Solve a*x^2 + b*x + c = 0. Returns the two roots in (smaller, larger)
/// order, or std::nullopt if the discriminant is negative.
///
/// Numerically stable: we pick the s = -b +/- sqrt(disc) variant that keeps
/// the two summands the same sign (so cancellation can't happen), and then
/// recover the other root via Vieta's formula r1 * r2 = c/a. When the
/// chosen s is zero (a double root at the origin) we fall back to the
/// direct formula to avoid 0/0.
inline std::optional<std::pair<Real, Real>>
solve_quadratic(const Real &a, const Real &b, const Real &c) {
  Real discriminant = b * b - 4 * a * c;
  if (discriminant < 0)
    return std::nullopt;

  Real sqrt_disc = sqrt(discriminant);
  Real two_a = 2 * a;

  if (b >= 0) {
    Real s = -b - sqrt_disc;
    if (s.is_zero())
      // (-b + sqrt_disc) / 2a = 0 already in this branch.
      return std::make_pair(Real(0), -b / a);
    return std::make_pair(s / two_a, (2 * c) / s);
  }
  Real s = -b + sqrt_disc;
  if (s.is_zero())
    // (-b - sqrt_disc) / 2a = 0 already in this branch.
    return std::make_pair(-b / a, Real(0));
  return std::make_pair((2 * c) / s, s / two_a);
}

} // namespace cudaq::synth
