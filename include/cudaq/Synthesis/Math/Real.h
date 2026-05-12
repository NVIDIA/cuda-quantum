/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Integer.h"

// Enable 64-bit integer types in MPFR (required for mpfr_set_sj/mpfr_get_sj,
// which accept intmax_t; needed to bridge i64 → MPFR without truncation on
// LLP64 platforms where long is 32-bit).
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
/// MPFR-based floating point class with arbitrary precision arithmetic.
class Real {
private:
  mpfr_t value_;
  static mpfr_prec_t default_precision_;
  // Internal tag to construct directly with a specified precision without
  // re-init overhead
  struct direct_init_tag {};
  Real(direct_init_tag, mpfr_prec_t prec) {
    mpfr_init2(value_, prec);
    mpfr_set_zero(value_, 1);
  }

public:
  // Static precision management
  static void set_default_precision(mpfr_prec_t prec) {
    default_precision_ = prec;
  }
  static mpfr_prec_t get_default_precision() { return default_precision_; }

  // Static factory method for creating Float with specific precision
  static Real with_precision(mpfr_prec_t precision, double val = 0.0) {
    Real result(direct_init_tag{}, precision);
    if (val == 0.0) {
      // Already zeroed
    } else {
      mpfr_set_d(result.value_, val, MPFR_RNDN);
    }
    return result;
  }

  // Constructors
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

  // Copy constructor
  Real(const Real &other) {
    mpfr_init2(value_, mpfr_get_prec(other.value_));
    mpfr_set(value_, other.value_, MPFR_RNDN);
  }

  // Move constructor
  Real(Real &&other) noexcept {
    std::memcpy(&value_, &other.value_, sizeof(mpfr_t));
    other.value_[0]._mpfr_d = nullptr; // Mark moved-from
  }

  // Destructor
  ~Real() {
    if (value_[0]._mpfr_d != nullptr)
      mpfr_clear(value_);
  }

  // Assignment operators
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
    if (this != &other) {
      mpfr_swap(value_, other.value_);
    }
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

  // Conversion operators
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

  // Explicit conversion methods
  [[nodiscard]] double to_double() const noexcept {
    return mpfr_get_d(value_, MPFR_RNDN);
  }

  // Arithmetic operators
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

  // Compound assignment operators
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

  // Compound assignment with double (avoids temporary Float allocations)
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

  // Compound assignment with i32 (uses mpfr_*_si helpers)
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

  // Increment and decrement operators
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

  // Comparison operators
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

  // Unary operators
  Real operator-() const {
    Real result(direct_init_tag{}, mpfr_get_prec(value_));
    mpfr_neg(result.value_, value_, MPFR_RNDN);
    return result;
  }

  Real operator+() const { return *this; }

  bool operator!() const { return mpfr_zero_p(value_) != 0; }

  // Utility functions
  bool is_nan() const { return mpfr_nan_p(value_) != 0; }
  bool is_inf() const { return mpfr_inf_p(value_) != 0; }
  bool is_zero() const { return mpfr_zero_p(value_) != 0; }
  bool is_finite() const { return mpfr_number_p(value_) != 0; }

  mpfr_prec_t precision() const { return mpfr_get_prec(value_); }
  void set_precision(mpfr_prec_t prec) {
    mpfr_prec_round(value_, prec, MPFR_RNDN);
  }

  std::string to_string(i32 digits = 40) const;

  // Give access to internal mpfr_t for advanced operations
  const mpfr_t &get_mpfr() const { return value_; }
  mpfr_t &get_mpfr() { return value_; }

  // Mathematical constants (function-local statics avoid static init order
  // issues; initialized at default precision on first use)
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
  // Infinity constants (precision-independent: ∞ is ∞ at any precision).
  // Use mpfr_set_inf directly rather than converting via double.
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

// Static member initialization (defined in real.cpp)

// Global mathematical functions (mimicking std namespace functions)
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

// ADL-friendly swap function
inline void swap(Real &a, Real &b) noexcept {
  mpfr_swap(a.get_mpfr(), b.get_mpfr());
}

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

/// Computes (√2)^k in arbitrary precision.
inline Real pow_sqrt2(const Integer &k) {
  if (k == 0)
    return Real(1);

  bool negative = k < 0;
  Integer abs_k = negative ? -k : k;

  // 2^(abs_k/2) in O(1): mpfr_mul_2ui shifts the MPFR exponent field directly.
  // In practice abs_k/2 always fits in unsigned long (bounded by precision
  // bits).
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

// Comparison operators with double (optimized to avoid temporaries)
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

// Optimized mixed operations with double (avoid temporary Float creation)
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

// Optimized mixed operations with i32 using mpfr_*_si helpers
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

// Comparison operators with i32 (optimized to avoid temporaries)
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

// Mixed operations between Integer and Float (optimized to avoid temporaries)
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

// Comparison operators between Integer and Float
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

/// Solves ax² + bx + c = 0, returning the two roots.
/// Note: The order of the roots is important! The smaller root comes first.
inline std::optional<std::pair<Real, Real>>
solve_quadratic(const Real &a, const Real &b, const Real &c) {
  Real discriminant = b * b - 4 * a * c;
  if (discriminant < 0)
    return std::nullopt;

  Real sqrt_disc = sqrt(discriminant);
  Real two_a = 2 * a;

  // Numerically stable: pick the s that avoids cancellation (both terms same
  // sign), then derive the other root via Vieta (r1*r2 = c/a → r2 = 2c/s).
  // Guard: when s == 0 (double root at 0, or b == sqrt_disc), Vieta's formula
  // 2c/s produces 0/0. Fall back to the direct formula for the second root.

  if (b >= 0) {
    Real s = -b - sqrt_disc;
    if (s.is_zero() )
      return std::make_pair(Real(0), -b / a); // direct: (-b + sqrt_disc) / 2a = 0 already
    return std::make_pair(s / two_a, (2 * c) / s);
  }
  Real s = -b + sqrt_disc;
  if (s.is_zero())
    return std::make_pair(-b / a, Real(0)); // direct: (-b - sqrt_disc) / 2a = 0 already
  return std::make_pair((2 * c) / s, s / two_a);
}

} // namespace cudaq::synth
