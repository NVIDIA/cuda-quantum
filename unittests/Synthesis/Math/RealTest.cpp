/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/Synthesis/Math/Real.h"

namespace {

using cudaq::synth::Integer;
using cudaq::synth::Real;
using cudaq::synth::i64;

// Helper: absolute difference of two Reals as a double
static double absdiff(const Real &a, const Real &b) {
  Real diff = a - b;
  return cudaq::synth::abs(diff).to_double();
}

// ============================================================
// pow_sqrt2 tests
// ============================================================

TEST(PowSqrt2Test, ZeroExponent) {
  Real result = cudaq::synth::pow_sqrt2(Integer(0));
  EXPECT_NEAR(result.to_double(), 1.0, 1e-15);
}

TEST(PowSqrt2Test, ExponentOne) {
  // (√2)^1 = √2
  Real result = cudaq::synth::pow_sqrt2(Integer(1));
  EXPECT_NEAR(result.to_double(), std::sqrt(2.0), 1e-14);
}

TEST(PowSqrt2Test, ExponentTwo) {
  // (√2)^2 = 2
  Real result = cudaq::synth::pow_sqrt2(Integer(2));
  EXPECT_NEAR(result.to_double(), 2.0, 1e-14);
}

TEST(PowSqrt2Test, ExponentThree) {
  // (√2)^3 = 2√2
  Real result = cudaq::synth::pow_sqrt2(Integer(3));
  EXPECT_NEAR(result.to_double(), 2.0 * std::sqrt(2.0), 1e-14);
}

TEST(PowSqrt2Test, ExponentFour) {
  // (√2)^4 = 4
  Real result = cudaq::synth::pow_sqrt2(Integer(4));
  EXPECT_NEAR(result.to_double(), 4.0, 1e-14);
}

TEST(PowSqrt2Test, ExponentSix) {
  // (√2)^6 = 8
  Real result = cudaq::synth::pow_sqrt2(Integer(6));
  EXPECT_NEAR(result.to_double(), 8.0, 1e-13);
}

TEST(PowSqrt2Test, ExponentEight) {
  // (√2)^8 = 16
  Real result = cudaq::synth::pow_sqrt2(Integer(8));
  EXPECT_NEAR(result.to_double(), 16.0, 1e-13);
}

TEST(PowSqrt2Test, NegativeExponentOne) {
  // (√2)^-1 = 1/√2
  Real result = cudaq::synth::pow_sqrt2(Integer(-1));
  EXPECT_NEAR(result.to_double(), 1.0 / std::sqrt(2.0), 1e-14);
}

TEST(PowSqrt2Test, NegativeExponentTwo) {
  // (√2)^-2 = 1/2
  Real result = cudaq::synth::pow_sqrt2(Integer(-2));
  EXPECT_NEAR(result.to_double(), 0.5, 1e-14);
}

TEST(PowSqrt2Test, NegativeExponentFour) {
  // (√2)^-4 = 1/4
  Real result = cudaq::synth::pow_sqrt2(Integer(-4));
  EXPECT_NEAR(result.to_double(), 0.25, 1e-14);
}

TEST(PowSqrt2Test, InverseConsistency) {
  // pow_sqrt2(k) * pow_sqrt2(-k) == 1
  for (int k = 1; k <= 8; ++k) {
    Real pos = cudaq::synth::pow_sqrt2(Integer(k));
    Real neg = cudaq::synth::pow_sqrt2(Integer(-k));
    Real product = pos * neg;
    EXPECT_NEAR(product.to_double(), 1.0, 1e-13) << "k = " << k;
  }
}

TEST(PowSqrt2Test, SquaringConsistency) {
  // pow_sqrt2(k)^2 == pow_sqrt2(2k)
  for (int k = 0; k <= 5; ++k) {
    Real pk = cudaq::synth::pow_sqrt2(Integer(k));
    Real p2k = cudaq::synth::pow_sqrt2(Integer(2 * k));
    Real squared = pk * pk;
    EXPECT_NEAR(absdiff(squared, p2k), 0.0, 1e-13) << "k = " << k;
  }
}

// ============================================================
// solve_quadratic tests
// ============================================================

TEST(SolveQuadraticTest, DistinctRoots) {
  // x^2 - 5x + 6 = 0  →  roots 2 and 3
  Real a(1.0), b(-5.0), c(6.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->first.to_double(), 2.0, 1e-12);
  EXPECT_NEAR(result->second.to_double(), 3.0, 1e-12);
}

TEST(SolveQuadraticTest, RepeatedRoot) {
  // x^2 + 2x + 1 = 0  →  root -1 (double)
  Real a(1.0), b(2.0), c(1.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->first.to_double(), -1.0, 1e-12);
  EXPECT_NEAR(result->second.to_double(), -1.0, 1e-12);
}

TEST(SolveQuadraticTest, NoRealRoots) {
  // x^2 + 1 = 0  →  no real solutions
  Real a(1.0), b(0.0), c(1.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  EXPECT_FALSE(result.has_value());
}

TEST(SolveQuadraticTest, RootsSumAndProduct) {
  // ax^2 + bx + c = 0  →  r1 + r2 = -b/a,  r1 * r2 = c/a
  Real a(2.0), b(-3.0), c(-2.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  ASSERT_TRUE(result.has_value());
  Real sum = result->first + result->second;
  Real prod = result->first * result->second;
  // sum = -b/a = 3/2
  EXPECT_NEAR(sum.to_double(), 1.5, 1e-12);
  // product = c/a = -1
  EXPECT_NEAR(prod.to_double(), -1.0, 1e-12);
}

TEST(SolveQuadraticTest, SqrtRoots) {
  // x^2 - 2 = 0  →  roots ±√2
  Real a(1.0), b(0.0), c(-2.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->first.to_double(), -std::sqrt(2.0), 1e-12);
  EXPECT_NEAR(result->second.to_double(), std::sqrt(2.0), 1e-12);
}

TEST(SolveQuadraticTest, OrderSmallRootFirst) {
  // Roots should be returned in ascending order
  Real a(1.0), b(-5.0), c(6.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  ASSERT_TRUE(result.has_value());
  EXPECT_LE(result->first.to_double(), result->second.to_double());
}

TEST(SolveQuadraticTest, RootsVerifyEquation) {
  // Plugging each root back in should give ~0
  Real a(3.0), b(1.0), c(-2.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  ASSERT_TRUE(result.has_value());
  for (const Real &r : {result->first, result->second}) {
    Real val = a * r * r + b * r + c;
    EXPECT_NEAR(val.to_double(), 0.0, 1e-11);
  }
}

TEST(SolveQuadraticTest, ZeroDiscriminantExact) {
  // Discriminant exactly zero: 4x^2 - 4x + 1 = 0  →  root 0.5
  Real a(4.0), b(-4.0), c(1.0);
  auto result = cudaq::synth::solve_quadratic(a, b, c);
  ASSERT_TRUE(result.has_value());
  EXPECT_NEAR(result->first.to_double(), 0.5, 1e-12);
  EXPECT_NEAR(result->second.to_double(), 0.5, 1e-12);
}

// ============================================================
// floor / ceil / round conversion tests
// ============================================================

TEST(ConversionTest, FloorPositive) {
  Real x(3.7);
  Integer result = cudaq::synth::floor_to_integer(x);
  EXPECT_EQ(static_cast<i64>(result), 3LL);
}

TEST(ConversionTest, FloorNegative) {
  Real x(-3.2);
  Integer result = cudaq::synth::floor_to_integer(x);
  EXPECT_EQ(static_cast<i64>(result), -4LL);
}

TEST(ConversionTest, CeilPositive) {
  Real x(3.2);
  Integer result = cudaq::synth::ceil_to_integer(x);
  EXPECT_EQ(static_cast<i64>(result), 4LL);
}

TEST(ConversionTest, CeilNegative) {
  Real x(-3.7);
  Integer result = cudaq::synth::ceil_to_integer(x);
  EXPECT_EQ(static_cast<i64>(result), -3LL);
}

TEST(ConversionTest, RoundHalfUp) {
  Real x(2.5);
  Integer result = cudaq::synth::round_to_integer(x);
  EXPECT_EQ(static_cast<i64>(result), 2LL); // MPFR_RNDN rounds to even
}

TEST(ConversionTest, RoundPositive) {
  Real x(3.6);
  Integer result = cudaq::synth::round_to_integer(x);
  EXPECT_EQ(static_cast<i64>(result), 4LL);
}

TEST(ConversionTest, RoundNegative) {
  Real x(-2.8);
  Integer result = cudaq::synth::round_to_integer(x);
  EXPECT_EQ(static_cast<i64>(result), -3LL);
}

// ============================================================
// Math functions: abs, sqrt, log, sin, cos
// ============================================================

TEST(MathFuncTest, AbsPositive) {
  Real x(3.5);
  EXPECT_NEAR(cudaq::synth::abs(x).to_double(), 3.5, 1e-15);
}

TEST(MathFuncTest, AbsNegative) {
  Real x(-7.25);
  EXPECT_NEAR(cudaq::synth::abs(x).to_double(), 7.25, 1e-15);
}

TEST(MathFuncTest, SqrtFour) {
  Real x(4.0);
  EXPECT_NEAR(cudaq::synth::sqrt(x).to_double(), 2.0, 1e-14);
}

TEST(MathFuncTest, SqrtTwo) {
  Real x(2.0);
  EXPECT_NEAR(cudaq::synth::sqrt(x).to_double(), std::sqrt(2.0), 1e-14);
}

TEST(MathFuncTest, LogE) {
  // ln(e) == 1
  Real e(std::exp(1.0));
  EXPECT_NEAR(cudaq::synth::log(e).to_double(), 1.0, 1e-13);
}

TEST(MathFuncTest, LogOne) {
  Real one(1.0);
  EXPECT_NEAR(cudaq::synth::log(one).to_double(), 0.0, 1e-15);
}

TEST(MathFuncTest, SinZero) {
  Real zero(0.0);
  EXPECT_NEAR(cudaq::synth::sin(zero).to_double(), 0.0, 1e-15);
}

TEST(MathFuncTest, SinPiOver2) {
  Real x = Real::pi() / Real(2.0);
  EXPECT_NEAR(cudaq::synth::sin(x).to_double(), 1.0, 1e-14);
}

TEST(MathFuncTest, CosZero) {
  Real zero(0.0);
  EXPECT_NEAR(cudaq::synth::cos(zero).to_double(), 1.0, 1e-15);
}

TEST(MathFuncTest, CosPi) {
  EXPECT_NEAR(cudaq::synth::cos(Real::pi()).to_double(), -1.0, 1e-14);
}

TEST(MathFuncTest, SinCosIdentity) {
  // sin^2(x) + cos^2(x) == 1
  Real x(1.23456);
  Real s = cudaq::synth::sin(x);
  Real c = cudaq::synth::cos(x);
  Real identity = s * s + c * c;
  EXPECT_NEAR(identity.to_double(), 1.0, 1e-13);
}

// ============================================================
// Utility / predicate tests
// ============================================================

TEST(UtilityTest, IsZero) {
  Real zero(0.0);
  EXPECT_TRUE(zero.is_zero());
  EXPECT_FALSE(Real(1.0).is_zero());
}

TEST(UtilityTest, IsNan) {
  Real nan_val;
  mpfr_set_nan(nan_val.get_mpfr());
  EXPECT_TRUE(nan_val.is_nan());
  EXPECT_FALSE(Real(1.0).is_nan());
}

TEST(UtilityTest, IsInf) {
  EXPECT_TRUE(Real::inf().is_inf());
  EXPECT_TRUE(Real::neg_inf().is_inf());
  EXPECT_FALSE(Real(1.0).is_inf());
}

TEST(UtilityTest, IsFinite) {
  EXPECT_TRUE(Real(42.0).is_finite());
  EXPECT_FALSE(Real::inf().is_finite());
}

TEST(UtilityTest, Sqrt2Constant) {
  Real s2 = Real::sqrt2();
  Real sq = s2 * s2;
  EXPECT_NEAR(sq.to_double(), 2.0, 1e-13);
}

} // namespace
