/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "Math/Ring/Zsqrt2.h"

namespace {

using namespace cudaq::synth;

// λ = 1 + √2, the fundamental unit of Z[√2] (norm = -1).
static const ZSqrt2 kLambda{Integer(1), Integer(1)};
static const ZSqrt2 kZero{Integer(0), Integer(0)};
static const ZSqrt2 kOne{Integer(1), Integer(0)};

static long long ll(const Integer &n) { return static_cast<long long>(static_cast<i64>(n)); }

// ============================================================
// Arithmetic operators
// ============================================================

TEST(ZSqrt2ArithTest, AddComponents) {
  ZSqrt2 x(3, 5), y(1, -2);
  ZSqrt2 r = x + y;
  EXPECT_EQ(ll(r.a()), 4LL);
  EXPECT_EQ(ll(r.b()), 3LL);
}

TEST(ZSqrt2ArithTest, SubComponents) {
  ZSqrt2 x(3, 5), y(1, 2);
  ZSqrt2 r = x - y;
  EXPECT_EQ(ll(r.a()), 2LL);
  EXPECT_EQ(ll(r.b()), 3LL);
}

TEST(ZSqrt2ArithTest, UnaryNegate) {
  ZSqrt2 x(3, -5);
  ZSqrt2 r = -x;
  EXPECT_EQ(ll(r.a()), -3LL);
  EXPECT_EQ(ll(r.b()), 5LL);
}

TEST(ZSqrt2ArithTest, MultiplyByOne) { EXPECT_EQ(kLambda * kOne, kLambda); }

TEST(ZSqrt2ArithTest, MultiplyByZero) { EXPECT_EQ(kLambda * kZero, kZero); }

TEST(ZSqrt2ArithTest, MultiplyFormula) {
  // (a + b√2)(c + d√2) = (ac + 2bd) + (ad + bc)√2
  ZSqrt2 x(2, 3), y(5, 1);
  ZSqrt2 r = x * y;
  // a = 2*5 + 2*3*1 = 10 + 6 = 16
  // b = 2*1 + 3*5 = 2 + 15 = 17
  EXPECT_EQ(ll(r.a()), 16LL);
  EXPECT_EQ(ll(r.b()), 17LL);
}

TEST(ZSqrt2ArithTest, MultiplyCommutativity) {
  ZSqrt2 x(3, 7), y(2, -1);
  EXPECT_EQ(x * y, y * x);
}

TEST(ZSqrt2ArithTest, MultiplyAssociativity) {
  ZSqrt2 x(1, 2), y(3, -1), z(0, 4);
  EXPECT_EQ((x * y) * z, x * (y * z));
}

TEST(ZSqrt2ArithTest, DistributivityOverAdd) {
  ZSqrt2 x(2, 1), y(3, -2), z(-1, 5);
  EXPECT_EQ(x * (y + z), x * y + x * z);
}

TEST(ZSqrt2ArithTest, LambdaSquared) {
  // λ² = (1+√2)² = 3 + 2√2
  ZSqrt2 l2 = kLambda * kLambda;
  EXPECT_EQ(ll(l2.a()), 3LL);
  EXPECT_EQ(ll(l2.b()), 2LL);
}

// ============================================================
// norm()
// ============================================================

TEST(ZSqrt2NormTest, OfOne) { EXPECT_EQ(ll(kOne.norm()), 1LL); }

TEST(ZSqrt2NormTest, OfZero) { EXPECT_EQ(ll(kZero.norm()), 0LL); }

TEST(ZSqrt2NormTest, OfSqrt2) {
  // (0 + 1·√2): norm = 0² - 2·1² = -2
  ZSqrt2 sq2(0, 1);
  EXPECT_EQ(ll(sq2.norm()), -2LL);
}

TEST(ZSqrt2NormTest, OfLambda) {
  // λ = 1+√2: norm = 1 - 2 = -1
  EXPECT_EQ(ll(kLambda.norm()), -1LL);
}

TEST(ZSqrt2NormTest, OfLambdaSquared) {
  // λ² = 3+2√2: norm = 9 - 8 = 1
  ZSqrt2 l2(3, 2);
  EXPECT_EQ(ll(l2.norm()), 1LL);
}

TEST(ZSqrt2NormTest, Multiplicativity) {
  // N(αβ) = N(α) · N(β)
  ZSqrt2 x(2, 3), y(5, -1);
  Integer nxy = (x * y).norm();
  Integer nxny = x.norm() * y.norm();
  EXPECT_EQ(nxy, nxny);
}

TEST(ZSqrt2NormTest, EqualsSelfTimesConj) {
  // N(α) = α · α^●, result should be a pure integer (b-part zero)
  ZSqrt2 x(3, 5);
  ZSqrt2 prod = x * x.conj_sq2();
  EXPECT_EQ(prod, ZSqrt2(x.norm(), Integer(0)));
}

// ============================================================
// conj_sq2()
// ============================================================

TEST(ZSqrt2ConjTest, FlipsB) {
  ZSqrt2 x(4, -7);
  ZSqrt2 c = x.conj_sq2();
  EXPECT_EQ(ll(c.a()), 4LL);
  EXPECT_EQ(ll(c.b()), 7LL);
}

TEST(ZSqrt2ConjTest, DoubleCojugateIsIdentity) {
  ZSqrt2 x(3, -9);
  EXPECT_EQ(x.conj_sq2().conj_sq2(), x);
}

TEST(ZSqrt2ConjTest, ConjOfOneIsOne) { EXPECT_EQ(kOne.conj_sq2(), kOne); }

TEST(ZSqrt2ConjTest, ProductWithConjIsNorm) {
  ZSqrt2 x(5, 3);
  ZSqrt2 prod = x * x.conj_sq2();
  EXPECT_EQ(ll(prod.a()), ll(x.norm()));
  EXPECT_EQ(ll(prod.b()), 0LL);
}

// ============================================================
// inv()
// ============================================================

TEST(ZSqrt2InvTest, InvOfOne) {
  auto r = inv(kOne);
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(*r, kOne);
}

TEST(ZSqrt2InvTest, InvOfLambdaIsLambdaInverse) {
  // λ^-1 = -λ^● = -(1,-1) = (-1, 1)
  auto r = inv(kLambda);
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(ll(r->a()), -1LL);
  EXPECT_EQ(ll(r->b()), 1LL);
}

TEST(ZSqrt2InvTest, InvOfLambdaSquared) {
  // λ² = (3,2), norm=1: inv = conj = (3,-2)
  ZSqrt2 l2(3, 2);
  auto r = inv(l2);
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(ll(r->a()), 3LL);
  EXPECT_EQ(ll(r->b()), -2LL);
}

TEST(ZSqrt2InvTest, InvTimesOriginalIsOne) {
  for (auto &x : {ZSqrt2(1, 0), ZSqrt2(1, 1), ZSqrt2(3, 2), ZSqrt2(-1, 1)}) {
    auto r = inv(x);
    if (succeeded(r))
      EXPECT_EQ(x * (*r), kOne)
          << "x = (" << ll(x.a()) << "," << ll(x.b()) << ")";
  }
}

TEST(ZSqrt2InvTest, InvOfNonUnitFails) {
  EXPECT_TRUE(failed(inv(ZSqrt2(1, 2)))); // norm = 1-8 = -7
  EXPECT_TRUE(failed(inv(ZSqrt2(2, 0)))); // norm = 4
  EXPECT_TRUE(failed(inv(kZero)));        // norm = 0
}

// ============================================================
// pow()
// ============================================================

TEST(ZSqrt2PowTest, PowZero) { EXPECT_EQ(pow(kLambda, Integer(0)), kOne); }

TEST(ZSqrt2PowTest, PowOne) { EXPECT_EQ(pow(kLambda, Integer(1)), kLambda); }

TEST(ZSqrt2PowTest, PowTwo) {
  // λ² = 3+2√2
  ZSqrt2 r = pow(kLambda, Integer(2));
  EXPECT_EQ(ll(r.a()), 3LL);
  EXPECT_EQ(ll(r.b()), 2LL);
}

TEST(ZSqrt2PowTest, PowThree) {
  // λ³ = λ²·λ = (3+2√2)(1+√2) = 3+3√2+2√2+2·2 = 7+5√2
  ZSqrt2 r = pow(kLambda, Integer(3));
  EXPECT_EQ(ll(r.a()), 7LL);
  EXPECT_EQ(ll(r.b()), 5LL);
}

TEST(ZSqrt2PowTest, PowNegativeOne) {
  // λ^-1 = (-1, 1)
  ZSqrt2 r = pow(kLambda, Integer(-1));
  EXPECT_EQ(ll(r.a()), -1LL);
  EXPECT_EQ(ll(r.b()), 1LL);
}

TEST(ZSqrt2PowTest, PowNegativeTwo) {
  // λ^-2 = (λ^-1)^2 = (-1+√2)^2 = 1-2√2+2 = 3-2√2
  ZSqrt2 r = pow(kLambda, Integer(-2));
  EXPECT_EQ(ll(r.a()), 3LL);
  EXPECT_EQ(ll(r.b()), -2LL);
}

TEST(ZSqrt2PowTest, PowPositivePlusNegativeCancels) {
  // λ^k · λ^-k = 1 for several k
  for (int k = 1; k <= 6; ++k) {
    ZSqrt2 pos = pow(kLambda, Integer(k));
    ZSqrt2 neg = pow(kLambda, Integer(-k));
    EXPECT_EQ(pos * neg, kOne) << "k = " << k;
  }
}

TEST(ZSqrt2PowTest, NormOfPowerIsNormPow) {
  // N(λ^k) = N(λ)^k = (-1)^k
  for (int k = 0; k <= 6; ++k) {
    Integer n = pow(kLambda, Integer(k)).norm();
    long long expected = (k % 2 == 0) ? 1LL : -1LL;
    EXPECT_EQ(ll(n), expected) << "k = " << k;
  }
}

// ============================================================
// sqrt()
// ============================================================

TEST(ZSqrt2SqrtTest, SqrtOfZero) {
  FailureOr<ZSqrt2> r = sqrt(kZero);
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(*r, kZero);
}

TEST(ZSqrt2SqrtTest, SqrtOfOne) {
  FailureOr<ZSqrt2> r = sqrt(kOne);
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(*r * *r, kOne);
}

TEST(ZSqrt2SqrtTest, SqrtOf2AsInteger) {
  // ZSqrt2(2,0) = 2: sqrt is (0,1) because (0+1·√2)² = 2+0√2
  FailureOr<ZSqrt2> r = sqrt(ZSqrt2(2, 0));
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(*r * *r, ZSqrt2(2, 0));
}

TEST(ZSqrt2SqrtTest, SqrtOfLambdaSquared) {
  // λ² = (3,2): sqrt should be λ = (1,1)
  FailureOr<ZSqrt2> r = sqrt(ZSqrt2(3, 2));
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(*r * *r, ZSqrt2(3, 2));
}

TEST(ZSqrt2SqrtTest, SqrtOfLambdaFourth) {
  // λ^4 = (17,12): sqrt should be λ² = (3,2)
  ZSqrt2 l4 = pow(kLambda, Integer(4));
  FailureOr<ZSqrt2> r = sqrt(l4);
  ASSERT_TRUE(succeeded(r));
  EXPECT_EQ(*r * *r, l4);
}

TEST(ZSqrt2SqrtTest, SqrtResultSquaresToInput) {
  // Exhaustive: any element that has a sqrt must square back correctly
  for (int a = 0; a <= 10; ++a) {
    for (int b = -5; b <= 5; ++b) {
      ZSqrt2 x(a, b);
      FailureOr<ZSqrt2> r = sqrt(x);
      if (succeeded(r))
        EXPECT_EQ(*r * *r, x) << "x = (" << a << "," << b << ")";
    }
  }
}

TEST(ZSqrt2SqrtTest, SqrtOfNonSquareIsNullopt) {
  // ZSqrt2(3,0) = 3 has no square root in Z[√2]
  EXPECT_FALSE(succeeded(sqrt(ZSqrt2(3, 0))));
  // ZSqrt2(5,0) = 5 likewise
  EXPECT_FALSE(succeeded(sqrt(ZSqrt2(5, 0))));
}

// ============================================================
// divmod() / operator/ / operator%
// ============================================================

TEST(ZSqrt2DivmodTest, DivideByOne) {
  ZSqrt2 x(7, 3);
  auto [q, r] = divmod(x, kOne);
  EXPECT_EQ(q, x);
  EXPECT_EQ(r, kZero);
}

TEST(ZSqrt2DivmodTest, DivideByItself) {
  ZSqrt2 x(3, 2);
  auto [q, r] = divmod(x, x);
  EXPECT_EQ(q, kOne);
  EXPECT_EQ(r, kZero);
}

TEST(ZSqrt2DivmodTest, LambdaCubedDivLambda) {
  // λ³ = (7,5), λ = (1,1): quotient λ², remainder 0
  ZSqrt2 l3 = pow(kLambda, Integer(3));
  auto [q, r] = divmod(l3, kLambda);
  EXPECT_EQ(r, kZero);
  EXPECT_EQ(q * kLambda, l3);
}

TEST(ZSqrt2DivmodTest, RemainderNormLessThanDivisorNorm) {
  // Euclidean property: |N(r)| < |N(d)| for non-zero d
  ZSqrt2 dividend(13, 7), divisor(3, 2);
  auto [q, r] = divmod(dividend, divisor);
  if (!(r == kZero)) {
    long long nr = std::abs(ll(r.norm()));
    long long nd = std::abs(ll(divisor.norm()));
    EXPECT_LT(nr, nd);
  }
}

TEST(ZSqrt2DivmodTest, QuotientRemainderReconstruct) {
  // dividend == divisor * quotient + remainder
  for (int da = -5; da <= 5; ++da) {
    for (int db = -3; db <= 3; ++db) {
      ZSqrt2 divisor(da, db);
      if (divisor == kZero)
        continue;
      ZSqrt2 dividend(7, 3);
      auto [q, r] = divmod(dividend, divisor);
      EXPECT_EQ(divisor * q + r, dividend)
          << "divisor=(" << da << "," << db << ")";
    }
  }
}

// ============================================================
// are_associates() — associateness
// ============================================================

TEST(ZSqrt2SimTest, UnitAssociatesOfLambda) {
  // λ and λ^-1 are associates (each divides the other)
  EXPECT_TRUE(are_associates(kLambda, pow(kLambda, Integer(-1))));
}

TEST(ZSqrt2SimTest, LambdaAndNegativeLambda) {
  EXPECT_TRUE(are_associates(kLambda, -kLambda));
}

TEST(ZSqrt2SimTest, NonAssociates) {
  // 2 and 3 are not associates in Z[√2]: neither divides the other
  EXPECT_FALSE(are_associates(ZSqrt2(2, 0), ZSqrt2(3, 0)));
}

TEST(ZSqrt2SimTest, SameElementIsAssociate) {
  ZSqrt2 x(5, 3);
  EXPECT_TRUE(are_associates(x, x));
}

// ============================================================
// gcd()
// ============================================================

TEST(ZSqrt2GcdTest, GcdWithZero) {
  // gcd(a, 0) should return a (up to are_associates)
  ZSqrt2 x(3, 2);
  ZSqrt2 g = gcd(x, kZero);
  EXPECT_TRUE(are_associates(g, x));
}

TEST(ZSqrt2GcdTest, GcdWithItself) {
  ZSqrt2 x(1, 1);
  ZSqrt2 g = gcd(x, x);
  EXPECT_TRUE(are_associates(g, x));
}

TEST(ZSqrt2GcdTest, GcdDividesOperands) {
  ZSqrt2 a = pow(kLambda, Integer(3)); // λ³ = (7,5)
  ZSqrt2 b = pow(kLambda, Integer(2)); // λ² = (3,2)
  ZSqrt2 g = gcd(a, b);
  // g must divide both a and b (i.e., remainder = 0)
  EXPECT_EQ(a % g, kZero);
  EXPECT_EQ(b % g, kZero);
}

TEST(ZSqrt2GcdTest, GcdIsSimToExpected) {
  // gcd(λ³, λ²) ~ λ²
  ZSqrt2 a = pow(kLambda, Integer(3));
  ZSqrt2 b = pow(kLambda, Integer(2));
  ZSqrt2 g = gcd(a, b);
  EXPECT_TRUE(are_associates(g, b));
}

// ============================================================
// Comparison operators
// ============================================================

TEST(ZSqrt2CmpTest, EqualElements) {
  ZSqrt2 x(3, 5), y(3, 5);
  EXPECT_TRUE(x == y);
  EXPECT_FALSE(x != y);
}

TEST(ZSqrt2CmpTest, OrderByRealValue) {
  // 1+√2 ≈ 2.414, 3 = 3: λ < (3,0)
  EXPECT_TRUE(kLambda < ZSqrt2(3, 0));
  EXPECT_FALSE(ZSqrt2(3, 0) < kLambda);
}

TEST(ZSqrt2CmpTest, NegativeVsPositive) { EXPECT_TRUE(ZSqrt2(-1, 0) < kOne); }

TEST(ZSqrt2CmpTest, OrderConsistentWithReal) {
  // Sample several pairs and verify the real-valued order agrees
  auto real_val = [](const ZSqrt2 &z) {
    return static_cast<double>(ll(z.a())) +
           static_cast<double>(ll(z.b())) * 1.41421356237;
  };
  ZSqrt2 elems[] = {ZSqrt2(-2, 1), ZSqrt2(0, 1), ZSqrt2(1, 1), ZSqrt2(3, 0),
                    ZSqrt2(3, 2)};
  for (auto &x : elems) {
    for (auto &y : elems) {
      bool expected = real_val(x) < real_val(y) - 1e-9;
      bool actual = x < y;
      if (expected)
        EXPECT_TRUE(actual) << "(" << ll(x.a()) << "," << ll(x.b()) << ") < ("
                            << ll(y.a()) << "," << ll(y.b()) << ")";
    }
  }
}

TEST(ZSqrt2CmpTest, LessEqAndGreater) {
  ZSqrt2 x(1, 0), y(1, 1);
  EXPECT_TRUE(x <= y);
  EXPECT_TRUE(y > x);
  EXPECT_TRUE(y >= x);
  EXPECT_TRUE(x <= x);
  EXPECT_TRUE(x >= x);
}

// ============================================================
// parity()
// ============================================================

TEST(ZSqrt2ParityTest, ParityIsLowBitOfA) {
  EXPECT_EQ(ll(ZSqrt2(4, 7).parity()), 0LL); // a = 4, even
  EXPECT_EQ(ll(ZSqrt2(3, 8).parity()), 1LL); // a = 3, odd
}

// ============================================================
// to_real()
// ============================================================

TEST(ZSqrt2ToRealTest, OneIsOne) {
  EXPECT_NEAR(to_real(kOne).to_double(), 1.0, 1e-12);
}

TEST(ZSqrt2ToRealTest, LambdaIsOnePlusSqrt2) {
  EXPECT_NEAR(to_real(kLambda).to_double(), 1.0 + std::sqrt(2.0), 1e-12);
}

TEST(ZSqrt2ToRealTest, MultiplyConsistentWithReal) {
  ZSqrt2 x(2, 3), y(5, -1);
  double rxy = to_real(x * y).to_double();
  double rx = to_real(x).to_double();
  double ry = to_real(y).to_double();
  EXPECT_NEAR(rxy, rx * ry, 1e-9);
}

} // namespace
