/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/Synthesis/Math/Integer.h"

namespace {

using cudaq::synth::i64;
using cudaq::synth::Integer;

static long long ll(const Integer &n) {
  return static_cast<long long>(static_cast<i64>(n));
}

// ============================================================
// ntz — number of trailing zeros
// ============================================================

TEST(NtzTest, Zero) { EXPECT_EQ(ll(cudaq::synth::ntz(Integer(0))), 0LL); }

TEST(NtzTest, One) {
  // 1 = 0b1 — no trailing zeros
  EXPECT_EQ(ll(cudaq::synth::ntz(Integer(1))), 0LL);
}

TEST(NtzTest, Two) {
  // 2 = 0b10
  EXPECT_EQ(ll(cudaq::synth::ntz(Integer(2))), 1LL);
}

TEST(NtzTest, Four) {
  // 4 = 0b100
  EXPECT_EQ(ll(cudaq::synth::ntz(Integer(4))), 2LL);
}

TEST(NtzTest, Six) {
  // 6 = 0b110 — one trailing zero
  EXPECT_EQ(ll(cudaq::synth::ntz(Integer(6))), 1LL);
}

TEST(NtzTest, Twelve) {
  // 12 = 0b1100
  EXPECT_EQ(ll(cudaq::synth::ntz(Integer(12))), 2LL);
}

TEST(NtzTest, PowersOfTwo) {
  for (int k = 0; k <= 10; ++k) {
    Integer n(static_cast<i64>(1LL << k));
    EXPECT_EQ(ll(cudaq::synth::ntz(n)), static_cast<long long>(k))
        << "k = " << k;
  }
}

TEST(NtzTest, OddAlwaysZero) {
  for (int v : {1, 3, 7, 11, 99}) {
    EXPECT_EQ(ll(cudaq::synth::ntz(Integer(v))), 0LL) << "v = " << v;
  }
}

// ============================================================
// sign
// ============================================================

TEST(SignTest, Positive) {
  EXPECT_EQ(ll(cudaq::synth::sign(Integer(42))), 1LL);
}

TEST(SignTest, Negative) {
  EXPECT_EQ(ll(cudaq::synth::sign(Integer(-7))), -1LL);
}

TEST(SignTest, Zero) { EXPECT_EQ(ll(cudaq::synth::sign(Integer(0))), 0LL); }

TEST(SignTest, One) { EXPECT_EQ(ll(cudaq::synth::sign(Integer(1))), 1LL); }

TEST(SignTest, MinusOne) {
  EXPECT_EQ(ll(cudaq::synth::sign(Integer(-1))), -1LL);
}

// ============================================================
// floorsqrt
// ============================================================

TEST(FloorsqrtTest, Zero) {
  EXPECT_EQ(ll(cudaq::synth::floorsqrt(Integer(0))), 0LL);
}

TEST(FloorsqrtTest, One) {
  EXPECT_EQ(ll(cudaq::synth::floorsqrt(Integer(1))), 1LL);
}

TEST(FloorsqrtTest, PerfectSquares) {
  for (int k = 0; k <= 10; ++k) {
    EXPECT_EQ(ll(cudaq::synth::floorsqrt(Integer(k * k))),
              static_cast<long long>(k))
        << "k = " << k;
  }
}

TEST(FloorsqrtTest, NonPerfectSquareFloor) {
  // floor(`sqrt`(8)) = 2, floor(`sqrt`(15)) = 3, floor(`sqrt`(24)) = 4
  EXPECT_EQ(ll(cudaq::synth::floorsqrt(Integer(8))), 2LL);
  EXPECT_EQ(ll(cudaq::synth::floorsqrt(Integer(15))), 3LL);
  EXPECT_EQ(ll(cudaq::synth::floorsqrt(Integer(24))), 4LL);
}

TEST(FloorsqrtTest, ResultSquaredLeInput) {
  // floor(`sqrt`(n))^2 <= n < (floor(`sqrt`(n))+1)^2
  for (int n = 0; n <= 50; ++n) {
    Integer result = cudaq::synth::floorsqrt(Integer(n));
    long long r = ll(result);
    EXPECT_LE(r * r, static_cast<long long>(n)) << "n = " << n;
    EXPECT_GT((r + 1) * (r + 1), static_cast<long long>(n)) << "n = " << n;
  }
}

// ============================================================
// floordiv
// ============================================================

TEST(FloordivTest, PositiveDivisor) {
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(7), Integer(2))), 3LL);
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(6), Integer(2))), 3LL);
}

TEST(FloordivTest, NegativeDividend) {
  // floor(-7 / 2) = floor(-3.5) = -4
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(-7), Integer(2))), -4LL);
  // floor(-6 / 2) = -3
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(-6), Integer(2))), -3LL);
}

TEST(FloordivTest, NegativeDivisor) {
  // floor(7 / -2) = floor(-3.5) = -4
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(7), Integer(-2))), -4LL);
}

TEST(FloordivTest, BothNegative) {
  // floor(-7 / -2) = floor(3.5) = 3
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(-7), Integer(-2))), 3LL);
}

TEST(FloordivTest, IntOverloadPositive) {
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(7), 2)), 3LL);
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(-7), 2)), -4LL);
}

TEST(FloordivTest, IntOverloadPowerOfTwo) {
  // Fast path: divisor is a positive power of 2
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(7), 4)), 1LL);
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(-7), 4)), -2LL);
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(8), 4)), 2LL);
  EXPECT_EQ(ll(cudaq::synth::floordiv(Integer(16), 8)), 2LL);
}

TEST(FloordivTest, IntOverloadMatchesIntegerOverload) {
  for (int x = -20; x <= 20; ++x) {
    for (int y : {1, 2, 3, 4, 7}) {
      long long expected = ll(cudaq::synth::floordiv(Integer(x), Integer(y)));
      long long actual = ll(cudaq::synth::floordiv(Integer(x), y));
      EXPECT_EQ(actual, expected) << "x=" << x << " y=" << y;
    }
  }
}

// ============================================================
// rounddiv
// ============================================================

TEST(RounddivTest, ExactDivision) {
  EXPECT_EQ(ll(cudaq::synth::rounddiv(Integer(6), Integer(2))), 3LL);
  EXPECT_EQ(ll(cudaq::synth::rounddiv(Integer(-6), Integer(2))), -3LL);
}

TEST(RounddivTest, RoundsUp) {
  // 7/2 = 3.5 → rounds to 4
  EXPECT_EQ(ll(cudaq::synth::rounddiv(Integer(7), Integer(2))), 4LL);
}

TEST(RounddivTest, RoundsDown) {
  // 5/3 = 1.666... → rounds to 2
  EXPECT_EQ(ll(cudaq::synth::rounddiv(Integer(5), Integer(3))), 2LL);
  // 1/3 = 0.333... → rounds to 0
  EXPECT_EQ(ll(cudaq::synth::rounddiv(Integer(1), Integer(3))), 0LL);
}

TEST(RounddivTest, NegativeDividend) {
  // -7/2 = -3.5, floordiv(-7+1, 2) = floordiv(-6, 2) = -3
  EXPECT_EQ(ll(cudaq::synth::rounddiv(Integer(-7), Integer(2))), -3LL);
}

TEST(RounddivTest, NegativeDivisor) {
  // rounddiv(7, -2): y < 0 branch → floordiv(7 - floordiv(2, 2), -2)
  //                                = floordiv(7 - 1, -2) = floordiv(6, -2) = -3
  EXPECT_EQ(ll(cudaq::synth::rounddiv(Integer(7), Integer(-2))), -3LL);
}

TEST(RounddivTest, ResultNearExact) {
  // rounddiv(n, d) should be the closest integer to n/d
  for (int n = -20; n <= 20; ++n) {
    for (int d : {1, 2, 3, 5}) {
      long long r = ll(cudaq::synth::rounddiv(Integer(n), Integer(d)));
      double exact = static_cast<double>(n) / static_cast<double>(d);
      // r must be one of the two nearest integers
      EXPECT_LE(std::abs(r - exact), 0.5 + 1e-9) << "n=" << n << " d=" << d;
    }
  }
}

// ============================================================
// gcd
// ============================================================

TEST(GcdTest, BasicCases) {
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(12), Integer(8))), 4LL);
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(9), Integer(6))), 3LL);
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(100), Integer(75))), 25LL);
}

TEST(GcdTest, Coprime) {
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(13), Integer(7))), 1LL);
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(1), Integer(99))), 1LL);
}

TEST(GcdTest, OneOperandZero) {
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(0), Integer(5))), 5LL);
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(7), Integer(0))), 7LL);
}

TEST(GcdTest, NegativeInputs) {
  // gcd is defined over absolute values
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(-12), Integer(8))), 4LL);
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(12), Integer(-8))), 4LL);
  EXPECT_EQ(ll(cudaq::synth::gcd(Integer(-12), Integer(-8))), 4LL);
}

TEST(GcdTest, GcdDividesOperands) {
  for (int a = 1; a <= 30; ++a) {
    for (int b = 1; b <= 30; ++b) {
      Integer g = cudaq::synth::gcd(Integer(a), Integer(b));
      EXPECT_EQ(ll(Integer(a) % g), 0LL) << "a=" << a << " b=" << b;
      EXPECT_EQ(ll(Integer(b) % g), 0LL) << "a=" << a << " b=" << b;
    }
  }
}

// ============================================================
// is_probably_prime
// ============================================================

TEST(IsProbablyPrimeTest, SmallComposites) {
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(0)));
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(1)));
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(4)));
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(6)));
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(9)));
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(100)));
}

TEST(IsProbablyPrimeTest, SmallPrimes) {
  for (int p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}) {
    EXPECT_TRUE(cudaq::synth::is_probably_prime(Integer(p))) << "p = " << p;
  }
}

TEST(IsProbablyPrimeTest, NegativeUsesAbsoluteValue) {
  EXPECT_TRUE(cudaq::synth::is_probably_prime(Integer(-7)));
  EXPECT_TRUE(cudaq::synth::is_probably_prime(Integer(-13)));
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(-4)));
}

TEST(IsProbablyPrimeTest, LargerPrimes) {
  EXPECT_TRUE(cudaq::synth::is_probably_prime(Integer(104729)));
  EXPECT_TRUE(cudaq::synth::is_probably_prime(Integer(999983)));
}

TEST(IsProbablyPrimeTest, LargerComposites) {
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(104728)));
  EXPECT_FALSE(cudaq::synth::is_probably_prime(Integer(999981)));
}

// ============================================================
// is_odd
// ============================================================

TEST(IsOddTest, Odd) {
  EXPECT_TRUE(Integer(1).is_odd());
  EXPECT_TRUE(Integer(3).is_odd());
  EXPECT_TRUE(Integer(-5).is_odd());
}

TEST(IsOddTest, Even) {
  EXPECT_FALSE(Integer(0).is_odd());
  EXPECT_FALSE(Integer(2).is_odd());
  EXPECT_FALSE(Integer(-4).is_odd());
}

// ============================================================
// Compound assignment operators with long long
// ============================================================

TEST(CompoundAssignLLTest, AddPositive) {
  Integer x(10);
  x += 5LL;
  EXPECT_EQ(ll(x), 15LL);
}

TEST(CompoundAssignLLTest, AddNegative) {
  // Negative `rhs` routes through mpz_sub_ui
  Integer x(10);
  x += -3LL;
  EXPECT_EQ(ll(x), 7LL);
}

TEST(CompoundAssignLLTest, SubPositive) {
  Integer x(10);
  x -= 4LL;
  EXPECT_EQ(ll(x), 6LL);
}

TEST(CompoundAssignLLTest, SubNegative) {
  // Negative `rhs` routes through mpz_add_ui
  Integer x(10);
  x -= -3LL;
  EXPECT_EQ(ll(x), 13LL);
}

TEST(CompoundAssignLLTest, MulPositive) {
  Integer x(6);
  x *= 7LL;
  EXPECT_EQ(ll(x), 42LL);
}

TEST(CompoundAssignLLTest, MulNegative) {
  Integer x(6);
  x *= -3LL;
  EXPECT_EQ(ll(x), -18LL);
}

TEST(CompoundAssignLLTest, DivPositive) {
  Integer x(20);
  x /= 4LL;
  EXPECT_EQ(ll(x), 5LL);
}

TEST(CompoundAssignLLTest, DivNegative) {
  // Division by negative `rhs`: result should be negated
  Integer x(20);
  x /= -4LL;
  EXPECT_EQ(ll(x), -5LL);
}

TEST(CompoundAssignLLTest, ModPositive) {
  Integer x(17);
  x %= 5LL;
  EXPECT_EQ(ll(x), 2LL);
}

TEST(CompoundAssignLLTest, ModSignFollowsDividend) {
  // C++ truncation semantics: sign of remainder == sign of dividend
  Integer x(-17);
  x %= 5LL;
  EXPECT_EQ(ll(x), -2LL);
}

// ============================================================
// Bitwise shifts
// ============================================================

TEST(ShiftTest, LeftShift) {
  EXPECT_EQ(ll(Integer(1) << 3), 8LL);
  EXPECT_EQ(ll(Integer(3) << 4), 48LL);
}

TEST(ShiftTest, RightShiftPositive) {
  EXPECT_EQ(ll(Integer(16) >> 2), 4LL);
  EXPECT_EQ(ll(Integer(7) >> 1), 3LL); // truncates toward zero
}

TEST(ShiftTest, RightShiftNegative) {
  // mpz_tdiv_q_2exp truncates toward zero: -7 >> 1 = -3 (not -4)
  EXPECT_EQ(ll(Integer(-7) >> 1), -3LL);
  EXPECT_EQ(ll(Integer(-8) >> 1), -4LL);
}

TEST(ShiftTest, LeftThenRight) {
  // For positive n: n << k >> k == n
  for (int n : {1, 5, 13, 100}) {
    for (int k : {1, 3, 5}) {
      Integer shifted = Integer(n) << k;
      Integer back = shifted >> k;
      EXPECT_EQ(ll(back), static_cast<long long>(n)) << "n=" << n << " k=" << k;
    }
  }
}

} // namespace
