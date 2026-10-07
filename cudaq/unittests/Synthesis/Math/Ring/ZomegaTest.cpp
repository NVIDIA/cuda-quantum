/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/Synthesis/Math/Ring/Zomega.h"

namespace {

using namespace cudaq::synth;

// sqrt(2) = omega - omega^3 in the (a, b, c, d) coefficient basis.
static const ZOmega kSqrt2{Integer(-1), Integer(0), Integer(1), Integer(0)};

// Deterministic coefficient source, mixed in sign and magnitude.
class Coefficients {
  uint64_t state = 0x9e3779b97f4a7c15ull;

public:
  int64_t next() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<int64_t>(state >> 33) - (1 << 30);
  }

  ZOmega next_zomega() {
    Integer a(next()), b(next()), c(next()), d(next());
    return ZOmega(a, b, c, d);
  }
};

TEST(ZOmegaInPlaceTest, MulByOmegaPowerMatchesOutOfPlace) {
  Coefficients gen;
  for (int32_t n = 0; n < 16; ++n) {
    ZOmega x = gen.next_zomega();
    ZOmega expected = mul_by_omega_power(x, n);
    mul_by_omega_power_in_place(x, n);
    EXPECT_TRUE(x == expected) << "omega^" << n;
  }
}

TEST(ZOmegaInPlaceTest, MulBySqrt2MatchesGeneralMultiply) {
  Coefficients gen;
  ZOmega scratch;
  for (int i = 0; i < 32; ++i) {
    ZOmega x = gen.next_zomega();
    ZOmega expected = x * kSqrt2;
    mul_by_sqrt2_in_place(x, scratch);
    EXPECT_TRUE(x == expected) << "iteration " << i;
  }
}

TEST(ZOmegaInPlaceTest, MulBySqrt2TwiceDoublesEachCoefficient) {
  Coefficients gen;
  ZOmega scratch;
  ZOmega x = gen.next_zomega();
  ZOmega doubled = x + x;
  mul_by_sqrt2_in_place(x, scratch);
  mul_by_sqrt2_in_place(x, scratch);
  EXPECT_TRUE(x == doubled);
}

TEST(ZOmegaInPlaceTest, HalveSumAndDifferenceInvertsSumAndDifference) {
  Coefficients gen;
  ZOmega scratch;
  for (int i = 0; i < 32; ++i) {
    ZOmega p = gen.next_zomega();
    ZOmega q = gen.next_zomega();
    ZOmega z = p + q;
    ZOmega w = p - q;
    halve_sum_and_difference(z, w, scratch);
    EXPECT_TRUE(z == p) << "iteration " << i;
    EXPECT_TRUE(w == q) << "iteration " << i;
  }
}

TEST(ZOmegaInPlaceTest, HalveSumAndDifferenceRoundsTowardZero) {
  ZOmega z(-3, -3, -3, -3);
  ZOmega w(-1, -1, -1, -1);
  ZOmega scratch;
  halve_sum_and_difference(z, w, scratch);
  EXPECT_TRUE(z == ZOmega(-2, -2, -2, -2));
  EXPECT_TRUE(w == ZOmega(-1, -1, -1, -1));
}

} // namespace
