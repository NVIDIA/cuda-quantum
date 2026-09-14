/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/Synthesis/Math/Unitary.h"

namespace {

using namespace cudaq::synth;

// The two in-place rungs of the KMM peel loop are checked against the generic
// gate operators. Each is defined where the implied sqrt(2) divisions are
// exact, so every (unitary, m) pair is screened for that precondition first.

/// Divisibility by delta = 1 + omega, i.e. by sqrt(2): the precondition of
/// `mul_by_inv_sqrt2` (Remark D.2).
bool divisible_by_delta(const ZOmega &u) {
  return !(u.b() + u.d()).is_odd() && !(u.c() + u.a()).is_odd();
}

bool divisible_by_two(const ZOmega &u) {
  return !u.a().is_odd() && !u.b().is_odd() && !u.c().is_odd() &&
         !u.d().is_odd();
}

/// Clifford+T words of growing length, reaching a range of denominator
/// exponents.
std::vector<DOmegaUnitary> sample_unitaries() {
  static const Gate alphabet[] = {Gate::H, Gate::T, Gate::S,
                                  Gate::X, Gate::W, Gate::H};
  std::vector<DOmegaUnitary> unitaries;
  uint64_t state = 0x243f6a8885a308d3ull;
  for (int length = 1; length <= 40; ++length) {
    Circuit word;
    for (int i = 0; i < length; ++i) {
      state = state * 6364136223846793005ull + 1442695040888963407ull;
      word.push_back(alphabet[(state >> 33) % std::size(alphabet)]);
    }
    unitaries.push_back(DOmegaUnitary::from_gates(word));
  }
  return unitaries;
}

TEST(DOmegaUnitaryRungTest, ReducingRungMatchesOutOfPlace) {
  ZOmega scratch;
  int compared = 0;
  for (const DOmegaUnitary &u : sample_unitaries()) {
    if (u.k() < 1)
      continue;
    for (int32_t m = 0; m < 8; ++m) {
      DOmegaUnitary turned = u.mul_by_T_power_from_left(m);
      const ZOmega &z = turned.z().u();
      const ZOmega &w = turned.w().u();
      if (!divisible_by_two(z + w) || !divisible_by_two(z - w))
        continue;

      DOmegaUnitary expected =
          with_denom_exp(u.mul_by_H_and_T_power_from_left(m), u.k() - 1);
      DOmegaUnitary actual = u;
      actual.reduce_by_H_and_T_power_from_left(m, scratch);
      EXPECT_TRUE(actual == expected) << "m = " << m;
      EXPECT_EQ(actual.k(), u.k() - 1) << "m = " << m;
      ++compared;
    }
  }
  EXPECT_GT(compared, 0) << "no input satisfied the rung's precondition";
}

TEST(DOmegaUnitaryRungTest, BonusRungMatchesOutOfPlace) {
  ZOmega scratch;
  int compared = 0;
  for (const DOmegaUnitary &u : sample_unitaries()) {
    for (int32_t m = 0; m < 8; ++m) {
      DOmegaUnitary turned = u.mul_by_T_power_from_left(m);
      const ZOmega &z = turned.z().u();
      const ZOmega &w = turned.w().u();
      if (!divisible_by_delta(z + w) || !divisible_by_delta(z - w))
        continue;

      DOmegaUnitary expected = u.mul_by_H_and_T_power_from_left(m);
      DOmegaUnitary actual = u;
      actual.mul_by_H_and_T_power_from_left_in_place(m, scratch);
      EXPECT_TRUE(actual == expected) << "m = " << m;
      EXPECT_EQ(actual.k(), u.k()) << "m = " << m;
      ++compared;
    }
  }
  EXPECT_GT(compared, 0) << "no input satisfied the rung's precondition";
}

/// The peel loop reuses one scratch value across every rung, so each rung must
/// treat `scratch` as write-only. Every pair is screened for the rung's own
/// divisibility precondition first, exactly as the two tests above do.
TEST(DOmegaUnitaryRungTest, RungsIgnoreIncomingScratchContents) {
  int compared = 0;
  for (const DOmegaUnitary &u : sample_unitaries()) {
    for (int32_t m = 0; m < 8; ++m) {
      DOmegaUnitary turned = u.mul_by_T_power_from_left(m);
      const ZOmega &z = turned.z().u();
      const ZOmega &w = turned.w().u();

      ZOmega clean;
      ZOmega dirty(Integer(7), Integer(-11), Integer(13), Integer(-17));

      if (divisible_by_delta(z + w) && divisible_by_delta(z - w)) {
        DOmegaUnitary from_clean = u;
        DOmegaUnitary from_dirty = u;
        from_clean.mul_by_H_and_T_power_from_left_in_place(m, clean);
        from_dirty.mul_by_H_and_T_power_from_left_in_place(m, dirty);
        EXPECT_TRUE(from_clean == from_dirty) << "bonus rung, m = " << m;
        ++compared;
      }

      clean = ZOmega();
      dirty = ZOmega(Integer(7), Integer(-11), Integer(13), Integer(-17));

      if (u.k() >= 1 && divisible_by_two(z + w) && divisible_by_two(z - w)) {
        DOmegaUnitary from_clean = u;
        DOmegaUnitary from_dirty = u;
        from_clean.reduce_by_H_and_T_power_from_left(m, clean);
        from_dirty.reduce_by_H_and_T_power_from_left(m, dirty);
        EXPECT_TRUE(from_clean == from_dirty) << "reducing rung, m = " << m;
        ++compared;
      }
    }
  }
  EXPECT_GT(compared, 0) << "no input satisfied either rung's precondition";
}

} // namespace
