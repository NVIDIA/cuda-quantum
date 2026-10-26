/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Circuit/Clifford.h"
#include <gtest/gtest.h>
#include <sstream>

namespace {

using cudaq::synth::Clifford;
using cudaq::synth::CLIFFORD_I;

static std::string describe(const Clifford &c) {
  std::ostringstream os;
  os << c;
  return os.str();
}

// ============================================================
// Group inverse
// ============================================================

// Exhaustive check over the whole 192-element group (3 * 2 * 4 * 8): inv()
// must be a genuine two-sided inverse.
//
// Regression: CINV_TABLE's row for (a, b, c) = (1, 1, 0) -- i.e. E * X --
// held {2, 1, 1, 2} instead of {2, 1, 2, 2}, so C * C.inv() came out as S
// rather than the identity. The synthesis path only calls inv() on the fixed
// H and SH values, so nothing downstream exercised the defective row.
TEST(CliffordInverseTest, InverseIsTwoSidedForAllElements) {
  for (int32_t a = 0; a < 3; ++a)
    for (int32_t b = 0; b < 2; ++b)
      for (int32_t c = 0; c < 4; ++c)
        for (int32_t d = 0; d < 8; ++d) {
          Clifford elem(a, b, c, d);
          Clifford inverse = elem.inv();

          EXPECT_EQ(elem * inverse, CLIFFORD_I)
              << describe(elem) << " times its inverse " << describe(inverse)
              << " is " << describe(elem * inverse);
          EXPECT_EQ(inverse * elem, CLIFFORD_I)
              << "inverse " << describe(inverse) << " times " << describe(elem)
              << " is " << describe(inverse * elem);
        }
}

// The inverse is unique in a group, so inverting twice is the identity map.
TEST(CliffordInverseTest, InverseIsAnInvolution) {
  for (int32_t a = 0; a < 3; ++a)
    for (int32_t b = 0; b < 2; ++b)
      for (int32_t c = 0; c < 4; ++c)
        for (int32_t d = 0; d < 8; ++d) {
          Clifford elem(a, b, c, d);
          EXPECT_EQ(elem.inv().inv(), elem) << describe(elem);
        }
}

// Spot-check the row that was wrong, so a regression names itself.
TEST(CliffordInverseTest, InverseOfETimesX) {
  Clifford e_x(1, 1, 0, 0);
  Clifford inverse = e_x.inv();

  EXPECT_EQ(inverse, Clifford(2, 1, 2, 2)) << describe(inverse);
  EXPECT_EQ(e_x * inverse, CLIFFORD_I);
}

} // namespace
