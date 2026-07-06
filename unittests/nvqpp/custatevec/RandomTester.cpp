/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecRandom.h"

#include <gtest/gtest.h>

#include <algorithm>

TEST(CuStateVecRandomTester, GeneratesUnsortedBoundedValues) {
  cudaq::cusv::CuStateVecRandom generator;
  generator.setSeed(42);
  for (const std::size_t count : {std::size_t{1}, std::size_t{1024},
                                  std::size_t{100000}, std::size_t{1000000}}) {
    const auto values = generator.generate(count);
    ASSERT_EQ(values.size(), count);
    if (count > 1)
      EXPECT_FALSE(std::is_sorted(values.begin(), values.end()));
    EXPECT_TRUE(std::all_of(values.begin(), values.end(), [](double value) {
      return value >= 0.0 && value < 1.0;
    }));
  }
}

TEST(CuStateVecRandomTester, DefaultGeneratorsAreIndependent) {
  cudaq::cusv::CuStateVecRandom firstGenerator;
  cudaq::cusv::CuStateVecRandom secondGenerator;
  EXPECT_NE(firstGenerator.generate(64), secondGenerator.generate(64));
}

TEST(CuStateVecRandomTester, SeedIsReproducible) {
  cudaq::cusv::CuStateVecRandom generator;
  generator.setSeed(13);
  const auto first = generator.generate(64);
  generator.setSeed(13);
  EXPECT_EQ(generator.generate(64), first);
}
