/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <bit>
#include <cmath>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace cudaq::detail {
std::vector<double> convertAngles(std::span<double> alphas);
}

TEST(StatePreparationTest, AnglesMatchDirectTransform) {
  for (std::size_t size = 1; size <= 1024; size *= 2) {
    std::vector<double> alphas(size);
    for (std::size_t j = 0; j < size; ++j)
      alphas[j] = 3.0 * std::sin(0.73 * j) - std::cos(1.17 * j);
    const auto original = alphas;
    const auto thetas = cudaq::detail::convertAngles(alphas);
    ASSERT_EQ(thetas.size(), size);
    EXPECT_EQ(alphas, original);
    for (std::size_t i = 0; i < size; ++i) {
      long double expected = 0;
      for (std::size_t j = 0; j < size; ++j)
        expected +=
            std::popcount(j & ((i >> 1) ^ i)) % 2 ? -alphas[j] : alphas[j];
      EXPECT_NEAR(thetas[i], static_cast<double>(expected / size), 1e-13)
          << "size=" << size << ", row=" << i;
    }
  }
}

TEST(StatePreparationTest, ConstantAndImpulseAngles) {
  std::vector<double> empty;
  EXPECT_TRUE(cudaq::detail::convertAngles(empty).empty());
  for (std::size_t size = 1; size <= 256; size *= 2) {
    std::vector<double> constant(size, 0.25);
    auto actual = cudaq::detail::convertAngles(constant);
    EXPECT_DOUBLE_EQ(actual[0], 0.25);
    for (std::size_t i = 1; i < size; ++i)
      EXPECT_DOUBLE_EQ(actual[i], 0.0);

    for (std::size_t column = 0; column < size; ++column) {
      std::vector<double> impulse(size, 0.0);
      impulse[column] = 1.0;
      actual = cudaq::detail::convertAngles(impulse);
      for (std::size_t i = 0; i < size; ++i) {
        const double sign =
            std::popcount(column & ((i >> 1) ^ i)) % 2 ? -1.0 : 1.0;
        EXPECT_DOUBLE_EQ(actual[i], sign / size);
      }
    }
  }
}
