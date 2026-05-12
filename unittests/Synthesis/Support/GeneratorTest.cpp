/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include <type_traits>

#include "Support/Generator.h"

namespace {

using cudaq::synth::first_of;
using cudaq::synth::generator;
using cudaq::synth::to_vector;

// ============================================================
// Type properties
// ============================================================

TEST(GeneratorTypeTest, MoveOnly) {
  static_assert(!std::is_copy_constructible_v<generator<int>>);
  static_assert(!std::is_copy_assignable_v<generator<int>>);
  static_assert(std::is_move_constructible_v<generator<int>>);
  static_assert(std::is_move_assignable_v<generator<int>>);
}

TEST(GeneratorTypeTest, EmptyGeneratorIsValid) {
  auto empty = []() -> generator<int> { co_return; };
  auto results = to_vector(empty());
  EXPECT_TRUE(results.empty());
}

TEST(GeneratorTypeTest, ToVectorCollectsAll) {
  auto gen = []() -> generator<int> {
    for (int i = 0; i < 5; ++i)
      co_yield i;
  };
  auto v = to_vector(gen());
  ASSERT_EQ(v.size(), 5u);
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(v[i], i);
}

TEST(GeneratorTypeTest, FirstOfReturnsFirst) {
  auto gen = []() -> generator<int> {
    co_yield 42;
    co_yield 99;
  };
  auto f = first_of(gen());
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(*f, 42);
}

TEST(GeneratorTypeTest, FirstOfEmptyReturnsNullopt) {
  auto gen = []() -> generator<int> { co_return; };
  auto f = first_of(gen());
  EXPECT_FALSE(f.has_value());
}

} // namespace
