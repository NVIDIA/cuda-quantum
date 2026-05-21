/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include <type_traits>

#include "Support/Stepper.h"

namespace {

using cudaq::synth::first_of;
using cudaq::synth::StepperBase;
using cudaq::synth::to_vector;

// ============================================================
// Minimal concrete stepper used to exercise StepperBase / to_vector / first_of
// in isolation from the synth domain types.
// ============================================================
class Iota : public StepperBase<Iota, int> {
public:
  explicit Iota(int n) : n_(n) {}

  Iota(const Iota &) = delete;
  Iota &operator=(const Iota &) = delete;
  Iota(Iota &&) = delete;
  Iota &operator=(Iota &&) = delete;

  const int *next() {
    if (i_ >= n_)
      return nullptr;
    last_ = i_++;
    return &last_;
  }

private:
  int n_;
  int i_ = 0;
  int last_ = 0;
};

// ============================================================
// Compile-time properties
// ============================================================

TEST(StepperBaseTest, NonCopyableNonMovable) {
  static_assert(!std::is_copy_constructible_v<Iota>);
  static_assert(!std::is_copy_assignable_v<Iota>);
  static_assert(!std::is_move_constructible_v<Iota>);
  static_assert(!std::is_move_assignable_v<Iota>);
}

TEST(StepperBaseTest, IteratorIsInputIterator) {
  using Iter = Iota::iterator;
  static_assert(std::is_same_v<std::iterator_traits<Iter>::iterator_category,
                               std::input_iterator_tag>);
  // LLVM's iterator_facade_base sets `value_type = T` directly from the
  // template arg; we pass `const int` so the iterator dereferences to
  // `const int &`.
  static_assert(
      std::is_same_v<std::iterator_traits<Iter>::value_type, const int>);
  static_assert(
      std::is_same_v<std::iterator_traits<Iter>::reference, const int &>);
}

// ============================================================
// Range / next() behaviour
// ============================================================

TEST(StepperBaseTest, EmptyStepperIsImmediatelyEnd) {
  Iota empty(0);
  EXPECT_EQ(empty.next(), nullptr);
  EXPECT_EQ(empty.next(), nullptr);

  Iota empty2(0);
  EXPECT_TRUE(to_vector(empty2).empty());
}

TEST(StepperBaseTest, ToVectorCollectsAll) {
  auto v = to_vector(Iota(5));
  ASSERT_EQ(v.size(), 5u);
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(v[i], i);
}

TEST(StepperBaseTest, FirstOfReturnsFirst) {
  auto f = first_of(Iota(3));
  ASSERT_TRUE(f.has_value());
  EXPECT_EQ(*f, 0);
}

TEST(StepperBaseTest, FirstOfEmptyReturnsNullopt) {
  auto f = first_of(Iota(0));
  EXPECT_FALSE(f.has_value());
}

TEST(StepperBaseTest, RangeForOverNamedLvalue) {
  Iota gen(4);
  int expected = 0;
  for (const int &x : gen) {
    EXPECT_EQ(x, expected++);
  }
  EXPECT_EQ(expected, 4);
}

TEST(StepperBaseTest, NextReturnsNullptrAfterExhaustion) {
  Iota gen(2);
  EXPECT_NE(gen.next(), nullptr);
  EXPECT_NE(gen.next(), nullptr);
  EXPECT_EQ(gen.next(), nullptr);
  EXPECT_EQ(gen.next(), nullptr);
}

TEST(StepperBaseTest, IteratorEqualsEndAtExhaustion) {
  Iota gen(1);
  auto it = gen.begin();
  ASSERT_NE(it, gen.end());
  EXPECT_EQ(*it, 0);
  ++it;
  EXPECT_EQ(it, gen.end());
}

} // namespace
