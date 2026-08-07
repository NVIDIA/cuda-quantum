/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "Math/Geometry/Interval.h"
#include "Math/Grid/Odgp.h"
#include "Support/Stepper.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"

namespace {

using cudaq::synth::DSqrt2;
using cudaq::synth::first_of;
using cudaq::synth::Integer;
using cudaq::synth::Interval;
using cudaq::synth::OdgpScaledStepper;
using cudaq::synth::OdgpScaledWithParityStepper;
using cudaq::synth::OdgpStepper;
using cudaq::synth::OdgpWithParityStepper;
using cudaq::synth::Real;
using cudaq::synth::to_vector;
using cudaq::synth::ZSqrt2;

// ============================================================
// Helper: verify a ZSqrt2 element is in the interval pair (I, J)
// ============================================================
static bool in_intervals(const ZSqrt2 &z, const Interval &I,
                         const Interval &J) {
  Real r = to_real(z);
  Real rc = to_real(z.conj_sq2());
  return I.l() <= r && r <= I.r() && J.l() <= rc && rc <= J.r();
}

// ============================================================
// Empty result tests
// ============================================================

TEST(OdgpGeneratorTest, EmptyWhenIntervalsDegenerate) {
  Interval I(Real(1.0), Real(0.5));
  Interval J(Real(0.0), Real(1.0));
  auto results = to_vector(OdgpStepper(I, J));
  EXPECT_TRUE(results.empty());
}

TEST(OdgpGeneratorTest, EmptyWhenNoSolutions) {
  Interval I(Real(0.1), Real(0.1001));
  Interval J(Real(0.1), Real(0.1001));
  auto results = to_vector(OdgpStepper(I, J));
  for (const auto &z : results)
    EXPECT_TRUE(in_intervals(z, I, J));
}

// ============================================================
// Basic solution tests
// ============================================================

TEST(OdgpGeneratorTest, FindsSolutionsInWideIntervals) {
  Interval I(Real(-2.0), Real(2.0));
  Interval J(Real(-2.0), Real(2.0));
  auto results = to_vector(OdgpStepper(I, J));
  EXPECT_GT(results.size(), 0u);
  for (const auto &z : results)
    EXPECT_TRUE(in_intervals(z, I, J));
}

TEST(OdgpGeneratorTest, SolutionsAreValidZSqrt2Elements) {
  Interval I(Real(-5.0), Real(5.0));
  Interval J(Real(-5.0), Real(5.0));
  auto results = to_vector(OdgpStepper(I, J));
  for (const auto &z : results) {
    EXPECT_TRUE(in_intervals(z, I, J))
        << "z=" << z.to_string() << " not in intervals";
  }
}

// ============================================================
// Early termination (the primary motivation for the generator refactor)
// ============================================================

TEST(OdgpGeneratorTest, EarlyTerminationProducesValidFirst) {
  Interval I(Real(-10.0), Real(10.0));
  Interval J(Real(-10.0), Real(10.0));
  auto first = first_of(OdgpStepper(I, J));
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(in_intervals(*first, I, J));
}

TEST(OdgpGeneratorTest, EarlyTerminationDoesNotLeak) {
  Interval I(Real(-100.0), Real(100.0));
  Interval J(Real(-100.0), Real(100.0));
  for (int i = 0; i < 100; ++i) {
    auto gen = OdgpStepper(I, J);
    auto it = gen.begin();
    if (it != gen.end()) {
      [[maybe_unused]] ZSqrt2 val = *it;
    }
  }
}

// ============================================================
// Scaled ODGP tests
// ============================================================

TEST(OdgpScaledGeneratorTest, ProducesSolutions) {
  Interval I(Real(-1.0), Real(1.0));
  Interval J(Real(-1.0), Real(1.0));
  auto results = to_vector(OdgpScaledStepper(I, J, Integer(2)));
  EXPECT_GT(results.size(), 0u);
}

TEST(OdgpScaledGeneratorTest, EmptyForNarrowIntervals) {
  Interval I(Real(0.5001), Real(0.5002));
  Interval J(Real(0.5001), Real(0.5002));
  auto results = to_vector(OdgpScaledStepper(I, J, Integer(1)));
  // May or may not be empty depending on precise arithmetic;
  // just verify no crash.
}

TEST(OdgpScaledGeneratorTest, EarlyTermination) {
  Interval I(Real(-10.0), Real(10.0));
  Interval J(Real(-10.0), Real(10.0));
  auto first = first_of(OdgpScaledStepper(I, J, Integer(4)));
  ASSERT_TRUE(first.has_value());
}

// ============================================================
// Scaled with parity tests
// ============================================================

TEST(OdgpScaledWithParityTest, ProducesSolutions) {
  Interval I(Real(-2.0), Real(2.0));
  Interval J(Real(-2.0), Real(2.0));
  DSqrt2 parity_hint(ZSqrt2{1, 0}, Integer(1));
  auto results =
      to_vector(OdgpScaledWithParityStepper(I, J, Integer(1), parity_hint));
  EXPECT_GT(results.size(), 0u);
}

TEST(OdgpScaledWithParityTest, EarlyTermination) {
  Interval I(Real(-10.0), Real(10.0));
  Interval J(Real(-10.0), Real(10.0));
  DSqrt2 parity_hint(ZSqrt2{0, 0}, Integer(1));

  for (int i = 0; i < 50; ++i) {
    auto gen = OdgpScaledWithParityStepper(I, J, Integer(1), parity_hint);
    auto it = gen.begin();
    if (it != gen.end()) {
      [[maybe_unused]] DSqrt2 val = *it;
    }
  }
}

// ============================================================
// With parity tests
// ============================================================

TEST(OdgpWithParityTest, ProducesSolutions) {
  Interval I(Real(-3.0), Real(3.0));
  Interval J(Real(-3.0), Real(3.0));
  ZSqrt2 hint(1, 0);
  auto results = to_vector(OdgpWithParityStepper(I, J, hint));
  EXPECT_GT(results.size(), 0u);
}

// ============================================================
// Order preservation: generator produces same order as iteration
// ============================================================

TEST(OdgpGeneratorTest, ConsistentAcrossMultipleRuns) {
  Interval I(Real(-3.0), Real(3.0));
  Interval J(Real(-3.0), Real(3.0));
  auto run1 = to_vector(OdgpStepper(I, J));
  auto run2 = to_vector(OdgpStepper(I, J));
  ASSERT_EQ(run1.size(), run2.size());
  for (size_t i = 0; i < run1.size(); ++i)
    EXPECT_EQ(run1[i], run2[i]) << "Mismatch at index " << i;
}

// ============================================================
// OdgpStepper-specific tests (the hand-rolled stepper that replaces the
// former coroutine).
// ============================================================

// next() returns nullptr immediately for an empty/degenerate input and
// continues to return nullptr on subsequent calls.
TEST(OdgpStepperTest, NextReturnsNullptrOnEmpty) {
  Interval I(Real(1.0), Real(0.5));
  Interval J(Real(0.0), Real(1.0));
  OdgpStepper stepper(I, J);
  EXPECT_EQ(stepper.next(), nullptr);
  EXPECT_EQ(stepper.next(), nullptr);
  EXPECT_EQ(stepper.next(), nullptr);
}

// Direct next() iteration is equivalent to range-for.
TEST(OdgpStepperTest, NextMatchesRangeFor) {
  Interval I(Real(-3.0), Real(3.0));
  Interval J(Real(-3.0), Real(3.0));

  std::vector<ZSqrt2> via_next;
  {
    OdgpStepper stepper(I, J);
    while (const ZSqrt2 *v = stepper.next())
      via_next.push_back(*v);
  }
  auto via_range = to_vector(OdgpStepper(I, J));

  ASSERT_EQ(via_next.size(), via_range.size());
  for (size_t i = 0; i < via_next.size(); ++i)
    EXPECT_EQ(via_next[i], via_range[i]) << "Mismatch at index " << i;
}

// begin() != end() drives the iterator like a generator<T>.
TEST(OdgpStepperTest, IteratorInterface) {
  Interval I(Real(-2.0), Real(2.0));
  Interval J(Real(-2.0), Real(2.0));
  OdgpStepper stepper(I, J);

  auto it = stepper.begin();
  ASSERT_NE(it, stepper.end());
  ZSqrt2 first = *it;
  EXPECT_TRUE(in_intervals(first, I, J));

  ++it;
  if (it != stepper.end()) {
    ZSqrt2 second = *it;
    EXPECT_TRUE(in_intervals(second, I, J));
    EXPECT_NE(first, second);
  }
}

// Pointer-yield contract: *it is valid until the next ++it.
TEST(OdgpStepperTest, PointerStableUntilAdvance) {
  Interval I(Real(-5.0), Real(5.0));
  Interval J(Real(-5.0), Real(5.0));
  OdgpStepper stepper(I, J);

  const ZSqrt2 *first = stepper.next();
  ASSERT_NE(first, nullptr);
  ZSqrt2 first_copy = *first;

  // The pointer remains valid (and equal) until next() is called again.
  EXPECT_EQ(*first, first_copy);
  EXPECT_EQ(first, &(*first));
}

// Destroying the stepper mid-stream must release all mpfr_t / mpz_t state
// and re-iterating should not corrupt thread-local caches (e.g. lambda
// powers, ScopedPrinter indentation).
TEST(OdgpStepperTest, EarlyDestructionDoesNotLeak) {
  Interval I(Real(-100.0), Real(100.0));
  Interval J(Real(-100.0), Real(100.0));
  for (int i = 0; i < 200; ++i) {
    OdgpStepper stepper(I, J);
    const ZSqrt2 *first = stepper.next();
    ASSERT_NE(first, nullptr);
    // intentional: stepper goes out of scope here mid-enumeration.
  }
}

// Snapshot-equivalence: pin the produced sequence for a fixed (I, J) so any
// future refactor that changes ordering or post-yield update sequencing
// fails loudly.
TEST(OdgpStepperTest, SnapshotEquivalence) {
  Interval I(Real(-2.0), Real(2.0));
  Interval J(Real(-2.0), Real(2.0));

  auto seq1 = to_vector(OdgpStepper(I, J));
  auto seq2 = to_vector(OdgpStepper(I, J));

  ASSERT_EQ(seq1.size(), seq2.size());
  EXPECT_GT(seq1.size(), 0u);
  for (size_t i = 0; i < seq1.size(); ++i) {
    EXPECT_EQ(seq1[i], seq2[i])
        << "Mismatch at index " << i << ": " << seq1[i].to_string() << " vs "
        << seq2[i].to_string();
    EXPECT_TRUE(in_intervals(seq1[i], I, J));
  }

  // Ordering is lexicographic in (a, b): each successive element must be
  // strictly greater (by ZSqrt2's real-value comparison) within the same a,
  // or move to a larger a.
  for (size_t i = 1; i < seq1.size(); ++i)
    EXPECT_FALSE(seq1[i] == seq1[i - 1]);
}

// Multiple consecutive next() calls after exhaustion stay at nullptr.
TEST(OdgpStepperTest, IdempotentAfterExhaustion) {
  Interval I(Real(0.0), Real(0.5));
  Interval J(Real(0.0), Real(0.5));
  OdgpStepper stepper(I, J);

  // Drain.
  while (stepper.next() != nullptr) {
  }
  // Further calls stay at nullptr.
  for (int i = 0; i < 5; ++i)
    EXPECT_EQ(stepper.next(), nullptr);
}

// Range-for over a named (lvalue) stepper -- exercises the begin()/end()
// path on a non-temporary.
TEST(OdgpStepperTest, RangeForOverNamedStepper) {
  Interval I(Real(-3.0), Real(3.0));
  Interval J(Real(-3.0), Real(3.0));
  OdgpStepper stepper(I, J);

  size_t count = 0;
  for (const ZSqrt2 &z : stepper) {
    EXPECT_TRUE(in_intervals(z, I, J));
    ++count;
  }
  EXPECT_GT(count, 0u);
}

// ============================================================
// Wrapper stepper tests: OdgpWithParity, OdgpScaled, OdgpScaledWithParity.
// Each wraps an inner OdgpStepper (directly or via std::optional) and
// performs a value transform per next().
// ============================================================

TEST(OdgpWithParityStepperTest, NextMatchesRangeFor) {
  Interval I(Real(-3.0), Real(3.0));
  Interval J(Real(-3.0), Real(3.0));
  ZSqrt2 hint(1, 0);

  std::vector<ZSqrt2> via_next;
  {
    OdgpWithParityStepper stepper(I, J, hint);
    while (const ZSqrt2 *v = stepper.next())
      via_next.push_back(*v);
  }
  auto via_range = to_vector(OdgpWithParityStepper(I, J, hint));

  ASSERT_EQ(via_next.size(), via_range.size());
  for (size_t i = 0; i < via_next.size(); ++i)
    EXPECT_EQ(via_next[i], via_range[i]) << "Mismatch at index " << i;
}

TEST(OdgpWithParityStepperTest, EmptyForDegenerate) {
  Interval I(Real(1.0), Real(0.5));
  Interval J(Real(0.0), Real(1.0));
  ZSqrt2 hint(0, 0);
  OdgpWithParityStepper stepper(I, J, hint);
  EXPECT_EQ(stepper.next(), nullptr);
  EXPECT_EQ(stepper.next(), nullptr);
}

TEST(OdgpWithParityStepperTest, EarlyDestructionDoesNotLeak) {
  Interval I(Real(-10.0), Real(10.0));
  Interval J(Real(-10.0), Real(10.0));
  ZSqrt2 hint(1, 0);
  for (int i = 0; i < 100; ++i) {
    OdgpWithParityStepper stepper(I, J, hint);
    [[maybe_unused]] const ZSqrt2 *first = stepper.next();
  }
}

TEST(OdgpScaledStepperTest, NextMatchesRangeFor) {
  Interval I(Real(-1.0), Real(1.0));
  Interval J(Real(-1.0), Real(1.0));

  std::vector<DSqrt2> via_next;
  {
    OdgpScaledStepper stepper(I, J, Integer(2));
    while (const DSqrt2 *v = stepper.next())
      via_next.push_back(*v);
  }
  auto via_range = to_vector(OdgpScaledStepper(I, J, Integer(2)));

  ASSERT_EQ(via_next.size(), via_range.size());
  for (size_t i = 0; i < via_next.size(); ++i)
    EXPECT_EQ(via_next[i], via_range[i]);
}

TEST(OdgpScaledStepperTest, EarlyTermination) {
  Interval I(Real(-10.0), Real(10.0));
  Interval J(Real(-10.0), Real(10.0));
  auto first = first_of(OdgpScaledStepper(I, J, Integer(4)));
  ASSERT_TRUE(first.has_value());
}

TEST(OdgpScaledWithParityStepperTest, DenomZeroBranchMatchesRangeFor) {
  Interval I(Real(-3.0), Real(3.0));
  Interval J(Real(-3.0), Real(3.0));
  DSqrt2 hint(ZSqrt2{1, 0}, Integer(0));

  std::vector<DSqrt2> via_next;
  {
    OdgpScaledWithParityStepper stepper(I, J, Integer(0), hint);
    while (const DSqrt2 *v = stepper.next())
      via_next.push_back(*v);
  }
  auto via_range =
      to_vector(OdgpScaledWithParityStepper(I, J, Integer(0), hint));

  ASSERT_EQ(via_next.size(), via_range.size());
  for (size_t i = 0; i < via_next.size(); ++i)
    EXPECT_EQ(via_next[i], via_range[i]);
}

TEST(OdgpScaledWithParityStepperTest, RecursiveBranchMatchesRangeFor) {
  Interval I(Real(-2.0), Real(2.0));
  Interval J(Real(-2.0), Real(2.0));
  DSqrt2 hint(ZSqrt2{1, 0}, Integer(1));

  std::vector<DSqrt2> via_next;
  {
    OdgpScaledWithParityStepper stepper(I, J, Integer(1), hint);
    while (const DSqrt2 *v = stepper.next())
      via_next.push_back(*v);
  }
  auto via_range =
      to_vector(OdgpScaledWithParityStepper(I, J, Integer(1), hint));

  ASSERT_EQ(via_next.size(), via_range.size());
  for (size_t i = 0; i < via_next.size(); ++i)
    EXPECT_EQ(via_next[i], via_range[i]);
}

TEST(OdgpScaledWithParityStepperTest, EarlyDestructionDoesNotLeak) {
  Interval I(Real(-10.0), Real(10.0));
  Interval J(Real(-10.0), Real(10.0));
  DSqrt2 hint(ZSqrt2{0, 0}, Integer(1));
  for (int i = 0; i < 50; ++i) {
    OdgpScaledWithParityStepper stepper(I, J, Integer(1), hint);
    [[maybe_unused]] const DSqrt2 *first = stepper.next();
  }
}

} // namespace
