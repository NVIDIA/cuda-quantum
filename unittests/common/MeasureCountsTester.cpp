/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/SampleResult.h"

using namespace cudaq;

CUDAQ_TEST(sample_resultTester, checkConstruction) {
  ExecutionResult r{CountsDictionary{{"0", 400}, {"1", 600}}};
  cudaq::sample_result mc(r);
  EXPECT_EQ(2, mc.size());
}

CUDAQ_TEST(MeasureCountsTester, checkProbability) {
  ExecutionResult r{CountsDictionary{{"0", 400}, {"1", 600}}};
  cudaq::sample_result mc(r);
  EXPECT_NEAR(2. / 5., mc.probability("0"), 1e-9);
  EXPECT_NEAR(3. / 5., mc.probability("1"), 1e-9);
}

CUDAQ_TEST(MeasureCountsTester, checkCount) {
  ExecutionResult r{CountsDictionary{{"0", 400}, {"1", 600}}};
  cudaq::sample_result mc(r);
  EXPECT_EQ(400, mc.count("0"));
  EXPECT_EQ(600, mc.count("1"));
  EXPECT_EQ(0, mc.count("1111"));
}

CUDAQ_TEST(MeasureCountsTester, checkExpValZ) {
  ExecutionResult r{CountsDictionary{{"0", 400}, {"1", 600}}};
  cudaq::sample_result mc(r);
  EXPECT_NEAR(-1. / 5., mc.expectation(), 1e-9);
}

// TEST Sample Result / sample_result serialize / deserialize

CUDAQ_TEST(MeasureCountsTester, checkSampleResultSerialize) {
  ExecutionResult r{CountsDictionary{{"0", 400}, {"1", 600}}};
  auto data = r.serialize();
  ExecutionResult rr;
  rr.deserialize(data);
  EXPECT_TRUE(rr == r);

  cudaq::sample_result mc(rr);
  EXPECT_EQ(2, mc.size());
  EXPECT_EQ(400, mc.count("0"));
  EXPECT_EQ(600, mc.count("1"));
}

CUDAQ_TEST(MeasureCountsTester, checkMeasureCountsSerialize) {
  ExecutionResult r{CountsDictionary{{"0", 400}, {"1", 600}}};
  ExecutionResult rr{CountsDictionary{{"01", 400}, {"11", 600}}, "c0"};

  cudaq::sample_result mc(r), mm;
  mc.append(rr);
  auto data = mc.serialize();
  mm.deserialize(data);

  EXPECT_TRUE(mm == mc);
}

CUDAQ_TEST(MeasureResultTester, checkConstructors) {
  static_assert(!std::is_default_constructible_v<cudaq::measure_result>);
  static_assert(std::is_copy_constructible_v<cudaq::measure_result>);
  static_assert(std::is_move_constructible_v<cudaq::measure_result>);
  static_assert(!std::is_copy_assignable_v<cudaq::measure_result>);
  static_assert(!std::is_move_assignable_v<cudaq::measure_result>);

  cudaq::measure_result r1(int64_t(1));
  EXPECT_EQ(static_cast<int>(r1), 1);
  EXPECT_TRUE(static_cast<bool>(r1));

  cudaq::measure_result r2(int64_t(0), int64_t(42));
  EXPECT_EQ(static_cast<int>(r2), 0);
  EXPECT_FALSE(static_cast<bool>(r2));
  EXPECT_NEAR(static_cast<double>(r2), 0.0, 1e-9);

  cudaq::measure_result r3(r1);
  EXPECT_EQ(static_cast<int>(r3), 1);

  cudaq::measure_result r4(std::move(r1));
  EXPECT_EQ(static_cast<int>(r4), 1);
}

CUDAQ_TEST(MeasureResultTester, checkComparisons) {
  cudaq::measure_result a(int64_t(1), int64_t(10));
  cudaq::measure_result b(int64_t(1), int64_t(10));
  cudaq::measure_result c(int64_t(0), int64_t(10));
  cudaq::measure_result d(int64_t(1), int64_t(20));

  EXPECT_TRUE(a == b);
  EXPECT_TRUE(a != c);
  EXPECT_TRUE(a != d);
  EXPECT_TRUE(a == true);
  EXPECT_TRUE(true == a);
  EXPECT_TRUE(c == false);
  EXPECT_TRUE(c != true);
  EXPECT_TRUE(false != a);
}
