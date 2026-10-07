/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/SampleResult.h"
#include <type_traits>
#include <utility>

using namespace cudaq;

CUDAQ_TEST(sample_resultTester, checkConstruction) {
  ExecutionResult r{CountsDictionary{{"0", 400}, {"1", 600}}};
  cudaq::sample_result mc(r);
  EXPECT_EQ(2, mc.size());
}

CUDAQ_TEST(sample_resultTester, checkMoveConstructionPreservesNamedResult) {
  ExecutionResult input("result");
  input.appendResult("00", 2);
  input.appendResult("01", 3);
  input.appendResult("11", 5);
  input.expectationValue = 0.4;
  const ExecutionResult expected(input);

  sample_result result(std::move(input));

  EXPECT_EQ(result.get_total_shots(), 10);
  EXPECT_EQ(result.register_names(), std::vector<std::string>{"result"});
  EXPECT_EQ(result.to_map("result"), expected.counts);
  EXPECT_EQ(result.sequential_data("result"), expected.sequentialData);
  EXPECT_DOUBLE_EQ(result.probability("01", "result"), 0.3);
  EXPECT_TRUE(result.has_expectation("result"));
  EXPECT_DOUBLE_EQ(result.expectation("result"), 0.4);
}

CUDAQ_TEST(sample_resultTester, checkMoveConstructionPreservesEmptyResult) {
  sample_result result(ExecutionResult{});

  EXPECT_EQ(result.get_total_shots(), 0);
  EXPECT_EQ(result.size(), 0);
  EXPECT_TRUE(result.sequential_data().empty());
}

CUDAQ_TEST(MeasureCountsTester, checkMoveConstructionTransfersShotStorage) {
  static_assert(std::is_nothrow_move_constructible_v<ExecutionResult>);
  ExecutionResult original("measurements");
  original.appendResult("01", 2);
  original.appendResult("10", 1);
  original.expectationValue = -1.0;
  const ExecutionResult expected(original);
  const auto *shotStorage = original.sequentialData.data();

  ExecutionResult moved(std::move(original));

  EXPECT_EQ(moved.sequentialData.data(), shotStorage);
  EXPECT_EQ(moved.counts, expected.counts);
  EXPECT_EQ(moved.sequentialData, expected.sequentialData);
  EXPECT_EQ(moved.registerName, expected.registerName);
  EXPECT_EQ(moved.expectationValue, expected.expectationValue);
}

CUDAQ_TEST(MeasureCountsTester, checkMoveAssignmentTransfersShotStorage) {
  static_assert(std::is_nothrow_move_assignable_v<ExecutionResult>);
  ExecutionResult original("measurements");
  original.appendResult("01", 2);
  original.appendResult("10", 1);
  original.expectationValue = -1.0;
  const ExecutionResult expected(original);
  const auto *shotStorage = original.sequentialData.data();
  ExecutionResult moved;
  moved.appendResult("00", 4);
  moved.expectationValue = 1.0;

  moved = std::move(original);

  EXPECT_EQ(moved.sequentialData.data(), shotStorage);
  EXPECT_EQ(moved.counts, expected.counts);
  EXPECT_EQ(moved.sequentialData, expected.sequentialData);
  EXPECT_EQ(moved.registerName, expected.registerName);
  EXPECT_EQ(moved.expectationValue, expected.expectationValue);
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

CUDAQ_TEST(MeasureCountsTester, checkMarginalIndexBounds) {
  ExecutionResult r{CountsDictionary{{"00", 400}, {"11", 600}}};
  cudaq::sample_result result(r);
  // Both valid boundary indices must remain accepted.
  EXPECT_NO_THROW(result.get_marginal({0}));
  EXPECT_NO_THROW(result.get_marginal({1}));
  // Index 2 equals the bitstring width and is therefore out of bounds.
  EXPECT_ANY_THROW(result.get_marginal({2}));
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

CUDAQ_TEST(MeasureCountsTester, checkDeserializeRejectsMalformed) {
  {
    std::vector<std::size_t> data = {255};
    cudaq::sample_result mm;
    EXPECT_ANY_THROW(mm.deserialize(data));
  }

  {
    std::vector<std::size_t> data = {1, static_cast<std::size_t>('a')};
    cudaq::sample_result mm;
    EXPECT_ANY_THROW(mm.deserialize(data));
  }

  {
    std::vector<std::size_t> data = {0, 255};
    cudaq::sample_result mm;
    EXPECT_ANY_THROW(mm.deserialize(data));
  }

  {
    std::vector<std::size_t> data = {0, 1, /*value*/ 1,
                                     /*length*/ static_cast<std::size_t>(-1),
                                     /*count*/ 1};
    cudaq::sample_result mm;
    EXPECT_ANY_THROW(mm.deserialize(data));
  }

  {
    std::vector<std::size_t> data = {255};
    ExecutionResult rr;
    EXPECT_ANY_THROW(rr.deserialize(data));
  }
}

CUDAQ_TEST(MeasureCountsTester, checkDeserializeAcceptsValid) {
  ExecutionResult r{CountsDictionary{{"01", 400}, {"11", 600}}};
  auto data = r.serialize();
  ExecutionResult rr;
  EXPECT_NO_THROW(rr.deserialize(data));
  EXPECT_TRUE(rr == r);
}
