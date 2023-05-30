/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/MeasureCounts.h"

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
  EXPECT_NEAR(-1. / 5., mc.exp_val_z(), 1e-9);
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
