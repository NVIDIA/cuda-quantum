
/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "cudaq/algorithm.h"
#include "cudaq/runtime/logger/logger.h"
#include <fstream>
#include <gtest/gtest.h>
#include <regex>

CUDAQ_TEST(ScalewayTester, checkSampleSync) {
  GTEST_SKIP() << "XFAIL: outdated by the explicit-record sampling refactor "
                  "(the mock samples a kernel that measures for itself).";
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.record(qubit[0]);

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(ScalewayTester, checkSampleAsync) {
  GTEST_SKIP() << "XFAIL: outdated by the explicit-record sampling refactor "
                  "(the mock samples a kernel that measures for itself).";
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.record(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(ScalewayTester, executeOneMeasuredQubitProgram) {
  GTEST_SKIP() << "XFAIL: outdated by the explicit-record sampling refactor "
                  "(the mock samples a kernel that measures for itself).";
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.y(qubit[0]);
  kernel.z(qubit[0]);
  kernel.h(qubit[0]);
  kernel.record(qubit[0]);

  auto counts = cudaq::sample(kernel);
  EXPECT_EQ(counts.size(), 2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
