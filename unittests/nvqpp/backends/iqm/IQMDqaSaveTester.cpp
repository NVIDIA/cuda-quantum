
/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 IQM Quantum Computers                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"
#include <filesystem>

namespace fs = std::filesystem;
const char dqa_filename[] = "dqa_mock_qpu_saved.txt";

CUDAQ_TEST(IQMTester, dqaSaveFile) {

  fs::remove(dqa_filename);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  kernel.mz(qubit[1]);

  auto counts = cudaq::sample(kernel);

  EXPECT_GE(counts.size(), 2);
  EXPECT_LE(counts.size(), 4);
}

CUDAQ_TEST(IQMTester, dqaLoadFile) {

  // Test 2: use quantum architecture file referenced in environment variable
  // with the content of this file.
  ASSERT_TRUE(fs::exists(dqa_filename));
  EXPECT_THAT(getenv("IQM_QPU_QA"), dqa_filename);

  auto kernel2 = cudaq::make_kernel();
  auto qubit2 = kernel2.qalloc(2);
  kernel2.h(qubit2[0]);
  kernel2.mz(qubit2[0]);
  kernel2.mz(qubit2[1]);

  auto counts = cudaq::sample(kernel2);

  EXPECT_GE(counts.size(), 2);
  EXPECT_LE(counts.size(), 4);
}

CUDAQ_TEST(IQMTester, dqaRemoveFile) {
  // Test 3: The file must have been removed by the previous run
  ASSERT_FALSE(fs::exists(dqa_filename));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
