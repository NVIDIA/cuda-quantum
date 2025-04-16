/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "cudaq/algorithm.h"
#include <fstream>
#include <gtest/gtest.h>
#include <stdlib.h>

// TODO: Implement proper test configuration
std::string mockPort = "62448";
std::string backendStringTemplate =
    "quantum_machines;emulate;false;url;http://localhost:{}";

CUDAQ_TEST(QuantumMachinesTester, checkSampleSync) {
  // TODO: Implement synchronous sampling test
  GTEST_SKIP() << "Test not implemented yet.";
}

CUDAQ_TEST(QuantumMachinesTester, checkSampleAsync) {
  // TODO: Implement asynchronous sampling test
  GTEST_SKIP() << "Test not implemented yet.";
}

CUDAQ_TEST(QuantumMachinesTester, checkSampleAsyncLoadFromFile) {
  // TODO: Implement asynchronous sampling with file loading test
  GTEST_SKIP() << "Test not implemented yet.";
}

int main(int argc, char **argv) {
  setenv("QUANTUM_MACHINES_API_KEY", "00000000000000000000000000000000", 0);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
} 