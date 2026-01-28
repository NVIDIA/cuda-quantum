/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"
#include "gtest/gtest.h"

std::string backendString = "qibo;";

bool result_maps_are_matching(
    const std::unordered_map<std::string, std::size_t> &results,
    const std::unordered_map<std::string, std::size_t> &expected) {
  for (const auto &[key, value] : expected) {
    auto it = results.find(key);
    if (it == results.end() || it->second != value) {
      return false;
    }
  }
  return true;
}

TEST(QiboTester, checkSimpleCircuitSync) {
  // Initialize the platform
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  // Create a simple circuit
  auto kernel = cudaq::make_kernel();
  auto qubits = kernel.qalloc(2);
  kernel.h(qubits[0]);
  kernel.mz(qubits);

  // Execute the circuit
  auto counts = cudaq::sample(kernel);
  counts.dump();
  // Check results
  EXPECT_EQ(counts.size(), 2);
  std::unordered_map<std::string, std::size_t> expected = {{"00", 500},
                                                           {"11", 500}};
  EXPECT_TRUE(result_maps_are_matching(counts.to_map(), expected));
}

TEST(QiboTester, checkSimpleCircuitAsync) {
  // Initialize the platform
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  // Create a simple circuit
  auto kernel = cudaq::make_kernel();
  auto qubits = kernel.qalloc(2);
  kernel.h(qubits[0]);
  kernel.mz(qubits);

  // Execute the circuit
  auto counts = cudaq::sample_async(kernel).get();
  counts.dump();
  // Check results
  EXPECT_EQ(counts.size(), 2);
  std::unordered_map<std::string, std::size_t> expected = {{"00", 500},
                                                           {"11", 500}};
  EXPECT_TRUE(result_maps_are_matching(counts.to_map(), expected));
}

int main(int argc, char **argv) {
  setenv("QIBO_API_TOKEN", "api_key", 0);
  setenv("QIBO_API_URL", "http://localhost:62450", 0);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
