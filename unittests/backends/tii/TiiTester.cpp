/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"
#include <gtest/gtest.h>

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

TEST(TiiTester, checkSimpleCircuitSync) {
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

TEST(TiiTester, checkSimpleCircuitAsync) {
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
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
