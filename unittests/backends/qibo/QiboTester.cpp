/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "CUDAQTestUtils.h"
#include "cudaq/platform/quantum_platform.h"
#include "gtest/gtest.h"
#include <string>
#include <unordered_map>

std::string mockPort = "62450";
std::string auth_token = "api_key";
std::string backendStringTemplate =
    "qibo;emulate;false;url;http://localhost:{};auth_token;{};";

bool result_maps_are_matching(
  const std::unordered_map<std::string, std::size_t>& results,
  const std::unordered_map<std::string, std::size_t>& expected) {
    for (const auto& [key, value] : expected) {
        auto it = results.find(key);
        if (it == results.end() || it->second != value) {
            return false;
        }
    }
    return true;
}

TEST(QiboTester, checkSimpleCircuitSync) {
  // Initialize the platform
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, auth_token);
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
  std::unordered_map<std::string, std::size_t> expected = {
    {"00", 500},
    {"11", 500}
  };
  EXPECT_TRUE(result_maps_are_matching(counts.to_map(), expected));
}

TEST(QiboTester, checkSimpleCircuitAsync) {
  // Initialize the platform
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, auth_token);
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
  std::unordered_map<std::string, std::size_t> expected = {
    {"00", 500},
    {"11", 500}
  };
  EXPECT_TRUE(result_maps_are_matching(counts.to_map(), expected));
}