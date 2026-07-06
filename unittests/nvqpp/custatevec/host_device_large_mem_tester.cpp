/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <gtest/gtest.h>

TEST(HostDeviceLargeMemTester, BasicCheck) {
  // This must be set before running this test to make sure host-device
  // migration is activated
  EXPECT_TRUE(std::getenv("CUDAQ_MAX_GPU_MEMORY_GB") != nullptr);
  const auto gpuMemGb = std::atoi(std::getenv("CUDAQ_MAX_GPU_MEMORY_GB"));
  // This test expects to test large memory
  EXPECT_GE(gpuMemGb, 16);
  EXPECT_TRUE(std::getenv("CUDAQ_MAX_CPU_MEMORY_GB") != nullptr);
  const auto cpuMemGb = std::atoi(std::getenv("CUDAQ_MAX_CPU_MEMORY_GB"));
  // Some very large value
  EXPECT_GE(cpuMemGb, 1024);
}

struct ghz {
  auto operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; ++i) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

TEST(HostDeviceLargeMemTester, checkBell) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 32;
  auto counts = cudaq::sample(ghz{}, numQubits);
  counts.dump();
  int counter = 0;
  const std::string allZero(numQubits, '0');
  const std::string allOne(numQubits, '1');
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == allZero || bits == allOne);
  }
  EXPECT_EQ(counter, 1000);
}
