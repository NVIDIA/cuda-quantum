/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 4 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>

struct simpleX {
  auto operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    x(q);
    mz(q);
  }
};

int main() {
  auto &platform = cudaq::get_platform();
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);
  REMOTE_TEST_ASSERT(num_qpus == 4);
  std::vector<cudaq::async_sample_result> countFutures;
  // sample_async API with default shots
  for (std::size_t i = 0; i < num_qpus; i++) {
    countFutures.emplace_back(cudaq::sample_async(i, simpleX{}, i + 1));
  }
  // Shots-based sample_async API
  std::vector<cudaq::async_sample_result> countFuturesWithShots;
  for (std::size_t i = 0; i < num_qpus; i++) {
    countFuturesWithShots.emplace_back(
        cudaq::sample_async(/*shots=*/1024, /*qpu_id=*/i, simpleX{}, i + 1));
  }

  for (std::size_t i = 0; i < num_qpus; i++) {
    for (auto counts :
         {countFutures[i].get(), countFuturesWithShots[i].get()}) {
      counts.dump();
      const std::string expectedBitStr(i + 1, '1');
      REMOTE_TEST_ASSERT(counts.size() == 1);
      REMOTE_TEST_ASSERT(counts.begin()->first == expectedBitStr);
    }
  }
  return 0;
}
