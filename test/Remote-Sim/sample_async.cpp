/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target remote-sim --remote-sim-auto-launch 4 %s -o %t && %t 
// RUN: nvq++ --enable-mlir --target remote-sim --remote-sim-auto-launch 4 %s -o %t && %t
// clang-format on

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
  assert(num_qpus == 4);
  std::vector<cudaq::async_sample_result> countFutures;
  for (std::size_t i = 0; i < num_qpus; i++) {
    countFutures.emplace_back(cudaq::sample_async(i, simpleX{}, i + 1));
  }

  for (std::size_t i = 0; i < num_qpus; i++) {
    auto counts = countFutures[i].get();
    counts.dump();
    const std::string expectedBitStr(i + 1, '1');
    assert(counts.size() == 1);
    assert(counts.begin()->first == expectedBitStr);
  }
  return 0;
}
