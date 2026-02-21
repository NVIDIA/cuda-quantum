/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin `RunAsync`]
#include <cstdio>
#include <cudaq.h>
#include <vector>

// Define a quantum kernel that returns an integer based on measurement
__qpu__ int simple_count(double angle) {
  auto q = cudaq::qubit();
  rx(angle, q);
  return mz(q);
}

int main() {
  // Execute the kernel asynchronously with different parameters
  std::vector<float> angles = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4};
  std::vector<std::future<std::vector<int>>> futures;
  for (auto i = 0; i < angles.size(); ++i) {
    futures.push_back(cudaq::run_async(0, 10, simple_count, angles[i]));
  }
  for (auto i = 0; i < futures.size(); ++i) {
    std::vector<int> results = futures[i].get();
    int ones_count = std::accumulate(results.begin(), results.end(), 0);
    printf("Angle %.1f : %d/10 ones measured\n", angles[i], ones_count);
  }
  return 0;
}
// [End `RunAsync`]
/* [Begin `RunAsyncOutput`]
Angle 0.0 : 0/10 ones measured
Angle 0.2 : 1/10 ones measured
Angle 0.4 : 1/10 ones measured
Angle 0.6 : 3/10 ones measured
Angle 0.8 : 0/10 ones measured
Angle 1.0 : 5/10 ones measured
Angle 1.2 : 4/10 ones measured
Angle 1.4 : 3/10 ones measured
[End `RunAsyncOutput`] */
