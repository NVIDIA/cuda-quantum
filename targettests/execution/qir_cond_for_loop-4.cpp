/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>

struct kernel {
  void operator()(const int n_iter) __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    std::vector<int> resultVector(n_iter);
    for (int i = 0; i < n_iter; i++) {
      h(q0);
      resultVector[i] = mz(q0);
      if (resultVector[i])
        x(q1); // toggle q1 on every q0 coin toss that lands heads
    }
    auto q1result = mz(q1); // the measured q1 should contain the parity bit for
                            // the q0 measurements
  }
};

int main() {

  int nShots = 100;
  int nIter = 5;
  cudaq::set_random_seed(13);

  // Sample
  auto counts = cudaq::sample(/*shots=*/nShots, kernel{}, nIter);
  counts.dump();

  auto q1result_0 = counts.count("0", "q1result");
  auto q1result_1 = counts.count("1", "q1result");
  if (q1result_0 + q1result_1 != nShots) {
    std::cout << "q1result_0 (" << q1result_0 << ") + q1result_1 ("
              << q1result_1 << ") != nShots (" << nShots << ")\n";
    return 1;
  }
  if (q1result_0 < static_cast<int>(0.3 * nShots) ||
      q1result_0 > static_cast<int>(0.7 * nShots)) {
    std::cout << "q1result_0 (" << q1result_0
              << ") is not within expected range ["
              << static_cast<int>(0.3 * nShots) << ","
              << static_cast<int>(0.7 * nShots) << "]\n";
    return 2;
  }

  std::cout << "SUCCESS\n";
  return 0;
}

// CHECK: SUCCESS
