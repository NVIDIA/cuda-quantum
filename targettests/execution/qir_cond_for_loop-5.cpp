/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: CUDAQ_DEFAULT_SIMULATOR=stim nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --enable-mlir %s -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>

#define NUM_ITERATIONS 5

struct kernel {
  void operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    int resultArray[NUM_ITERATIONS];
    for (int i = 0; i < NUM_ITERATIONS; i++) {
      h(q0);
      resultArray[i] = mz(q0);
      if (resultArray[i])
        x(q1); // toggle q1 on every q0 coin toss that lands heads
    }
    auto q1result = mz(q1); // the measured q1 should contain the parity bit for
                            // the q0 measurements
  }
};

int main() {

  int nShots = 100;
  cudaq::set_random_seed(13);

  // Sample
  auto counts = cudaq::sample(/*shots=*/nShots, kernel{});
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
