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
  void operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    h(q0);
    auto q0result = mz(q0);
    if (q0result)
      x(q1);
    auto q1result = mz(q1); // Every q1 measurement will be the same as q0
  }
};

int main() {

  int nShots = 100;
  // Sample
  auto counts = cudaq::sample(/*shots=*/nShots, kernel{});
  counts.dump();
  // Assert that all shots contained "00" or "11", exclusively
  if (counts.count("00") + counts.count("11") != nShots) {
    std::cout << "counts00 (" << counts.count("00") << ") + counts11 ("
              << counts.count("11") << ") != nShots (" << nShots << ")\n";
    return 1;
  }
  std::cout << "SUCCESS\n";
  return 0;
}

// CHECK: SUCCESS
