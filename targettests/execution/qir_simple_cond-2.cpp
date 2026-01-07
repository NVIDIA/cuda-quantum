/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --enable-mlir %s -o %t
// XFAIL: *
// ^^^^^ This is caused by this error: invalid instruction found:   %2 = xor i1 %0, true
//       This error is reasonable given the current version of the Adaptive
//       Profile that is supported, but future versions of the Adaptive
//       Profile (that contain optional capabilities) may legalize this.
// clang-format on

#include <cudaq.h>
#include <iostream>

struct kernel {
  void operator()() __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    cudaq::qubit q2;
    h(q0);
    h(q1);
    auto result0 = mz(q0);
    auto result1 = mz(q1);
    if (result0 && result1)
      x(q2); // toggle q2 when both q0 and q1 are heads
    auto result2 = mz(q2);
  }
};

int main() {

  int nShots = 100;
  cudaq::set_random_seed(13);

  // Sample
  auto counts = cudaq::sample(/*shots=*/nShots, kernel{});
  counts.dump();

  auto q2result_0 = counts.count("0", "q2result");
  auto q2result_1 = counts.count("1", "q2result");
  if (q2result_0 + q2result_1 != nShots) {
    std::cout << "q2result_0 (" << q2result_0 << ") + q2result_1 ("
              << q2result_1 << ") != nShots (" << nShots << ")\n";
    return 1;
  }
  if (q2result_0 < static_cast<int>(0.3 * nShots) ||
      q2result_0 > static_cast<int>(0.7 * nShots)) {
    std::cout << "q2result_0 (" << q2result_0
              << ") is not within expected range ["
              << static_cast<int>(0.3 * nShots) << ","
              << static_cast<int>(0.7 * nShots) << "]\n";
    return 2;
  }
  std::cout << "SUCCESS\n";
  return 0;
}

// CHECK: SUCCESS
