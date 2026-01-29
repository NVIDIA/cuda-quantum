/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
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
  std::vector<bool> operator()() __qpu__ {
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
    return {result0, result1, result2};
  }
};

int main() {

  int nShots = 100;
  cudaq::set_random_seed(13);

  auto counts = cudaq::run(/*shots=*/nShots, kernel{});

  std::size_t q2result_0 = 0, q2result_1 = 0;
  for (auto r : counts) {
    if (r[2])
      q2result_1++;
    else
      q2result_0++;
  }

  printf("q2 : { 1:%zu }\n", q2result_1);
  printf("q2 : { 0:%zu }\n", q2result_0);

  if (q2result_0 + q2result_1 != nShots) {
    std::cout << "q2result_0 (" << q2result_0 << ") + q2result_1 ("
              << q2result_1 << ") != nShots (" << nShots << ")\n";
    return 1;
  }
  if (q2result_0 < static_cast<int>(0.23 * nShots) ||
      q2result_0 > static_cast<int>(0.77 * nShots)) {
    std::cout << "q2result_0 (" << q2result_0
              << ") is not within expected range ["
              << static_cast<int>(0.23 * nShots) << ","
              << static_cast<int>(0.77 * nShots) << "]\n";
    return 2;
  }
  std::cout << "SUCCESS\n";
  return 0;
}

// CHECK: SUCCESS
