/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <algorithm>
#include <cudaq.h>

struct iqpe {
  std::vector<bool> operator()() __qpu__ {
    std::vector<bool> results(4);
    cudaq::qarray<2> q;
    h(q[0]);
    x(q[1]);
    for (int i = 0; i < 8; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    h(q[0]);
    results[0] = mz(q[0]);
    reset(q[0]);

    h(q[0]);
    for (int i = 0; i < 4; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (results[0])
      rz(-M_PI / 2., q[0]);

    h(q[0]);
    results[1] = mz(q[0]);
    reset(q[0]);

    h(q[0]);
    for (int i = 0; i < 2; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (results[0])
      rz(-M_PI / 4., q[0]);

    if (results[1])
      rz(-M_PI / 2., q[0]);

    h(q[0]);
    results[2] = mz(q[0]);
    reset(q[0]);
    h(q[0]);
    r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (results[0])
      rz(-M_PI / 8., q[0]);

    if (results[1])
      rz(-M_PI_4, q[0]);

    if (results[2])
      rz(-M_PI_2, q[0]);

    h(q[0]);
    results[3] = mz(q[0]);
    return results;
  }
};

int main() {

  int nShots = 10;
  auto results = cudaq::run(nShots, iqpe{});
  // Get the counts for all the measurements
  auto count_bit = [&](std::size_t idx) {
    return std::count_if(results.begin(), results.end(),
                         [idx](auto &r) { return r[idx]; });
  };
  printf("Iterative QPE Results:\n");
  printf("First : { 1:%zu }\n", count_bit(0));
  printf("Second: { 1:%zu }\n", count_bit(1));
  printf("Third : { 0:%zu }\n", 10 - count_bit(2));
  printf("Final : { 1:%zu }\n", count_bit(3));

  return 0;
}

// CHECK: Iterative QPE Results:
// CHECK: First : { 1:10 }
// CHECK: Second: { 1:10 }
// CHECK: Third : { 0:10 }
// CHECK: Final : { 1:10 }
