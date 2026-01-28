/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Documentation]
// Compile and run with:
// ```
// nvq++ iterative_qpe.cpp -o qpe.x && ./qpe.x
// ```

#include <algorithm>
#include <cudaq.h>

struct iqpe {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qarray<2> q;
    h(q[0]);
    x(q[1]);
    for (int i = 0; i < 8; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    h(q[0]);
    auto cr0 = mz(q[0]);
    reset(q[0]);

    h(q[0]);
    for (int i = 0; i < 4; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (cr0)
      rz(-M_PI / 2., q[0]);

    h(q[0]);
    auto cr1 = mz(q[0]);
    reset(q[0]);

    h(q[0]);
    for (int i = 0; i < 2; i++)
      r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (cr0)
      rz(-M_PI / 4., q[0]);

    if (cr1)
      rz(-M_PI / 2., q[0]);

    h(q[0]);
    auto cr2 = mz(q[0]);
    reset(q[0]);
    h(q[0]);
    r1<cudaq::ctrl>(-5 * M_PI / 8., q[0], q[1]);

    if (cr0)
      rz(-M_PI / 8., q[0]);

    if (cr1)
      rz(-M_PI_4, q[0]);

    if (cr2)
      rz(-M_PI_2, q[0]);

    h(q[0]);
    return {cr0, cr1, cr2, mz(q[0])};
  }
};

int main() {
  auto results = cudaq::run(/*shots*/ 10, iqpe{});
  // Get the counts for cr0, cr1, cr2 and the final measurement
  auto count_bit = [&](std::size_t idx) {
    return std::count_if(results.begin(), results.end(),
                         [idx](auto &r) { return r[idx]; });
  };
  printf("Iterative QPE Results:\n");
  printf("cr0 : { 1:%zu }\n", count_bit(0));
  printf("cr1 : { 1:%zu }\n", count_bit(1));
  printf("cr2 : { 0:%zu }\n", 10 - count_bit(2));
  printf("final: { 1:%zu }\n", count_bit(3));
  return 0;
}
// [End Documentation]
