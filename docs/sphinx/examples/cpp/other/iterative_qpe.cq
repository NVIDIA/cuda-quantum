/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ iterative_qpe.cpp -o qpe.x && ./qpe.x
// ```

#include <cudaq.h>

struct iqpe {
  void operator()() __qpu__ {
    cudaq::qreg<2> q;
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
    mz(q[0]);
  }
};

int main() {
  auto counts = cudaq::sample(/*shots*/ 10, iqpe{});
  counts.dump();

  return 0;
}
