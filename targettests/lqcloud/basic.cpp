/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile every documented LQCloud gate through the public C++ API.
// clang-format off
// RUN: if %lqcloud_avail; then nvq++ --target lqcloud --lqcloud-url https://example.invalid --lqcloud-qpu compile_only %s -o %t.x; fi

#include "cudaq.h"

int main() {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(3);
    h(q[0]);
    x(q[1]);
    y(q[2]);
    z(q[0]);
    s(q[1]);
    t(q[2]);
    rx(0.1, q[0]);
    ry(0.2, q[1]);
    rz(0.3, q[2]);
    u3(0.4, 0.5, 0.6, q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    y<cudaq::ctrl>(q[1], q[2]);
    z<cudaq::ctrl>(q[2], q[0]);
    swap(q[0], q[2]);
    mz(q);
  };

  auto counts = cudaq::sample(kernel);
  counts.dump();
  return 0;
}
