/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --lower-frequencies | FileCheck %s --check-prefix=NATIVE
// clang-format on

#include <cudaq.h>

struct prepare {
  void operator()(cudaq::qarray<2> &q, double theta) __qpu__ {
    ry(theta, q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  }
};

// The library wrapper remains an ordinary kernel. Its IR explicitly contains
// the preparation-taking sampling operation and publication of its result.
// clang-format off
// CHECK: quake.frequencies
// CHECK-SAME: shots
// CHECK: quake.emit
// NATIVE: call @__nvqir__frequencies_available
// NATIVE: cc.if
// NATIVE: call @__nvqir__frequencies_sample
// NATIVE: else
// NATIVE: cc.loop
// clang-format on
int main() { cudaq::sample_vnext(100, prepare{}, 0.5); }
