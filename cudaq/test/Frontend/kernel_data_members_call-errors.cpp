/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: not cudaq-quake %s 2>&1 | FileCheck %s

#include <cudaq.h>

// The values of the data members are leading arguments of the call operator
// that a call from another kernel cannot supply.

struct Rotation {
  double angle;
  void operator()(cudaq::qubit &q) __qpu__ { rx(angle, q); }
};

struct Caller {
  void operator()() __qpu__ {
    cudaq::qubit c, q;
    // CHECK: error: the call does not match the arguments of the kernel; a kernel class with data members cannot be called from a kernel
    cudaq::control(Rotation{1.0}, c, q);
  }
};

int main() {
  cudaq::sample(Caller{});
  return 0;
}
// clang-format on
