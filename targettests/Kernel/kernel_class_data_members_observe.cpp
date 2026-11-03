/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: if %quantinuum_avail; then nvq++ -target quantinuum -emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithm.h>

// The data members of a kernel class are read from the object that is passed
// to `cudaq::observe`, and they come before the explicit arguments of the
// kernel. The expectation values here are exact: every state is a
// computational basis state.

struct Obs {
  bool on = true;
  int target = 0;
  void operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    if (on)
      x(q[target]);
    rx(theta, q[1]);
  }
};

static void report(const char *tag, Obs &k, double theta) {
  double z0 = cudaq::observe(k, cudaq::spin_op::z(0), theta);
  double z1 = cudaq::observe(k, cudaq::spin_op::z(1), theta);
  std::printf("%s: %.3f %.3f\n", tag, z0, z1);
}

int main() {
  Obs a;
  report("default", a, 0.0);

  Obs b;
  b.target = 1;
  report("target", b, 0.0);

  Obs c;
  c.on = false;
  report("off", c, 0.0);

  // A member and an explicit argument together: rx(pi) flips qubit 1.
  Obs d;
  report("theta", d, M_PI);
  return 0;
}

// CHECK: default: -1.000 1.000
// CHECK: target: 1.000 -1.000
// CHECK: off: 1.000 1.000
// CHECK: theta: -1.000 -1.000
