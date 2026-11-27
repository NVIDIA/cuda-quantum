/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

#include <cstdio>
#include <cudaq.h>

// The data members of a kernel class are read from the object that is passed
// to `cudaq::get_state`, and they come before the explicit arguments of the
// kernel. The state that is returned is then used as the argument of another
// kernel, which is sampled.

struct Prep {
  int target = 1;
  bool on = true;
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    if (on)
      x(q[target]);
  }
};

struct PrepTheta {
  bool on = true;
  void operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    if (on)
      x(q[0]);
    rx(theta, q[1]);
  }
};

__qpu__ void use(cudaq::state *s) {
  cudaq::qvector q(s);
  mz(q);
}

static void report(const char *tag, cudaq::state &s,
                   const std::vector<int> &basis) {
  std::printf("%s: %.3f\n", tag, std::abs(s.amplitude(basis)));
}

int main() {
  Prep a;
  auto sa = cudaq::get_state(a);
  report("default", sa, {0, 1});

  Prep b;
  b.target = 0;
  auto sb = cudaq::get_state(b);
  report("target", sb, {1, 0});

  Prep c;
  c.on = false;
  auto sc = cudaq::get_state(c);
  report("off", sc, {0, 0});

  // A member and an explicit argument together: rx(pi) flips qubit 1.
  PrepTheta d;
  auto sd = cudaq::get_state(d, M_PI);
  report("theta", sd, {1, 1});

  // Feed the state of a kernel class with data members to another kernel.
  cudaq::sample(use, &sd).dump();
  cudaq::sample(use, &sb).dump();
  return 0;
}

// CHECK: default: 1.000
// CHECK: target: 1.000
// CHECK: off: 1.000
// CHECK: theta: 1.000
// CHECK: { 11:1000 }
// CHECK: { 10:1000 }
