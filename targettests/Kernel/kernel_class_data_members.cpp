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
// Packed-only launch with `quake-synthesizer`.
// RUN: if %quantinuum_avail; then nvq++ -target quantinuum -emulate -fkernel-exec-kind=1 %s -o %t && %t | FileCheck %s; fi
// Streamlined launch with `argument-synthesis`.
// RUN: if %quantinuum_avail; then nvq++ -target quantinuum -emulate -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <iostream>

// The data members of a kernel class are read from the object that is launched
// and are passed to the kernel as arguments. A write to a member in the kernel
// only changes the kernel's copy.

struct Flip {
  char padding = 1;
  int count = 2;
  double angle = 3.14159265358979;
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    count = count + 1;
    for (int k = 0; k < count; ++k)
      rx(angle, q[k]);
    mz(q);
  }
};

// A `bool` member is a byte in the object and an `i1` in the kernel.
struct Gate {
  bool on = true;
  int target = 1;
  bool off = false;
  void operator()() __qpu__ {
    cudaq::qvector q(3);
    if (on)
      x(q[target]);
    if (off)
      x(q[0]);
    mz(q);
  }
};

int main() {
  Flip f;
  cudaq::sample(f, 5).dump();
  Flip g;
  g.count = 0;
  cudaq::sample(g, 5).dump();
  Gate a;
  cudaq::sample(a).dump();
  Gate b;
  b.on = false;
  b.off = true;
  cudaq::sample(b).dump();
  std::cout << "host: " << f.count << ' ' << g.count << '\n';
  return 0;
}

// CHECK: { 11100:1000 }
// CHECK: { 10000:1000 }
// CHECK: { 010:1000 }
// CHECK: { 100:1000 }
// CHECK: host: 2 0
