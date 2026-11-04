/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Defining `CUDAQ_QUANTUM_DEVICE` selects the `get_state` of a quantum device,
// which records the name and the arguments of the kernel. That is all this test
// needs, so it does not depend on a backend being built.
// RUN: nvq++ -DCUDAQ_QUANTUM_DEVICE %s -o %t && not %t 2>&1 | FileCheck %s

#include <cudaq.h>
#include <exception>
#include <iostream>

// A state of a quantum device is the name of a kernel and its arguments. The
// object of a kernel class is not part of that, so `flip` would be lost and the
// state would be made with the wrong value. Getting the state of a kernel class
// with data members must be an error, and not a state of the wrong circuit.

struct Prep {
  int flip = 1;
  void operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    if (flip)
      x(q[0]);
    rx(theta, q[1]);
  }
};

__qpu__ void use(cudaq::state *s) {
  cudaq::qvector q(s);
  mz(q);
}

int main() {
  try {
    auto s = cudaq::get_state(Prep{}, M_PI);
    cudaq::sample(use, &s).dump();
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

// CHECK: error: kernel {{.*}}Prep{{.*}} is a class with data members
// CHECK-NOT: {{[01]+}}:{{[0-9]+}}
