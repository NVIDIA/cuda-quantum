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
// RUN: nvq++ -DCUDAQ_QUANTUM_DEVICE %s -o %t && %t | FileCheck %s

#include <cstdio>
#include <cudaq.h>
#include <exception>

// A state of a quantum device is the name of a kernel and its arguments. A
// kernel class without data members has nothing else, so getting its state
// must keep working. (A kernel class with data members is an error, which
// `kernel_class_data_members_get_state-errors.cpp` checks.)

struct WithoutMembers {
  void operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    rx(theta, q[1]);
  }
};

int main() {
  try {
    auto ok = cudaq::get_state(WithoutMembers{}, M_PI);
    std::printf("without members: state made\n");
  } catch (const std::exception &e) {
    std::printf("unexpected error: %s\n", e.what());
  }
  return 0;
}

// CHECK: without members: state made
