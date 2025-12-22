/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi

#include <cudaq.h>

struct angle_test {
  void operator()(const double param) __qpu__ {
    cudaq::qubit q1;
    cudaq::qubit q2;
    h(q1);
    h(q2);
    x(q1);
    x(q2);
    rx(param, q1);
    rx(-param, q1);
    mz(q1);
    mz(q2);
  }
};

int main() {
  auto counts = cudaq::sample(angle_test{}, 0.1);
  for (auto &[bits, count] : counts) {
    printf("%s\n", bits.data());
  }
}

// CHECK: 00
// CHECK: 11
