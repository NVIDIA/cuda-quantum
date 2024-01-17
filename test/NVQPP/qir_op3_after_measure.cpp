/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ %cpp_std -v %s -o %t --target oqc --emulate && %t |& FileCheck %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t

#include <cudaq.h>
#include <iostream>

__qpu__ void init_state() {
  cudaq::qubit q0;
  cudaq::qubit q1;
  x(q0);
  mz(q0);
  // The Base Profile spec technically allows a measurement like this because it
  // isn't operating on an already-measured qubit, but it requires that the
  // compiler to reorder the q1 operation to be before the q0 measurement.
  x(q1);
  mz(q1);
};

int main() {
  auto result = cudaq::sample(1000, init_state);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

// CHECK-NOT: reversible function __quantum__qis__x__body came after irreversible function
