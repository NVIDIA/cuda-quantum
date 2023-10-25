/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ -v %s -o %basename_t.x --target oqc --emulate && ./%basename_t.x |& FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void function_operation(cudaq::qubit &q) {
  x(q);   // base profile does not allow operations after measurements
}

__qpu__ void init_state() {
  cudaq::qubit q;
  x(q);
  auto result = mz(q);
  function_operation(q);   // base profile does not allow operations after measurements
};

int main() {
  auto result = cudaq::sample(1000, init_state);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

// CHECK: reversible function __quantum__qis__x__body came after irreversible function
