/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ -v %s -o %t --target ionq --emulate && %t |& FileCheck %s
// RUN: nvq++ --enable-mlir %s -o %t

#include <cudaq.h>
#include <iostream>

__qpu__ void qir_test() {
  cudaq::qubit q0;
  cudaq::qubit q1;
  x(q0);
  auto measureResult = mz(q0);
  if (measureResult)
    x(q1);
};

int main() {
  auto result = cudaq::sample(1000, qir_test);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

// CHECK: Do you have if statements in a Base Profile QIR program
