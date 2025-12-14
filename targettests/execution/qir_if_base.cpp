/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ %cpp_std -v %s -o %t --target ionq --emulate && %t |& FileCheck %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t

#include <cudaq.h>
#include <iostream>

__qpu__ bool qir_test() {
  cudaq::qubit q0;
  cudaq::qubit q1;
  x(q0);
  auto measureResult = mz(q0);
  if (measureResult)
    x(q1);
  return mz(q1);
};

int main() {
  auto result = cudaq::run(1000, qir_test);
  return 0;
}

// Note: Need to support `run` with Base profile so as to enable this check
// XCHECK: Do you have if statements in a Base Profile QIR program
// CHECK: `run` is not yet supported on this target.
