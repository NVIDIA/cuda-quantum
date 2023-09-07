/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1 if running in bash
// RUN: nvq++ -v %s -o %basename_t.x --target quantinuum --emulate && ./%basename_t.x |& FileCheck %s
// RUN: nvq++ -v %s -o %basename_t.x --target ionq --emulate && IONQ_API_KEY=0 ./%basename_t.x |& FileCheck %s

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

// CHECK: error: 'llvm.cond_br' op QIR base profile does not support control-flow
