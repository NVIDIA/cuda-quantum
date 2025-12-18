/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t --target quantinuum --emulate && CUDAQ_DUMP_JIT_IR=1 %t 2>&1 | FileCheck %s 
// RUN: nvq++ %s -o %t --target quantinuum --quantinuum-machine H2-1SC --emulate && CUDAQ_DUMP_JIT_IR=1 %t 2>&1 | FileCheck %s 
// RUN: nvq++ %s -o %t --target quantinuum --quantinuum-machine Helios-1SC --emulate && CUDAQ_DUMP_JIT_IR=1 %t 2>&1 | FileCheck %s --check-prefix=CHECK-NG
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qubit q0, q1;
  x(q0);
  x(q1);
  mz(q0);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  return 0;
}

// CHECK: requiredQubits
// CHECK: requiredResults
// CHECK-NG: required_num_qubits
// CHECK-NG: required_num_results
