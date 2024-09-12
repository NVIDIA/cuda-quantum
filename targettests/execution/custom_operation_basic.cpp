/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && %t %s 2>&1 | FileCheck %s -check-prefix=FAIL

#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(custom_h, 1, 0,
                         {M_SQRT1_2, M_SQRT1_2, M_SQRT1_2, -M_SQRT1_2})

CUDAQ_REGISTER_OPERATION(custom_cnot, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0})

__qpu__ void bell_pair() {
  cudaq::qubit q, r;
  custom_h(q);
  custom_cnot(q, r);
}

int main() {
  auto counts = cudaq::sample(bell_pair);
  for (auto &[bits, count] : counts) {
    printf("%s\n", bits.data());
  }
}

// CHECK: 11
// CHECK: 00

// FAIL: failed to legalize operation 'quake.custom_op'
