/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t %s 2>&1 | FileCheck %s -check-prefix=FAIL

#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(toffoli, 3, 0,
                         {1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0})

__qpu__ void kernel() {
  cudaq::qvector q(3);
  x(q);
  toffoli(q[0], q[1], q[2]);
}

int main() {
  auto counts = cudaq::sample(kernel);
  for (auto &[bits, count] : counts) {
    printf("%s\n", bits.data());
  }
}

// CHECK: 110

// FAIL: failed to legalize operation 'quake.custom_op'
