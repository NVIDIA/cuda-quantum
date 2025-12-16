/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1| if running in bash
// RUN: nvq++ %s -o %t --target quantinuum --quantinuum-machine Helios-1SC --emulate && %t |& FileCheck %s
// RUN: nvq++ %s -o %t --target qci --emulate && %t |& FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ bool init_state(int N) {
  cudaq::qvector q(N);
  x(q[0]);
  return mz(q[99]); // compiler can't catch this error, but runtime can
};

int main() {
  auto result = cudaq::run(100, init_state, 5);
  return 0;
}

// CHECK: error: 'quake.extract_ref' op invalid index [99] because >= size [5]
