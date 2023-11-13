/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Note: change |& to 2>&1| if running in bash
// RUN: cudaq-quake %s |& FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void init_state() {
  cudaq::qreg<5> q;
  x(q[0]);
  mz(q[99]); // compiler can flag this as an error
};

int main() {
  auto result = cudaq::sample(1000, init_state);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

// CHECK: error: 'quake.extract_ref' op invalid index [99] because >= size [5]
