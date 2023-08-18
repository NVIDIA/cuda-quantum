/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++ %s --target iqm --emulate --iqm-machine Apollo -o %t.x && %t.x | FileCheck %s

// CHECK: { 0:1000 }

template <std::size_t N>
struct kernel_with_z {
  auto operator()() __qpu__ {
    cudaq::qreg<N> q;
    z<cudaq::ctrl>(q[0], q[1]);
    mz(q[0]);
  }
};

int main() {
  auto kernel = kernel_with_z<2>{};
  auto counts = cudaq::sample(kernel);
  counts.dump();
  return 0;
}
