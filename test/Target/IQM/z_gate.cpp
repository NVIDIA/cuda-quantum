/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++ %s --target iqm --emulate --iqm-qpu-architecture Apollo -o %t.x && %t.x | FileCheck %s

template <std::size_t N>
struct kernel_with_z {
  auto operator()() __qpu__ {
    cudaq::qreg<N> q;

    // FIXME: an std::runtime_error exception when using --emulate
    // loc("<builder>":1:1): error: does not contain an entrypoint
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

// CHECK: { 0:{{[0-9]+}}, 1:{{[0-9]+}} }