/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target qpp-cpu %s -o %basename_t.x && ./%basename_t.x
// RUN: CUDAQ_DEFAULT_SIMULATOR="density-matrix-cpu" nvq++ %s -o %basename_t.x && ./%basename_t.x
// RUN: CUDAQ_DEFAULT_SIMULATOR="foo" nvq++ %s -o %basename_t.x && ./%basename_t.x
// RUN: CUDAQ_DEFAULT_SIMULATOR="qpp-cpu" nvq++ --target quantinuum --emulate %s -o %basename_t.x && ./%basename_t.x

#include <cudaq.h>

struct ghz {
  auto operator()(int N) __qpu__ {
    cudaq::qreg q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(ghz{}, 4);
  counts.dump();
  return 0;
}
