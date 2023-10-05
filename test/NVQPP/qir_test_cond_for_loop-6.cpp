/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %basename_t.x && ./%basename_t.x
// XFAIL: *
// ^^^^^ Produces nvq++ compiler segmentation fault
// clang-format on

// The test here is the assert statement.

#include <cudaq.h>
#include <iostream>

struct kernel {
  void operator()(const int n_iter) __qpu__ {
    cudaq::qubit q0;
    bool keepGoing = true;
    for (int i = 0; i < n_iter; i++) {
      if (keepGoing) {
        h(q0);
        auto q0result = mz(q0);
        if (!q0result)
          keepGoing = false;
      }
    }
  }
};

int main() {

  int nShots = 100;
  int nIter = 20;
  cudaq::set_random_seed(13);

  // Sample
  auto counts = cudaq::sample(/*shots=*/nShots, kernel{}, nIter);
  counts.dump();

  // Count how many iterations it took
  int nIterRan = 0;
  for (int i = 0; i < nIter; i++) {
    char regName[32];
    snprintf(regName, sizeof(regName), "q0result%02d", i);
    if (counts.size(regName) == 0) {
      nIterRan = i + 1;
      break;
    }
  }

  assert(nIterRan < nIter);
  return 0;
}
