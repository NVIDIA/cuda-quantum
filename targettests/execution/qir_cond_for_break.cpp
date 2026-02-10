/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// RUN: CUDAQ_DEFAULT_SIMULATOR=stim nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>

struct kernel {
  int operator()(const int n_iter) __qpu__ {
    cudaq::qubit q0;
    for (int i = 0; i < n_iter; i++) {
      h(q0);
      auto q0result = mz(q0);
      if (q0result)
        return i; // loop until it lands heads
    }
    return n_iter; // didn't break early
  }
};

int main() {

  int nShots = 100;
  int nIter = 20;
  cudaq::set_random_seed(13);

  auto results = cudaq::run(/*shots=*/nShots, kernel{}, nIter);

  // Count the maximum number of iterations it took across all shots
  int nIterRan = results.empty() ? 0 : *std::ranges::max_element(results);

  int ret = 0; // return status

  if (nIterRan < nIter) {
    std::cout << "SUCCESS: nIterRan (" << nIterRan << ") < nIter (" << nIter
              << ")\n";
    ret = 0;
  } else {
    std::cout << "FAILURE: nIterRan (" << nIterRan << ") >= nIter (" << nIter
              << ")\n";
    ret = 1;
  }

  return ret;
}

// CHECK: SUCCESS
