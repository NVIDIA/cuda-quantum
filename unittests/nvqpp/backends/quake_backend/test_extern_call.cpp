/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qm_wait.h"

#include <cudaq.h>
#include <iostream>

// A Ramsey sequence. The mock server checks for the call in wire form.
__qpu__ bool ramsey_single(double wait_duration) {
  cudaq::qubit q;
  rx(M_PI_2, q);
  __qm__wait_function(wait_duration, q);
  rx(M_PI_2, q);
  return mz(q);
}

int main() {
  auto results = cudaq::run(100, ramsey_single, 1.0);
  if (results.size() != 100) {
    std::cerr << "ramsey_single returned " << results.size()
              << " results, expected 100\n";
    return 1;
  }
  std::cout << "Successfully sampled a kernel with an external quantum call\n";
  return 0;
}
