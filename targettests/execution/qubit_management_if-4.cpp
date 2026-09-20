/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// clang-format off
// RUN: nvq++ --build-target-from-config %cudaq_src_dir/targettests/targets/opt-test.yml --target-option dep-analysis,qpp %s -o %t && %t
// clang-format on

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    bool res;
    // Should be able to lift mz(q) before
    if (true) {
      x(q);
      y(q);
      res = true;
    } else {
      x(q);
      y(q);
      res = false;
    }

    return res;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
