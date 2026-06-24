/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++ --target opt-test --target-option dep-analysis,qpp %s -o %t && %t

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    bool res;

    h(q);
    bool b = mz(q);
    
    // Should be able to lift x(p/r)
    if (b) {
      cudaq::qubit p;
      x(p);
      y(p);
      res = mz(p);
    } else {
      cudaq::qubit r;
      x(r);
      z(r);
      res = mz(r);
    }

    return res;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
