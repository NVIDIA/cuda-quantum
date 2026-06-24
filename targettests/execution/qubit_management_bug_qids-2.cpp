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
    cudaq::qubit q,p,r;

    h(r);

    // Ensures that updates to qids get correctly propagated to sub-blocks
    if (true) {
      x(q);
      x(r);
      x<cudaq::ctrl>(q,p);
    } else {
      if (true)
        x(p);
      else
        x(p);
      y(q);
      x<cudaq::ctrl>(q,r);
    }

    bool b = mz(r);

    return b;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
