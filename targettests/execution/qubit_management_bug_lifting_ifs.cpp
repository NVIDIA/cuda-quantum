/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>

// RUN: nvq++ --target opt-test --target-option dep-analysis,qpp %s -o %t && %t
// XFAIL: *

struct run_test {
  __qpu__ auto operator()() {
    cudaq::qubit q;

    bool b = false;

    // This test was meant to highlight an issue with lifting ifs, where
    // the equivalence check ignores the body of the if. Ifs probably just
    // shouldn't be lifted at all.

    // Suggested fix: Add the following to tryLiftingBefore/After
    // if (then_use->isContainer())
    //   return false;
    if (true) {
      if (true) {
        x(q);
        b = true;
      }
    } else {
      if (true) {
        y(q);
        b = true;
      }
    }

    return b;
  }
};

int main() {
  bool result = run_test{}();
  printf("Result = %b\n", result);
  return 0;
}
