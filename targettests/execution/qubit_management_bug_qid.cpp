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

    // q will be duplicated in the then and else branches,
    // but then mapped to two different qubits.
    // This test ensures that this case is handled properly,
    // with a fresh qid being generated for q when it is split.
    if (true) {
      x(p);
      y(q);
      x<cudaq::ctrl>(q,r);
    } else {
      y(q);
      x(r);
      x<cudaq::ctrl>(q,p);
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
