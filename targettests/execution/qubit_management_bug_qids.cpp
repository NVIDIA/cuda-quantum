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
    cudaq::qubit q,p,r;

    h(r);

    // This fails with an expected error:
    // q will be duplicated in the then and else branches,
    // but then mapped to two different qubits, so both
    // versions will get lifted to the parent context,
    // so there will be two qs with the same qid but different
    // qubits. The error seen is that the block argument index
    // gets confused, which makes sense since there's probably
    // all sorts of weirdness going on.

    // Suggested fix steps:
    // 1. Pass a "unique id counter" to contractAllocsPass
    //    (use, say, DependencyAnalysisEngine::getNumVirtualAllocs()),
    // 2. In IfDependencyNode::lowerAlloc, use the "unique id counter"
    //    to make a new QID for the alloc/de-alloc copy.
    // 3. Make sure relevant metadata is updated properly
    // 4. That should hopefully be enough, if not, the suggested fixes
    //    to replaceLeaf/replaceRoot in the code may be necessary.
    // Corollary fix: Override `updateQID` in IfDependencyNode to also work inside the then/else blocks.
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
