/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Regression test for https://github.com/NVIDIA/cuda-quantum/issues/4601
// (`std::vector<cudaq::measure_handle>::operator=` was unimplemented in the
// bridge, causing `cudaq-quake` to abort.  After the fix, assignment must
// behave as a deep copy so that the LHS doesn't dangle when the RHS's
// backing buffer goes out of scope.)

// Compile and execute; verifies the runtime produces the correct bitstrings.
// RUN: nvq++ %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>
#include <vector>

// `outer` lives across an inner block in which we measure a fresh register
// and assign that result into `outer`.  After the inner block, the inner
// register and its measurement-handle buffer are gone; the deep-copy
// semantics required by this PR are what make `outer` still observe the
// inner measurement's bits.
__qpu__ std::vector<bool> outer_outlives_inner() {
  cudaq::qvector outerQ(2);
  x(outerQ[0]);            // |10>
  auto outer = mz(outerQ); // outer == [1, 0]
  {
    cudaq::qvector innerQ(2);
    x(innerQ[1]);            // |01>
    auto inner = mz(innerQ); // inner == [0, 1]
    outer = inner;           // requires deep-copy semantics
  }                          // inner buffer/qubits go out of scope here
  return cudaq::to_bools(outer);
}

// Loop-carry pattern adapted from the issue's "rep_code_vectorized" example.
// Each iteration creates a fresh `curr` whose buffer would be reused or
// freed between iterations; without a deep copy, `prev` would silently
// observe the next iteration's measurement rather than the carried one.
__qpu__ std::vector<bool> loop_carry_last() {
  const int nRounds = 3;
  cudaq::qvector qv(2);
  auto prev = mz(qv);
  for (int r = 1; r < nRounds; r++) {
    for (std::size_t i = 0; i < qv.size(); i++)
      reset(qv[i]);
    if (r == nRounds - 1) {
      x(qv[0]); // last round: deterministically |10>
    }
    auto curr = mz(qv);
    prev = curr; // deep-copy carry
  }
  return cudaq::to_bools(prev);
}

int main() {
  {
    auto bits = outer_outlives_inner();
    std::cout << "outer:";
    for (bool b : bits)
      std::cout << b;
    std::cout << "\n";
  }
  {
    auto bits = loop_carry_last();
    std::cout << "carry:";
    for (bool b : bits)
      std::cout << b;
    std::cout << "\n";
  }
}

// CHECK: outer:01
// CHECK: carry:10
