/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++17
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && %t
// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t

// The test here is the assert statement.

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    cudaq::qarray<3> q;
    // Initial state prep
    x(q[0]);

    // create bell pair
    h(q[1]);
    cx(q[1], q[2]);

    cx(q[0], q[1]);
    h(q[0]);

    auto b0 = mz(q[0]);
    auto b1 = mz(q[1]);

    if (b1)
      x(q[2]);
    if (b0)
      z(q[2]);

    mz(q[2]);
  }
};

int main() {

  // Can print the quake code, and get if it has if stmts
  printf("%s\n\n%d\n", cudaq::get_quake(kernel{}).data(),
         cudaq::kernelHasConditionalFeedback("kernel"));

  // Lower the number of shots
  int nShots = 100;
  auto &platform = cudaq::get_platform();

  // Sample
  auto counts = cudaq::sample(nShots, kernel{});
  counts.dump();

  // Get the marginal counts on the 2nd qubit
  auto resultsOnZero = counts.get_marginal({0});
  resultsOnZero.dump();

  // Count the "1"
  auto nOnes = resultsOnZero.count("1");

#ifndef SYNTAX_CHECK
  // Will fail if not equal to number of shots
  assert(nOnes == nShots && "Failure to teleport qubit in |1> state.");
#endif
  return 0;
}
