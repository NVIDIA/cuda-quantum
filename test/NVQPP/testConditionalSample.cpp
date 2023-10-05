/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o out_testifstmts.x && ./out_testifstmts.x

// The test here is the assert statement. 

#include <cudaq.h>

struct kernel {
  void operator()() __qpu__ {
    cudaq::qreg<3> q;
    // Initial state prep
    x(q[0]);

    // create bell pair
    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);

    x<cudaq::ctrl>(q[0], q[1]);
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

  // Will fail if not equal to number of shots
  assert(nOnes == nShots && "Failure to teleport qubit in |1> state.");
}
