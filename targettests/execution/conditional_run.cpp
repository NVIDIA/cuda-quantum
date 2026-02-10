/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ --enable-mlir %s -o %t && %t

// The test here is the assert statement.

#include <cudaq.h>

struct kernel {
  auto operator()() __qpu__ {
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

    return mz(q[2]);
  }
};

int main() {

  // Can print the quake code, and get if it has if stmts
  printf("%s\n\n%d\n", cudaq::get_quake(kernel{}).data(),
         cudaq::kernelHasConditionalFeedback("kernel"));

  // Lower the number of shots
  int nShots = 100;
  auto &platform = cudaq::get_platform();

  auto results = cudaq::run(/*shots*/ nShots, kernel{});

  // Count the number of times we measured "1"
  std::size_t nOnes = std::ranges::count(results, true);

#ifndef SYNTAX_CHECK
  // Will fail if not equal to number of shots
  assert(nOnes == nShots && "Failure to teleport qubit in |1> state.");
#endif
  return 0;
}
