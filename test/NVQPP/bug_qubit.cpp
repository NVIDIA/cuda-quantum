/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This code is from Issue 251.

// RUN: nvq++ --enable-mlir -v %s --target quantinuum --quantinuum-url http://localhost:6245 && ( ./a.out > %t 2>&1 || ( cat %t | FileCheck %s ) )

// CHECK-NOT: could not trace offset value
// CHECK: terminate called

#include <cudaq.h>
#include <iostream>

struct ak2 {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(ak2{});
  std::cout << "Test: ";
  counts.dump();
  return 0;
}
