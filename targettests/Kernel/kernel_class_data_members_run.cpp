/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>

// The entry point that `cudaq::run` calls does not have the object of a kernel
// class, so it cannot read the data members. A kernel class with data members
// can be sampled but not run. Running it must be an error, and not a run with
// the wrong values.

struct WithMembers {
  int n = 4;
  int operator()() __qpu__ { return n + 1; }
};

struct WithoutMembers {
  int operator()() __qpu__ { return 7; }
};

int main() {
  auto ok = cudaq::run(2, WithoutMembers{});
  std::cout << "without members: " << ok[0] << '\n';
  try {
    auto bad = cudaq::run(2, WithMembers{});
    std::cout << "ran with " << bad[0] << '\n';
  } catch (const std::exception &e) {
    std::cout << "error: " << e.what() << '\n';
  }
  return 0;
}

// CHECK: without members: 7
// CHECK: error: {{.*}}cannot be called by cudaq::run
// CHECK-NOT: ran with
