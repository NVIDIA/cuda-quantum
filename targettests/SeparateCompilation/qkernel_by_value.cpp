/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

#include "cudaq.h"
#include <iostream>

// The `qkernel` is passed by value, so it must be copyable.
__qpu__ void entry(cudaq::qkernel<void(cudaq::qvector<> &)> init) {
  cudaq::qvector q(2);
  init(q);
  mz(q);
}

int main() {
  auto l = [](cudaq::qvector<> &q) __qpu__ { x(q[0]); };
  cudaq::qkernel<void(cudaq::qvector<> &)> orig{l};

  // Copy construct, then copy assign; both copies must still launch.
  auto copy = orig;
  cudaq::qkernel<void(cudaq::qvector<> &)> assigned;
  assigned = orig;

  for (auto &k : {orig, copy, assigned}) {
    auto counts = cudaq::sample(entry, k);
    for (auto &[bits, n] : counts)
      std::cout << bits << ":" << n << "\n";
  }
  return 0;
}

// CHECK: 10:1000
// CHECK: 10:1000
// CHECK: 10:1000
