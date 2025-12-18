/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// clang-format on

#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

struct mykernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;

    x(q);

    auto m1 = mz(q);
    if (m1)
      x(q);
  }
};

int main() {
  auto kernel = mykernel{};
  std::function<bool()> choice = [&]() {
    auto counts1 = cudaq::sample(5, kernel);
    counts1.dump();
    return true;
  };
  auto exception_thrown = false;
  try {
    auto gateCounts = cudaq::estimate_resources(choice, kernel);
    gateCounts.dump();
  } catch (...) {
    exception_thrown = true;
  }
  if (exception_thrown)
    printf("success\n");
  else
    printf("FAILED!\n");

  return 0;
}

// CHECK: success
