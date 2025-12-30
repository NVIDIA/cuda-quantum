/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/


// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

// Basic check that the choice function works to determine the path taken
struct mykernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit p;

    h(q);

    auto m1 = mz(q);

    if (m1)
      x(p);

    mz(p);
  }
};

int main() {
  auto kernel = mykernel{};
  auto gateCountsTrue = cudaq::estimate_resources([](){ return true; }, kernel);
  auto gateCountsFalse = cudaq::estimate_resources([](){ return false; }, kernel);

  printf("True path\n");
  gateCountsTrue.dump();
  // CHECK-LABEL: True path
  // CHECK-DAG: h :  1
  // CHECK-DAG: x :  1

  printf("False path\n");
  gateCountsFalse.dump();
  // CHECK-LABEL: False path
  // CHECK-DAG: h :  1

  cudaq::set_random_seed(0);
  printf("True default path\n");
  auto default1Counts = cudaq::estimate_resources(kernel);
  // CHECK-LABEL: True default path

  cudaq::set_random_seed(1);
  printf("False default path\n");
  auto default2Counts = cudaq::estimate_resources(kernel);
  // CHECK-LABEL: False default path

  // Unfortunately, even setting the random seed isn't enough to guarantee
  // proper behavior, so handle either case here and hope for the best
  assert(default1Counts.count("x") == 1 || default2Counts.count("x") == 1);
  assert(default1Counts.count("x") == 0 || default2Counts.count("x") == 0);

  return 0;
}
