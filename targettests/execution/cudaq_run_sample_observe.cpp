/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <iostream>

__qpu__ int run_kernel(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v);
  z(v);
  for (int i = 0; i < count; i++) {
    bool w = mz(v[i]);
    result |= ((unsigned)w) << (count - 1 - i);
  }
  return result;
}

__qpu__ void sample_kernel() {
  cudaq::qvector v(2);
  h(v[0]);
  cx(v[0], v[1]);
}

__qpu__ void observe_kernel() {
  cudaq::qubit qubit;
  h(qubit);
}

int main() {
  const auto results = cudaq::run(5, run_kernel, 4);
  if (results.size() != 5)
    printf("FAILED! Expected 5 shots. Got %lu\n", results.size());
  else
    printf("success!\n");

  const auto direct_call = run_kernel(2);
  printf("direct call result: %d\n", direct_call);

  cudaq::sample(20, sample_kernel).dump();

  auto spin_operator = cudaq::spin_op::z(0);
  auto result = cudaq::observe(1000, observe_kernel, spin_operator);
  std::cout << "<kernel | spin_operator | kernel> = " << result.expectation()
            << "\n";
  return 0;
}

// CHECK: success!
// CHECK: direct call result: {{[0-9]+}}
// CHECK: { 00:{{[0-9]+}} 11:{{[0-9]+}} }
// CHECK: <kernel | spin_operator | kernel> =
