/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// clang-format off
// RUN: nvq++ %s -o %t --target oqc --emulate && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qvector q(3);
  x(q[0]);
  x(q[1]);
  x<cudaq::ctrl>(q[0], q[1]);
  x<cudaq::ctrl>(q[0], q[2]); // requires a swap(q0,q1)
  auto result = mz(q);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  result.dump();

  // If the swap is working correctly, this will show "101". If it is working
  // incorrectly, it may show something like "011".
  std::cout << "most_probable \"" << result.most_probable() << "\"\n";

  return 0;
}

// CHECK-DAG: __global__ : { 101:1000 }
// CHECK-DAG: result%0 : { 1:1000 }
// CHECK-DAG: result%1 : { 0:1000 }
// CHECK-DAG: result%2 : { 1:1000 }
// CHECK-DAG: most_probable "101"
