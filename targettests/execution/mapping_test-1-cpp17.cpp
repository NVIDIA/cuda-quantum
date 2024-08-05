/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++17
// RUN: nvq++ %cpp_std %s -o %t --target oqc --emulate && CUDAQ_DUMP_JIT_IR=1 %t |& FileCheck %s
// RUN: mkdir -p %t.dir && cp "%iqm_test_src_dir/Adonis.txt" "%t.dir/Adonis Variant.txt" && nvq++ %cpp_std %s -o %t --target iqm --iqm-machine Adonis --mapping-file "%t.dir/Adonis Variant.txt" --emulate && %t
// RUN: nvq++ %cpp_std --enable-mlir %s -o %t
// RUN: rm -rf %t.txt %t.dir

#include <cudaq.h>
#include <iostream>

__qpu__ void foo() {
  cudaq::qubit q0, q1, q2;
  x(q0);
  x(q1);
  cx(q0, q1);
  cx(q0, q2); // requires a swap(q0,q1)
  mz(q0);
  mz(q1);
  mz(q2);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  result.dump();

  // If the swap is working correctly, this will show "101". If it is working
  // incorrectly, it may show something like "011".
  std::cout << "most_probable \"" << result.most_probable() << "\"\n";

  return 0;
}

// CHECK: { 101:1000 }
// CHECK: most_probable "101"
