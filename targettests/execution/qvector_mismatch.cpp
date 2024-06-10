/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>

__qpu__ void test(std::vector<double> inState) {
  cudaq::qvector q = inState;
}

int main() {
  std::vector<double> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
  auto counts = cudaq::sample(test, vec);
  counts.dump();

  printf("size %zu\n", counts.size());
  return 0;
}

// CHECK: size {{[0-9]+}}
