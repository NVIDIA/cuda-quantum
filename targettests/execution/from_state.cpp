/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include "cudaq/builder/kernels.h"
#include <iostream>

__qpu__ void test(cudaq::state *inState) {
  cudaq::qvector q(inState);
}

// CHECK: size 2

int main() {
  std::vector<std::complex<float>> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
  auto state = cudaq::state::from_data(vec);
  auto counts = cudaq::sample(test, &state);
  counts.dump();

  printf("size %zu\n", counts.size());
  return !(counts.size() == 2);
}
