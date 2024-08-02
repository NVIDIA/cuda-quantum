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
#include <iostream>

__qpu__ void test(cudaq::state *state) {
  cudaq::qvector q(state);
}

// CHECK: size 2
int main() {
  std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
  auto state = cudaq::state::from_data(vec);
  // { 
  //   auto counts = cudaq::sample(test, &state);
  //   std::cout << cudaq::get_quake("test") << std::endl;
  //   counts.dump();
  //   printf("size %zu\n", counts.size());
  // }
  {
    auto [kernel, s] = cudaq::make_kernel<cudaq::state*>();
    auto qubits = kernel.qalloc(s);
    std::cout << kernel << std::endl;
    
    auto counts = cudaq::sample(kernel, &state);
    counts.dump();
    printf("size %zu\n", counts.size());
  }

  // {
  //   auto [kernel, s] = cudaq::make_kernel<std::vector<cudaq::complex>>();
  //   auto qubits = kernel.qalloc(s);
    
  //   auto counts = cudaq::sample(kernel, vec);
  //   counts.dump();
  //   printf("size %zu\n", counts.size());
  // }
  return 0;
}
