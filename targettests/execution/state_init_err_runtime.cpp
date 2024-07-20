/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Note: change |& to 2>&1| if running in bash
// RUN: nvq++ %cpp_std %s -o %t --target quantinuum --emulate && %t |& FileCheck %s
// Note: change |& to 2>&1| if running in bash
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-url localhost:9999 %s -o %t && %t |& FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void test(cudaq::state *inState) {
  cudaq::qvector q(inState);
}

int main() {
  std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
  auto state = cudaq::state::from_data(vec);
  { 
    auto counts = cudaq::sample(test, &state);
    counts.dump();
    printf("size %zu\n", counts.size());
  }
  return 0;
}

// CHECK: error: 'func.func' op synthesis: unsupported argument type for remote devices and simulators: state*
