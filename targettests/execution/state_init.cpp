/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --enable-mlir %s -o %t  && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void test(cudaq::state *inState) {
  cudaq::qvector q(inState);
}

void printCounts(cudaq::sample_result& result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << '\n';
  }
}

int main() {
  std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0.};
  auto state = cudaq::state::from_data(vec);
  {
    auto counts = cudaq::sample(test, &state);
    printCounts(counts);
  }
  return 0;
}

// CHECK: 00
// CHECK: 10
