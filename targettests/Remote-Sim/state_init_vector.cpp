/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu %s -o %t  && %t | FileCheck %s
// clang-format on

#include <cudaq.h>

__qpu__ void test(std::vector<cudaq::complex> inState) {
  cudaq::qvector q = inState;
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
  std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
  auto counts = cudaq::sample(test, vec);
  printCounts(counts);

  printf("size %zu\n", counts.size());
  return !(counts.size() == 2);
}

// CHECK: 00
// CHECK: 10