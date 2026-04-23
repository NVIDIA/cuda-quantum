/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ --enable-mlir --target remote-mqpu %s -o %t  && %t | FileCheck %s
// TODO-FIX-KERNEL-EXEC
// RUN: nvq++ --enable-mlir --target remote-mqpu -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

// This test allocates a constant double array for qvector state initialization.
__qpu__ void test_large_double_constant_array() {
  std::vector<double> vec(1ULL << 10);  // 1024 elements = 10 qubits
  vec[0] = M_SQRT1_2 / vec.size();
  vec[1] = M_SQRT1_2 / vec.size();
  for (std::size_t i = 2; i < vec.size(); i++) {
    vec[i] = 0;
  }
  cudaq::qvector v(vec);
}

void printCounts(cudaq::sample_result &result) {
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
  auto counts = cudaq::sample(test_large_double_constant_array);
  std::cout << "Large array test\n";
  printCounts(counts);

  // CHECK-LABEL: Large array test
  // CHECK: 0000000000
  // CHECK: 1000000000
}

