/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s

// Tests qvector initialization from a cudaq::state created in host code with a
// large number of qubits (19 qubits = 524288 elements).

#include <cudaq.h>
#include <iostream>

__qpu__ void test(cudaq::state *inState) {
  cudaq::qvector q(inState);
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
  // 19 qubits = 2^19 = 524288 elements
  std::vector<cudaq::complex> vec(1ULL << 19, 0.);
  vec[0] = M_SQRT1_2;
  vec[1] = M_SQRT1_2;
  auto state = cudaq::state::from_data(vec);
  auto counts = cudaq::sample(test, &state);
  std::cout << "Large state test\n";
  printCounts(counts);

  // CHECK-LABEL: Large state test
  // CHECK: 0000000000000000000
  // CHECK: 1000000000000000000
  return 0;
}
