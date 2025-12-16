/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// clang-format off
// RUN: nvq++ %cpp_std --target anyon      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target ionq       --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_20.txt %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc        --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: if %qci_avail; then nvq++ %cpp_std --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <algorithm>
#include <cudaq.h>
#include <iostream>
#include <unordered_set>

__qpu__ void reflect_uniform(cudaq::qvector<> &qubits) {
  h(qubits);
  x(qubits);
  z<cudaq::ctrl>(qubits[0], qubits[1], qubits[2], qubits[3]);
  x(qubits);
  h(qubits);
}

__qpu__ void oracle(cudaq::qvector<> &cs, cudaq::qubit &target) {
  x<cudaq::ctrl>(cs[0], !cs[1], !cs[2], cs[3], target);
  x<cudaq::ctrl>(!cs[0], cs[1], cs[2], !cs[3], target);
}

__qpu__ void grover() {
  cudaq::qvector qubits(4);
  cudaq::qubit ancilla;

  // Initialization
  x(ancilla);
  h(ancilla);
  h(qubits); // uniform initialization

  // Don't work?:
  for (int i = 0; i < 2; ++i) {
    oracle(qubits, ancilla);
    reflect_uniform(qubits);
  }
};

int main() {
  auto all_counts = cudaq::sample(1000, grover);

  auto counts_map = all_counts.to_map();
  std::size_t total_qubits = counts_map.begin()->first.size();
  // We need to drop the compiler generated qubits, if any, which are the
  // beginning, and also drop the ancilla qubit which is the last one
  std::vector<std::size_t> indices;
  for (std::size_t i = total_qubits - 5; i < total_qubits - 1; i++)
    indices.push_back(i);
  auto result = all_counts.get_marginal(indices);
  result.dump();

#ifndef SYNTAX_CHECK
  std::vector<std::string> strings;
  for (auto &&[bits, count] : result) {
    strings.push_back(bits);
  }
  std::sort(strings.begin(), strings.end(), [&](auto &a, auto &b) {
    return result.count(a) > result.count(b);
  });
  std::cout << strings[0] << '\n';
  std::cout << strings[1] << '\n';

  std::unordered_set<std::string> most_probable{strings[0], strings[1]};
  assert(most_probable.count("1001") == 1);
  assert(most_probable.count("0110") == 1);
#endif

  return 0;
}

// CHECK-DAG: 1001
// CHECK-DAG: 0110
