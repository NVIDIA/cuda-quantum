/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -v %s -o %basename_t.x --target quantinuum --emulate && ./%basename_t.x | FileCheck %s
// XFAIL: *

#include <cudaq.h>
#include <algorithm>
#include <iostream>

__qpu__ void reflect_uniform(cudaq::qreg<> &qubits) {
  h(qubits);
  x(qubits);
  z<cudaq::ctrl>(qubits[0], qubits[1], qubits[2], qubits[3]);
  x(qubits);
  h(qubits);
}

__qpu__ void oracle(cudaq::qreg<> &cs, cudaq::qubit &target) {
  x<cudaq::ctrl>(cs[0], !cs[1], !cs[2], cs[3], target);
  x<cudaq::ctrl>(!cs[0], cs[1], cs[2], !cs[3], target);
}

__qpu__ void grover() {
  cudaq::qreg qubits(4);
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
  mz(qubits);
};

int main() {
  auto result = cudaq::sample(1000, grover);
  // Didn't work:
  // std::vector<std::string> strings = result.sequential_data();
  std::vector<std::string> strings;
  for (auto &&[bits, count] : result) {
    strings.push_back(bits);
  }
  std::sort(strings.begin(), strings.end(), [&](auto& a, auto& b) {
    return result.count(a) > result.count(b);
  });
  std::cout << strings[0] << '\n';
  std::cout << strings[1] << '\n';
  return 0;
}

// CHECK-DAG: 1001
// CHECK-DAG: 0110