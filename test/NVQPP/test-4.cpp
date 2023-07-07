/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -v %s -o %basename_t.x --target quantinuum --emulate && ./%basename_t.x | FileCheck %s
// RUN: nvq++ -v %s -o %basename_t.x --target ionq --emulate && ./%basename_t.x | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void bar(cudaq::qspan<> qubits) {
  auto controls = qubits.front(qubits.size() - 1);
  auto &target = qubits.back();
  x<cudaq::ctrl>(controls, target);
}

__qpu__ void foo() {
  cudaq::qreg qubits(4);
  x(qubits);
  bar(qubits);
  mz(qubits);
}

int main() {
  auto result = cudaq::sample(1000, foo);
  std::cout << result.size() << '\n';
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

//CHECK: 1
//CHECK-NEXT: 1110
