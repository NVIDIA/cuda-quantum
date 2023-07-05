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
#include <iostream>

__qpu__ void init_state() {
  double theta = 2. * std::acos(1. / std::sqrt(3));
  cudaq::qreg qubits(2);
  ry(theta, qubits[0]);
  h<cudaq::ctrl>(qubits[0], qubits[1]);
  x(qubits[1]);
};

int main() {
  auto result = cudaq::sample(1000, init_state);
  for (auto &&[bits, counts] : result) {
    std::cout << bits << '\n';
  }
  return 0;
}

//CHECK-NOT: 00
