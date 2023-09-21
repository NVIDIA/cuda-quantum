/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -v %s -o %basename_t.x --target quantinuum --emulate && ./%basename_t.x | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ void load_value(unsigned value) {
  cudaq::qreg qubits(4);
  for (std::size_t i = 0; i < 4; ++i) {
    // Doesn't work, even with: `if (value)`
    if (value & (1 << i))
      x(qubits[3 - i]);
  }

  mz(qubits);
}

int main() {
  for (auto i = 0; i < 16; ++i) {
    auto result = cudaq::sample(1000, load_value, i);

#ifndef SYNTAX_CHECK
    std::cout << result.most_probable() << '\n';
    assert(i == std::stoi(result.most_probable(), nullptr, 2));
#endif
  }
  return 0;
}

// CHECK: 0000
// CHECK-NEXT: 0001
// CHECK-NEXT: 0010
// CHECK-NEXT: 0011
// CHECK-NEXT: 0100
// CHECK-NEXT: 0101
// CHECK-NEXT: 0110
// CHECK-NEXT: 0111
// CHECK-NEXT: 1000
// CHECK-NEXT: 1001
// CHECK-NEXT: 1010
// CHECK-NEXT: 1011
// CHECK-NEXT: 1100
// CHECK-NEXT: 1101
// CHECK-NEXT: 1110
// CHECK-NEXT: 1111
