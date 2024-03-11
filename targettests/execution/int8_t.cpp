/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>

struct variable_qreg {
  __qpu__ void operator()(std::uint8_t value) {
    cudaq::qvector qubits(value);

    mz(qubits);
  }
};

int main() {
  for (auto i = 1; i < 5; ++i) {
    auto result = cudaq::sample(1000, variable_qreg{}, i);

#ifndef SYNTAX_CHECK
    std::cout << result.most_probable() << '\n';
    assert(std::string(i, '0') == result.most_probable());
#endif
  }

  return 0;
}

// CHECK: 0
// CHECK: 00
// CHECK: 000
// CHECK: 0000
