/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target ionq                     --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s
// RUN: nvq++ --target iqm --iqm-machine Adonis --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s
// RUN: nvq++ --target oqc                      --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s
// RUN: nvq++ --target quantinuum               --emulate %s -o %basename_t.x && ./%basename_t.x | FileCheck %s

#include <cudaq.h>
#include <iostream>

struct variable_qreg {
  __qpu__ void operator()(std::uint8_t value) {
    cudaq::qreg qubits(value);

    auto result = mz(qubits);
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
