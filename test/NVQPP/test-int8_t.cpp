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

struct variable_qreg {
  __qpu__ void operator()(std::uint8_t value) {
    cudaq::qreg qubits(value);
    mz(qubits);
  }
};

int main() {
  for (auto i = 1; i < 5; ++i) {
    auto result = cudaq::sample(1000, variable_qreg{}, i);
    std::cout << result.most_probable() << '\n';
  }
  return 0;
}