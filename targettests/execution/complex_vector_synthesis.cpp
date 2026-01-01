/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --enable-mlir %s -o %t  && %t | FileCheck %s
// TODO-FIX-KERNEL-EXEC
// RUN: nvq++ --target quantinuum --emulate -fkernel-exec-kind=2 %s -o %t  && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate                      %s -o %t  && %t | FileCheck %s
// clang-format on

#include <complex>
#include <cudaq.h>
#include <iostream>
#include <vector>

__qpu__ void test(std::vector<std::complex<double>> &v) {
  cudaq::qvector q(2);
  for (std::size_t i = 0; i < v.size(); i++) {
    cudaq::exp_pauli(v[i].real(), q, "XX");
  }
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
  std::vector<std::complex<double>> vec{std::complex<double>(10., 0.),
                                        std::complex<double>(20., 0.)};
  auto counts = cudaq::sample(test, vec);
  printCounts(counts);
  return 0;
}

// CHECK: 00
// CHECK: 11
