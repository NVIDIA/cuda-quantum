/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Simulators
// RUN: nvq++ %cpp_std --enable-mlir                                                    %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu          -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
//
// Quantum emulators
// RUN: nvq++ %cpp_std --target quantinuum               --emulate -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target ionq                     --emulate -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// 2 different IQM machines for 2 different topologies
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Adonis --emulate -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Apollo --emulate -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target anion                    --emulate -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void test() {
    cudaq::qvector q(2);
    cudaq::exp_pauli(1.0, q, "XX");
}

__qpu__ void test_param(cudaq::pauli_word w) {
    cudaq::qvector q(2);
    cudaq::exp_pauli(1.0, q, w);
}

void printCounts(cudaq::sample_result& result) {
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
  auto counts = cudaq::sample(test);
  printCounts(counts);

  counts = cudaq::sample(test_param, cudaq::pauli_word{"XY"});
  printCounts(counts);
  return 0;
}

// CHECK: 00
// CHECK: 11

// CHECK: 00
// CHECK: 11
