/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target anyon                    --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target infleqtion               --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target ionq                     --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target iqm --iqm-machine Apollo --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target oqc                      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// RUN: if $braket_avail; then nvq++ %cpp_std --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>

CUDAQ_REGISTER_OPERATION(custom_s, 1, 0, {1, 0, 0, std::complex<double>{0.0, 1.0}})

CUDAQ_REGISTER_OPERATION(custom_s_adj, 1, 0, {1, 0, 0, std::complex<double>{0.0, -1.0}})

__qpu__ void kernel() {
  cudaq::qubit q;
  h(q);
  custom_s<cudaq::adj>(q);
  custom_s_adj(q);
  h(q);
  mz(q);
}

int main() {
  auto counts = cudaq::sample(kernel);
  for (auto &[bits, count] : counts) {
    printf("%s\n", bits.data());
  }
}

// CHECK: 1
