/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s

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
