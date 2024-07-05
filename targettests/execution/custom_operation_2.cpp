/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>

using namespace std::complex_literals;

CUDAQ_REGISTER_OPERATION(
    my_u3, 1, 3,
    {std::cos(parameters[0] / 2.),
     -std::exp(1i *parameters[2]) * std::sin(parameters[0] / 2.),
     std::exp(1i *parameters[1]) * std::sin(parameters[0] / 2.),
     std::exp(1i *(parameters[2] + parameters[1])) *
         std::cos(parameters[0] / 2.)})

__qpu__ void bell_pair() {
  cudaq::qvector qubits(2);
  my_u3(M_PI_2, 0., M_PI, qubits[0]);
  my_u3<cudaq::ctrl>(M_PI, M_PI, M_PI_2, qubits[0], qubits[1]);
}

int main() {
  auto counts = cudaq::sample(bell_pair);
  for (auto &[bits, count] : counts) {
    printf("%s\n", bits.data());
  }
}

// CHECK: 11
// CHECK: 00
