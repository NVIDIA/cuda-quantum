/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Spin]
#include <cudaq.h>

auto hamiltonian =
    2 * cudaq::spin::x(0) * cudaq::spin::y(1) * cudaq::spin::x(2) -
    3 * cudaq::spin::z(0) * cudaq::spin::z(1) * cudaq::spin::y(2);
// [End Spin]

// [Begin Pauli]
__qpu__ void kernel() {
  cudaq::qvector qvector(3);
  exp_pauli(0.432, qvector, "XYZ");
  exp_pauli(0.324, qvector, "IXX");
}

int main() {
  auto result = cudaq::sample(kernel);
  result.dump();
  return 0;
}
// [End Pauli]
