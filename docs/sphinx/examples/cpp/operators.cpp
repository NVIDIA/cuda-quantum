/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Spin]
#include <cudaq.h>
#include <iostream>

void build_operator() {
  auto op =
      2.0 * cudaq::spin_op::x(0) * cudaq::spin_op::y(1) * cudaq::spin_op::x(2) -
      3.0 * cudaq::spin_op::z(0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2);
  std::cout << op.to_string() << '\n';
}
// [End Spin]

// [Begin Pauli]
__qpu__ void kernel() {
  cudaq::qvector qvector(3);
  exp_pauli(0.432, qvector, "ZYX");
  exp_pauli(0.324, qvector, "ZXX");
}

int main() {
  auto result = cudaq::sample(kernel);
  result.dump();
  return 0;
}
// [End Pauli]
