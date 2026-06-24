/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Spin]
#include <cudaq.h>
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
