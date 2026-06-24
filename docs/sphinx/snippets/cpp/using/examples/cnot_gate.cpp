/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Docs]
#include <cudaq.h>
#include <iostream>

__qpu__ void kernel() {
  // 2 qubits both initialized to the ground/ zero state.
  cudaq::qvector qubits(2);
  x(qubits[0]);
  // Controlled-not gate operation.
  cx(qubits[0], qubits[1]);
  mz(qubits[0]);
  mz(qubits[1]);
}

int main() {
  auto result = cudaq::sample(kernel);
  result.dump();
}
// [End Docs]
