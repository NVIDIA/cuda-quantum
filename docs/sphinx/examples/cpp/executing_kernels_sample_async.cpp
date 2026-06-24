/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin `SampleAsync`]
#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithms/draw.h>

// Define a quantum kernel function.
__qpu__ void kernel(int qubit_count) {
  cudaq::qvector qvector(qubit_count);
  // 2-qubit GHZ state.
  h(qvector[0]);
  for (auto qubit : cudaq::range(qubit_count - 1)) {
    x<cudaq::ctrl>(qvector[qubit], qvector[qubit + 1]);
  }
  // If we do not specify measurements, all qubits are measured in
  // the Z-basis by default or we can manually specify it also
  mz(qvector);
}

int main() {
  int qubit_count = 2;
  auto result = cudaq::sample_async(kernel, qubit_count);
  result.get().dump();
  return 0;
}
// [End `SampleAsync`]
/* [Begin `SampleAsyncOutput`]
{ 00:510 11:490 }
 [End `SampleAsyncOutput`] */
