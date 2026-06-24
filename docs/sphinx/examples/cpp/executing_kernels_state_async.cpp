/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin `GetStateAsync`]
#include <cstdio>
#include <cudaq.h>

// Define a quantum kernel for Bell state preparation
__qpu__ void bell_state() {
  cudaq::qvector qubits(2);
  h(qubits[0]);
  x<cudaq::ctrl>(qubits[0], qubits[1]);
}

int main() {
  // Get state asynchronously
  auto state_future = cudaq::get_state_async(bell_state);
  // Do other work while waiting for state computation
  // Get and print the state when ready
  auto state = state_future.get();
  state.dump();
  return 0;
}
// [End `GetStateAsync`]
/* [Begin `GetStateAsyncOutput`]
(0.707107,0)
(0,0)
(0,0)
(0.707107,0)
[End `GetStateAsyncOutput`] */
