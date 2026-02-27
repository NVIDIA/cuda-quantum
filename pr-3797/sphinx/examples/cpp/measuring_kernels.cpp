/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Docs]
#include <cudaq.h>
// [End Docs]

// [Begin Sample1]
__qpu__ void kernel0() {
  cudaq::qvector qubits(2);
  mz(qubits[0]);
}
// [End Sample1]

// [Begin Sample2]
__qpu__ void kernel1() {
  cudaq::qvector qubits_a(2);
  cudaq::qubit qubits_b;
  mz(qubits_a);
  mx(qubits_b);
}
// [End Sample2]

// [Begin Sample3]
__qpu__ void kernel2() {
  cudaq::qvector q(2);
  h(q[0]);
  auto b0 = mz(q[0]);
  cudaq::reset(q[0]);
  x(q[0]);

  if (b0) {
    h(q[1]);
  }
}

int main() {
  auto result = cudaq::sample(kernel2);
  result.dump();
  return 0;
}
// [End Sample3]

/* [Begin Sample4]
{
  __global__ : { 10:728 11:272 }
   b0 : { 0:505 1:495 }
}
 [End Sample4] */
