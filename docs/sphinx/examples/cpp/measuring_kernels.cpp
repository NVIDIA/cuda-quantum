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

// [Begin Run0]
__qpu__ auto kernel2() {
  cudaq::qvector q(2);
  h(q[0]);
  auto b0 = mz(q[0]);
  cudaq::reset(q[0]);
  x(q[0]);

  if (b0) {
    h(q[1]);
  }
  return mz(q);
}

int main() {
  auto results = cudaq::run(1000, kernel2);
  // Count occurrences of each bitstring
  std::map<std::string, std::size_t> bitstring_counts;
  for (const auto &result : results) {
    std::string bits = std::to_string(result[0]) + std::to_string(result[1]);
    bitstring_counts[bits]++;
  }

  printf("Bitstring counts:\n{\n");
  for (const auto &[bits, count] : bitstring_counts) {
    printf("  %s: %zu\n", bits.c_str(), count);
  }
  printf("}\n");

  return 0;
}
// [End Run0]

/* [Begin Run1]
Bitstring counts:
{
  10: 771
  11: 229
}
 [End Run1] */
