/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: nvq++ %s -o %t && %t | FileCheck %s
/// FIXME: The following fail with the error QIR verification error -
///        invalid instruction found:   %1 = alloca i32, align 4 (adaptive profile)
// SKIPPED: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// TODO-FIX-KERNEL-EXEC
// SKIPPED: nvq++ -fkernel-exec-kind=2 --target quantinuum --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ auto cccx_measure_cleanup() {
  cudaq::qvector qubits(4);
  // Initialize
  x(qubits[0]);
  x(qubits[1]);
  x(qubits[2]);

  // Compute AND-operation
  cudaq::qubit ancilla;
  h(ancilla);
  t(ancilla);
  x<cudaq::ctrl>(qubits[1], ancilla);
  t<cudaq::adj>(ancilla);
  x<cudaq::ctrl>(qubits[0], ancilla);
  t(ancilla);
  x<cudaq::ctrl>(qubits[1], ancilla);
  t<cudaq::adj>(ancilla);
  h(ancilla);
  s<cudaq::adj>(ancilla);

  // Compute output
  x<cudaq::ctrl>(qubits[2], ancilla, qubits[3]);

  // AND's measurement based cleanup.
  bool result = mx(ancilla);
  if (result)
    z<cudaq::ctrl>(qubits[0], qubits[1]);

  return mz(qubits);
}

int main() {
  auto results = cudaq::run(10, cccx_measure_cleanup);
  std::map<std::string, std::size_t> bitstring_counts;
  for (const auto &res : results) {
    std::string bits;
    for (auto b : res)
      bits += std::to_string(b);
    bitstring_counts[bits]++;
  }
  for (const auto &[bits, count] : bitstring_counts)
    std::cout << bits << ": " << count << "\n";
 
  return 0;
}

// CHECK: 1111: 10
