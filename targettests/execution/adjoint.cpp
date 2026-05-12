/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --enable-mlir %s -o %t && %t
// XFAIL: *
// TODO: this currently should fail due to ApplyOpSpecialization not handling
// loops with multiple arguments, remove XFAIL when this is resolved.
// See: https://github.com/NVIDIA/cuda-quantum/issues/3818

#include <cudaq.h>

// TODO: this segfaults in ApplyOpSpecialization (without the multi-argument-loop
// safety net) because in the IR `i` is expected to be the first loop argument,
// but `j` winds up being first.
auto __qpu__ one_loop(cudaq::qview<> qubits, unsigned long num_qubits) {
  unsigned long j = num_qubits;
  for (unsigned long i = 0; i < num_qubits; i++) {
    x<cudaq::ctrl>(qubits[i], qubits[j]);
    j--;
  }
}

// TODO: `j` is threaded through the loops after `memtoreg`, which is not
// handled in ApplyOpSpecialization so it winds up being threaded in the
// same initial order even after the loops are reversed, causing invalid IR.
auto __qpu__ two_loops(cudaq::qview<> qubits, unsigned long num_qubits) {
  unsigned long j = 0;
  for (unsigned long i = 0; i < 4; i++) {
    x<cudaq::ctrl>(qubits[i], qubits[j]);
    j++;
  }
  for (unsigned long i = 4; i < num_qubits; i++) {
    x<cudaq::ctrl>(qubits[i], qubits[j]);
  }
}

auto __qpu__ kernel() {
  cudaq::qarray<10> q;
  h(q[0]);
  // Compile-time sized array like std::array
  cudaq::adjoint(one_loop, q, 10);
  cudaq::adjoint(two_loops, q, 10);
}

int main() {
  auto counts = cudaq::sample(kernel);

  return 0;
}
