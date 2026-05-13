/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Regression test for https://github.com/NVIDIA/cuda-quantum/issues/4333
// Verify that mixing mz/mx/my in a sampled kernel produces bitstrings that
// include all measured qubits.

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s

#include <cassert>
#include <cudaq.h>
#include <cstdio>

// Kernel that measures 7 out of 9 qubits using mixed bases.
// Deterministic bits: q0=0(mz), q1=1(mz), q4=0(mz), q6=1(mz).
// mx on |+> (q2) and |-> (q5) are deterministic: q2=0, q5=1.
// my on |0> (q3) is non-deterministic.
__qpu__ void mixed_basis_kernel() {
  cudaq::qvector q(9);

  x(q[1]);
  h(q[2]); // |+> so mx deterministically gives 0
  x(q[5]);
  h(q[5]); // |-> so mx deterministically gives 1
  x(q[6]);

  mz(q[4]);
  mx(q[2]);
  my(q[3]);
  mz(q[0]);
  mx(q[5]);
  mz(q[6]);
  mz(q[1]);
}

int main() {
  auto counts = cudaq::sample(100, mixed_basis_kernel);

  // Must have at least one bitstring.
  assert(counts.size() > 0);

  for (auto &[bits, count] : counts) {
    // Every bitstring must be 7 bits (7 measured qubits).
    assert(bits.size() == 7);

    // Measurement execution order: q4, q2, q3, q0, q5, q6, q1.
    assert(bits[0] == '0'); // q4 mz -> 0
    assert(bits[1] == '0'); // q2 mx(|+>) -> 0
    assert(bits[3] == '0'); // q0 mz -> 0
    assert(bits[4] == '1'); // q5 mx(|->) -> 1
    assert(bits[5] == '1'); // q6 mz -> 1
    assert(bits[6] == '1'); // q1 mz -> 1

    // q3: my(|0>) is non-deterministic, just check valid bit.
    assert(bits[2] == '0' || bits[2] == '1');
  }

  printf("passed\n");
  return 0;
}

// CHECK: passed
