/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Exercise the `cudaq::mx` / `cudaq::my` range overloads added by the
// `measure_handle` proposal. In MLIR mode the bridge intercepts the
// `mx(qvec)` / `my(qvec)` calls by name and emits
// `quake.{mx,my} ... -> !cc.stdvec<!cc.measure_handle>`; the library-mode
// templates in `runtime/cudaq/qis/qubit_qis.h` are exercised by the
// `--library-mode` RUN line below. Both modes must agree on the outcome
// for the deterministic basis preparations used here.

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s

#include <cassert>
#include <cstdio>
#include <cudaq.h>

// `mx(qvec)` on a register of |+> qubits must yield all-zero bits;
// `my(qvec)` on a register of |i> qubits (`h` then `s`) must yield
// all-zero bits as well. Both are deterministic so a single shot is
// sufficient.
__qpu__ void mx_range_kernel() {
  cudaq::qvector q(3);
  for (std::size_t i = 0; i < 3; i++)
    h(q[i]);
  mx(q);
}

__qpu__ void my_range_kernel() {
  cudaq::qvector q(3);
  for (std::size_t i = 0; i < 3; i++) {
    h(q[i]);
    s(q[i]);
  }
  my(q);
}

int main() {
  auto mx_counts = cudaq::sample(64, mx_range_kernel);
  for (auto &[bits, count] : mx_counts) {
    assert(bits.size() == 3);
    for (char b : bits)
      assert(b == '0');
  }

  auto my_counts = cudaq::sample(64, my_range_kernel);
  for (auto &[bits, count] : my_counts) {
    assert(bits.size() == 3);
    for (char b : bits)
      assert(b == '0');
  }

  printf("passed\n");
  return 0;
}

// CHECK: passed
