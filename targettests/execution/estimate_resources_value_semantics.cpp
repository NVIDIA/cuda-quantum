/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 ******************************************************************************/

// The `value-semantics-test` target leaves the module in value (linear) form,
// so resource counting sees quantum operations that thread wires rather than
// operate on references.

// clang-format off
// RUN: nvq++ --target value-semantics-test %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

struct bell_kernel {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    mz(q);
  }
};

int main() {
  auto gateCounts = cudaq::estimate_resources(bell_kernel{});

  gateCounts.dump();
  // The `h` and the `cx` are pre-counted and erased from the IR; the two `mz`
  // are counted by the simulator at run time. The control count and the depth
  // are only correct if the wire operands resolve back to their qubits.
  // clang-format off
  // CHECK: Total # of gates: 4, total # of qubits: 2, circuit depth: 3, multi-Q gate count: 1, multi-Q depth: 1
  // CHECK-DAG: cx :  1
  // CHECK-DAG: h :  1
  // CHECK-DAG: mz :  2
  // clang-format on

  return 0;
}
