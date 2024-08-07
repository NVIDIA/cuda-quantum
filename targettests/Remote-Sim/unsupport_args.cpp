/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --target remote-mqpu %s 2>&1 | FileCheck %s
// clang-format on

#include <cudaq.h>

__qpu__ void reflect_about_uniform(cudaq::qview<> q) {
  auto ctrlQubits = q.front(q.size() - 1);
  auto &lastQubit = q.back();

  // Compute (U) Action (V) produces
  // U V U::Adjoint
  cudaq::compute_action(
      [&]() {
        h(q);
        x(q);
      },
      [&]() { z<cudaq::ctrl>(ctrlQubits, lastQubit); });
}

struct run_grover {
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, const int n_iterations,
                          CallableKernel &&oracle) {
    cudaq::qvector q(n_qubits);
    h(q);
    for (int i = 0; i < n_iterations; i++) {
      oracle(q);
      reflect_about_uniform(q);
    }
    mz(q);
  }
};

struct oracle {
  void operator()(cudaq::qvector<> &q) __qpu__ {
    z<cudaq::ctrl>(q[0], q[2]);
    z<cudaq::ctrl>(q[1], q[2]);
  }
};

// Oracle as a free function
__qpu__ void oracle_func(cudaq::qvector<> &q) {
  z<cudaq::ctrl>(q[0], q[2]);
  z<cudaq::ctrl>(q[1], q[2]);
}

int main() {
  {
    auto counts = cudaq::sample(run_grover{}, 3, 1, oracle{});
    // CHECK: Callable entry-point kernel arguments are not supported for the
    // CHECK: remote simulator platform in library mode. Please rewrite the
    // entry CHECK: point kernel or use MLIR mode.
    counts.dump();
  }
  {
    auto counts = cudaq::sample(run_grover{}, 3, 1, oracle_func);
    // CHECK NEXT: Callable entry-point kernel arguments are not supported for
    // CHECK NEXT: the remote simulator platform in library mode. Please
    // CHECK NEXT: rewrite the entry  point kernel or use MLIR mode.
    counts.dump();
  }
  return 0;
}
