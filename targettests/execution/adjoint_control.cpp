/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s
// clang-format on

// Verify that the controlled-adjoint of a kernel produces the correct inverse
// action. See: https://github.com/NVIDIA/cuda-quantum/issues/854

#include <cstdio>
#include <cudaq.h>

struct s_gate {
  void operator()(cudaq::qubit &q) __qpu__ { s(q); }
};

struct s_adj {
  void operator()(cudaq::qubit &q) __qpu__ { cudaq::adjoint(s_gate{}, q); }
};

// Wrapper that applies s_gate under a control qubit passed as a regular arg.
struct s_ctrl {
  void operator()(cudaq::qubit &ctrl, cudaq::qubit &q) __qpu__ {
    cudaq::control(s_gate{}, {ctrl}, q);
  }
};

// S gate is used because S != S_dagger, making an incorrect adjoint detectable:
// S_dagger*S|+> = |+> (correct, q -> 0), but S*S|+> = Z|+> = |-> (wrong, q ->
// 1).

// Approach 1: control(adj(S)) -- wrap the adjoint in a struct, then control it.
// Circuit:
//   ctrl = |1>, q = |+>
//   S(q)           -> q = S|+>
//   ctrl(S_dag)(q) (with ctrl = |1>) -> q = S_dag*S|+> = |+>
//   H(q)           -> q = |0>
// Expected bitstring: ctrl=1, q=0 -> "10"
struct ctrl_adj_s {
  void operator()() __qpu__ {
    cudaq::qubit ctrl, q;
    x(ctrl);
    h(q);
    s(q);
    cudaq::control(s_adj{}, {ctrl}, q);
    h(q);
  }
};

// Approach 2: adj(control(S)) -- wrap the controlled form in a struct, then
// adjoint it.
// Same expected result as approach 1 since adj(ctrl(U)) = ctrl(adj(U)).
struct adj_ctrl_s {
  void operator()() __qpu__ {
    cudaq::qubit ctrl, q;
    x(ctrl);
    h(q);
    s(q);
    cudaq::adjoint(s_ctrl{}, ctrl, q);
    h(q);
  }
};

int main() {
  auto counts1 = cudaq::sample(ctrl_adj_s{});
  for (auto &[bits, count] : counts1)
    printf("%s\n", bits.data());
  auto counts2 = cudaq::sample(adj_ctrl_s{});
  for (auto &[bits, count] : counts2)
    printf("%s\n", bits.data());
  return 0;
}

// CHECK: 10
// CHECK: 10
