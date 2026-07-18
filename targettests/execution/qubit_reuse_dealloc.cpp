/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Verify that physical qubit slots are reused across successive ancilla
// allocations within a single cudaq::run kernel execution.
//
// Each call to `parity_check` allocates 4 ancilla qubits, entangles them with
// 4 data qubits, measures one, resets all four (signaling reuse), and returns.
// `outer` calls it 9 times → 4 (data) + 9×4 (ancilla) = 40 logical qubits
// total, but only 4+4 = 8 live at any one time.
//
// Without qubit reuse a 40-qubit state vector (2^40 ≈ 1 TB) would OOM.
// With reuse the peak-8 state vector (256 entries) finishes instantly.
// A successful run therefore proves reuse is working.

// clang-format off
// RUN: nvq++ %s -o %t && %t
// clang-format on

#include <cassert>
#include <cstdio>
#include <cudaq.h>

struct parity_check {
  int operator()(cudaq::qview<> data) __qpu__ {
    cudaq::qvector anc(4);
    x<cudaq::ctrl>(data[0], anc[0]);
    x<cudaq::ctrl>(data[1], anc[1]);
    x<cudaq::ctrl>(data[2], anc[2]);
    x<cudaq::ctrl>(data[3], anc[3]);
    bool b = mz(anc[0]);
    cudaq::reset(anc[0]);
    cudaq::reset(anc[1]);
    cudaq::reset(anc[2]);
    cudaq::reset(anc[3]);
    return b ? 1 : 0;
  }
};

// 40 logical qubits total, 8 peak concurrent — only works if reuse is active.
struct outer {
  int operator()() __qpu__ {
    cudaq::qvector data(4);
    x(data[0]); // data[0]=1, rest 0 → parity_check returns 1 each round
    int acc = 0;
    for (int i = 0; i < 9; ++i)
      acc += parity_check{}(data);
    return acc;
  }
};

int main() {
  auto results = cudaq::run(1, outer{});
  // data[0]=1 so mz(anc[0]) = 1 in every round → 9 rounds → acc == 9
  assert(results[0] == 9);
  printf("qubit reuse OK: 40 logical qubits, result=%d\n", results[0]);
  return 0;
}
