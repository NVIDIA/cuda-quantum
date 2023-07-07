/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

#include <numeric>

struct reflect_about_uniform {
  void operator()(cudaq::qspan<> q) __qpu__ {
    auto ctrl_qubits = q.front(q.size() - 1);
    auto &last_qubit = q.back();

    // Compute (U) Action (V) produces
    // U V U::Adjoint
    cudaq::compute_action(
        [&]() {
          h(q);
          x(q);
        },
        [&]() { z<cudaq::ctrl>(ctrl_qubits, last_qubit); });
  }
};

struct run_grover {
  template <typename CallableKernel>
  __qpu__ auto operator()(const int n_qubits, const int n_iterations,
                          CallableKernel &&oracle) {
    cudaq::qvector q(n_qubits);
    h(q);
    for (int i = 0; i < n_iterations; i++) {
      oracle(q);
      reflect_about_uniform{}(q);
    }
    mz(q);
  }
};

struct oracle {
  void operator()(cudaq::qvector<> &q) __qpu__ {
    cz(q[0], q[2]);
    cz(q[1], q[2]);
  }
};

CUDAQ_TEST(GroverTester, checkNISQ) {
  using namespace cudaq;
  auto counts = cudaq::sample(1000, run_grover{}, 3, 1, oracle{});
  counts.dump();
  int counter = 0;
  for (auto &[bits, count] : counts) {
    counter += count;
    // Note bits could be 110 or 011 depending on the backend
    EXPECT_TRUE(bits == "101" || bits == "110" || bits == "011");
  }
  EXPECT_EQ(counter, 1000);
}
