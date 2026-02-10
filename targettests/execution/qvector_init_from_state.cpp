/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Simulators
// RUN: nvq++ --enable-mlir  %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

struct test_init_state {
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    ry(M_PI / 2.0, q[0]);
  }
};

struct test_state_param {
  void operator()(cudaq::state *state) __qpu__ {
    cudaq::qvector q(state);
    x(q);
  }
};

void printCounts(cudaq::sample_result &result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << std::endl;
  }
}

int main() {
  std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0., 0., 0., 0., 0.};
  std::vector<cudaq::complex> vec1{0., 0., 0.,        0.,
                                   0., 0., M_SQRT1_2, M_SQRT1_2};
  auto state = cudaq::state::from_data(vec);
  auto state1 = cudaq::state::from_data(vec1);
  {
    std::cout << "Passing state created from data as argument (kernel mode)"
              << std::endl;
    auto counts = cudaq::sample(test_state_param{}, &state);
    printCounts(counts);

    counts = cudaq::sample(test_state_param{}, &state1);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Passing state created from data as argument (kernel mode)
  // CHECK: 011
  // CHECK: 111

  // CHECK: 000
  // CHECK: 100
  // clang-format on

  {
    std::cout << "Passing state from another kernel as argument (kernel mode)"
              << std::endl;
    auto state = cudaq::get_state(test_init_state{}, 2);
    auto counts = cudaq::sample(test_state_param{}, &state);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Passing state from another kernel as argument (kernel mode)
  // CHECK: 01
  // CHECK: 11
  // clang-format on

  {
    std::cout
        << "Passing large state from another kernel as argument (kernel mode)"
        << std::endl;
    // TODO: State larger than 5 qubits fails on iqm machines with Crystal_5
    // architecture
    // TODO: State larger than 8 qubits fails on oqc and anyon
    // Up to 14 bits works with quantinuum an ionq
    auto largeState = cudaq::get_state(test_init_state{}, 5);
    auto counts = cudaq::sample(test_state_param{}, &largeState);
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Passing large state from another kernel as argument (kernel mode)
  // CHECK: 01111
  // CHECK: 11111
  // clang-format on

  {
    std::cout << "Passing state from another kernel as argument iteratively "
                 "(kernel mode)"
              << std::endl;
    auto state = cudaq::get_state(test_init_state{}, 2);
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param{}, &state);
      std::cout << "Iteration: " << i << std::endl;
      printCounts(counts);
      state = cudaq::get_state(test_state_param{}, &state);
    }
  }
  // clang-format off
  // CHECK: Passing state from another kernel as argument iteratively (kernel mode)
  // CHECK: Iteration: 0
  // CHECK: 01
  // CHECK: 11
  // CHECK: Iteration: 1
  // CHECK: 00
  // CHECK: 10
  // CHECK: Iteration: 2
  // CHECK: 01
  // CHECK: 11
  // CHECK: Iteration: 3
  // CHECK: 00
  // CHECK: 10
  // clang-format on

  return 0;
}
