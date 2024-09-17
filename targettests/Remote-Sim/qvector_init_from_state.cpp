/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// REQUIRES: c++20

// clang-format off
// TODO-FIX-KERNEL-EXEC
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <vector>
#include <string>

__qpu__ void test_init_state() {
  cudaq::qvector q(2);
  ry(M_PI/2.0, q[0]);
}

__qpu__ void test_init_large_state() {
  cudaq::qvector q(14);
  ry(M_PI/2.0, q[0]);
}

__qpu__ void test_state_param(cudaq::state* state) {
  cudaq::qvector q1(state);
}

__qpu__ void test_state_param2(cudaq::state* state, cudaq::pauli_word w) {
    cudaq::qvector q(state);
    cudaq::exp_pauli(.5, q, w);
}

__qpu__ void test_state_param3(cudaq::state *initial_state, std::vector<cudaq::pauli_word>& words) {
  cudaq::qvector q(initial_state);
  for (std::size_t i = 0; i < words.size(); ++i) {
    cudaq::exp_pauli(.5, q, words[i]);
  }
}

__qpu__ void test_state_param4(cudaq::state *initial_state, std::vector<double> &coefficients, std::vector<cudaq::pauli_word>& words) {
  cudaq::qvector q(initial_state);
  for (std::size_t i = 0; i < words.size(); ++i) {
    cudaq::exp_pauli(coefficients[i], q, words[i]);
  }
}

void printCounts(cudaq::sample_result& result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << '\n';
  }
}

int main() {
  std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0., 0., 0., 0., 0.};
  std::vector<cudaq::complex> vec1{0., 0.,  0., 0., 0., 0., M_SQRT1_2, M_SQRT1_2};
  auto state = cudaq::state::from_data(vec);
  auto state1 = cudaq::state::from_data(vec1);
  {
      std::cout << "Passing state created from data as argument (kernel mode)\n";
      auto counts = cudaq::sample(test_state_param, &state);
      printCounts(counts);

      counts = cudaq::sample(test_state_param, &state1);
      printCounts(counts);
  }

// CHECK: Passing state created from data as argument (kernel mode)
// CHECK: 000
// CHECK: 100

// CHECK: 011
// CHECK: 111

  {
    std::cout << "Passing state from another kernel as argument (kernel mode)\n";
    auto state = cudaq::get_state(test_init_state);
    auto counts = cudaq::sample(test_state_param, &state);
    printCounts(counts);
  }

// CHECK: Passing state from another kernel as argument (kernel mode)
// CHECK: 00
// CHECK: 10

  {
    std::cout << "Passing large state from another kernel as argument (kernel mode)\n";
    auto largeState = cudaq::get_state(test_init_large_state);
    auto counts = cudaq::sample(test_state_param, &largeState);
    printCounts(counts);
  }

// CHECK: Passing large state from another kernel as argument (kernel mode)
// CHECK: 00000000000000
// CHECK: 10000000000000

  {
    std::cout << "Passing state from another kernel as argument iteratively (kernel mode)\n";
    auto state = cudaq::get_state(test_init_state);
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param2, &state, cudaq::pauli_word{"XX"});
      std::cout << "Iteration: " << i << std::endl;
      printCounts(counts);
      state = cudaq::get_state(test_state_param2, &state, cudaq::pauli_word{"XX"});
    }
  }

// CHECK: Passing state from another kernel as argument iteratively (kernel mode)
// CHECK: Iteration: 0
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 1
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 2
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 3
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11

  {
    std::cout << "Passing state from another kernel as argument iteratively with vector args (kernel mode)\n";
    auto state = cudaq::get_state(test_init_state);
    auto words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XX"}};
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param3, &state, words);
      std::cout << "Iteration: " << i << std::endl;
      printCounts(counts);
      state = cudaq::get_state(test_state_param3, &state, words);
      words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XY"}};
    }
  }

// Passing state from another kernel as argument iteratively with vector args (kernel mode)
// CHECK: Iteration: 0
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 1
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 2
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 3
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11

  {
    std::cout << "Passing state from another kernel as argument iteratively with vector args with 2 elements (kernel mode)\n";
    auto state = cudaq::get_state(test_init_state);
    auto words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XX"}, cudaq::pauli_word{"YY"}};
    auto coeffs = std::vector<double>{0.5, 0.6};
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param4, &state, coeffs, words);
      std::cout << "Iteration: " << i << std::endl;
      printCounts(counts);
      state = cudaq::get_state(test_state_param4, &state, coeffs, words);
      words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XY"}, cudaq::pauli_word{"YX"}};
      coeffs = std::vector<double>{0.5 * i};
    }
  }

// CHECK: Passing state from another kernel as argument iteratively with vector args with 2 elements (kernel mode)
// CHECK: Iteration: 0
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 1
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 2
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: Iteration: 3
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
}
