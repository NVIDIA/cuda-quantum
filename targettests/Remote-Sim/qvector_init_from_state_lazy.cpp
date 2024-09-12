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

struct test_init_state {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    ry(M_PI/2.0, q[0]);
  }
};

struct test_init_large_state {
  void operator()() __qpu__ {
    cudaq::qvector q(14);
    ry(M_PI/2.0, q[0]);
  }
};

struct test_state_param {
  void operator()(cudaq::state *initial_state) __qpu__ {
    cudaq::qvector q(initial_state);
  }
};

struct test_state_param2 {
  void operator()(cudaq::state *initial_state, cudaq::pauli_word w) __qpu__ {
    cudaq::qvector q(initial_state);
    cudaq::exp_pauli(.5, q, w);
  }
};

struct test_state_param3 {
  void operator()(cudaq::state *initial_state, std::vector<cudaq::pauli_word>& words) __qpu__ {
    cudaq::qvector q(initial_state);
    for (std::size_t i = 0; i < words.size(); ++i) {
      cudaq::exp_pauli(.5, q, words[i]);
    }
  }
};


struct test_state_param4 {
  void operator()(cudaq::state *initial_state, std::vector<double> &coefficients, std::vector<cudaq::pauli_word>& words) __qpu__ {
    cudaq::qvector q(initial_state);
    for (std::size_t i = 0; i < words.size(); ++i) {
      cudaq::exp_pauli(coefficients[i], q, words[i]);
    }
  }
};

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
      // Passing state created from data as argument (kernel mode)
      auto counts = cudaq::sample(test_state_param{}, &state);
      printCounts(counts);

      counts = cudaq::sample(test_state_param{}, &state1);
      printCounts(counts);
  }

// CHECK: 000
// CHECK: 100

// CHECK: 011
// CHECK: 111

  {
    // Passing state from another kernel as argument (kernel mode)
    auto state = cudaq::get_state(test_init_state{});
    auto counts = cudaq::sample(test_state_param{}, &state);
    printCounts(counts);
  }

// CHECK: 00
// CHECK: 10

  {
    // Passing large state from another kernel as argument (kernel mode)
    auto largeState = cudaq::get_state(test_init_large_state{});
    auto counts = cudaq::sample(test_state_param{}, &largeState);
    printCounts(counts);
  }

// CHECK: 00000000000000
// CHECK: 10000000000000

  {
    // Passing state from another kernel as argument iteratively (kernel mode)
    auto state = cudaq::get_state(test_init_state{});
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param2{}, &state, cudaq::pauli_word{"XX"});
      printCounts(counts);
      state = cudaq::get_state(test_state_param2{}, &state, cudaq::pauli_word{"XX"});
    }
  }

// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11

  {
    // Passing state from another kernel as argument iteratively (kernel mode)
    auto state = cudaq::get_state(test_init_state{});
    auto words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XX"}};
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param3{}, &state, words);
      printCounts(counts);
      state = cudaq::get_state(test_state_param3{}, &state, words);
      words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XY"}};
    }
  }

// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11

  {
    // Passing state from another kernel as argument iteratively (kernel mode)
    auto state = cudaq::get_state(test_init_state{});
    auto words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XX"}, cudaq::pauli_word{"YY"}};
    auto coeffs = std::vector<double>{0.5, 0.6};
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param4{}, &state, coeffs, words);
      printCounts(counts);
      state = cudaq::get_state(test_state_param4{}, &state, coeffs, words);
      words = std::vector<cudaq::pauli_word>{cudaq::pauli_word{"XY"}, cudaq::pauli_word{"YX"}};
      coeffs = std::vector<double>{0.5 * i};
    }
  }

// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
// CHECK: 00
// CHECK: 01
// CHECK: 10
// CHECK: 11
}
