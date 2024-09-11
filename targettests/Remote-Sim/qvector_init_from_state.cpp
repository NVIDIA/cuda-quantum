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
/*
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
*/
// __qpu__ void test_state_param2(cudaq::pauli_word w, int n) {
//     cudaq::qvector q(n);
//     cudaq::exp_pauli(.5, q, w);
// }

__qpu__ void test_state_param2(int k, int n) {
    cudaq::qvector q(n);
    cudaq::exp_pauli(k/2, q, "XY");
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
  /*
  std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0., 0., 0., 0., 0.};
  std::vector<cudaq::complex> vec1{0., 0.,  0., 0., 0., 0., M_SQRT1_2, M_SQRT1_2};
  auto state = cudaq::state::from_data(vec);
  auto state1 = cudaq::state::from_data(vec1);
  {
      // Passing state created from data as argument (kernel mode)
      auto counts = cudaq::sample(test_state_param, &state);
      printCounts(counts);

      counts = cudaq::sample(test_state_param, &state1);
      printCounts(counts);
  }

// CHECK: 000
// CHECK: 100

// CHECK: 011
// CHECK: 111

  {
    // Passing state from another kernel as argument (kernel mode)
    auto state = cudaq::get_state(test_init_state);
    auto counts = cudaq::sample(test_state_param, &state);
    printCounts(counts);
  }

// CHECK: 00
// CHECK: 10

  {
    // Passing large state from another kernel as argument (kernel mode)
    auto largeState = cudaq::get_state(test_init_large_state);
    auto counts = cudaq::sample(test_state_param, &largeState);
    printCounts(counts);
  }

// CHECK: 00000000000000
// CHECK: 10000000000000

  {
    
    // Passing state from another kernel as argument iteratively (kernel mode)
    auto state = cudaq::get_state(test_init_state);
    for (auto i = 0; i < 4; i++) {
      auto counts = cudaq::sample(test_state_param, &state);
      printCounts(counts);
      state = cudaq::get_state(test_state_param, &state);
    }
  }*/

  {
    //std::vector<cudaq::complex> vec{0., 0.,  0., 0., 0., 0., M_SQRT1_2, M_SQRT1_2};
    //auto state = cudaq::state::from_data(vec);
    //auto counts = cudaq::sample(test_state_param2, cudaq::pauli_word{"XY"}, 2);
    auto counts = cudaq::sample(test_state_param2, 3, 2);
    printCounts(counts);
  }

// CHECK: 00
// CHECK: 10
// CHECK: 00
// CHECK: 10
// CHECK: 00
// CHECK: 10
// CHECK: 00
// CHECK: 10
}
