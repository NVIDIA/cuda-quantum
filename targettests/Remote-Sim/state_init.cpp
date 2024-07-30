/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu %s -o %t && %t
// clang-format on

#include <cudaq.h>
#include <iostream>

__qpu__ void test_init_state() {
  cudaq::qvector q(2);
  ry(M_PI/2.0, q[0]);
}

__qpu__ void test_state_param(cudaq::state* inState) {
  cudaq::qvector q1(inState);
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
  // std::vector<cudaq::complex> vec{M_SQRT1_2, M_SQRT1_2, 0., 0., 0., 0., 0., 0.};
  // std::vector<cudaq::complex> vec1{0., 0.,  0., 0., 0., 0., M_SQRT1_2, M_SQRT1_2};
  // auto state = cudaq::state::from_data(vec);
  // auto state1 = cudaq::state::from_data(vec1);
  // {
  //     // Passing state data as argument (kernel mode)
  //     auto counts = cudaq::sample(test_state_param, &state);
  //     printCounts(counts);

  //     counts = cudaq::sample(test_state_param, &state1);
  //     printCounts(counts);
  // }

// CHECK: 000
// CHECK: 100

// CHECK: 011
// CHECK: 111

  {
    auto state = cudaq::get_state(test_init_state);
    //std::cout << "State sim precision: " << (state.get_precision() == cudaq::SimulationState::precision::fp32) << std::endl;
    auto counts = cudaq::sample(test_state_param, &state);
    printCounts(counts);
  }
}

// CHECK: 00
// CHECK: 10
