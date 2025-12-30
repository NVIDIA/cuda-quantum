/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// Simulators
// RUN: nvq++ --enable-mlir  %s                              -o %t && %t | FileCheck %s

// Quantum emulators
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target anyon      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt %t | FileCheck %s
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

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
  void operator()(cudaq::state *state, cudaq::pauli_word w) __qpu__ {
    cudaq::qvector q(state);
    cudaq::exp_pauli(1.0, q, w);
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
    std::cout << "Passing state from another kernel as argument"
                 " with pauli word arg (kernel mode)"
              << std::endl;
    auto state = cudaq::get_state(test_init_state{}, 2);
    auto counts =
        cudaq::sample(test_state_param{}, &state, cudaq::pauli_word{"XX"});
    printCounts(counts);
  }
  // clang-format off
  // CHECK: Passing state from another kernel as argument with pauli word arg (kernel mode)
  // CHECK: 00
  // CHECK: 01
  // CHECK: 10
  // CHECK: 11
  // clang-format on

  // TODO: add tests for vectors of pauli words after we can lifts the arrays of pauli words.

  return 0;
}
