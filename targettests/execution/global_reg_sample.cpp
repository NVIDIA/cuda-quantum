/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if %iqm_avail; then nvq++ --target iqm --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s; fi
// RUN: if %quantinuum_avail; then nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

#define SAMPLE_AND_PRINT_GLOBAL_REG(TEST_NAME)                                 \
  do {                                                                         \
    auto result = cudaq::sample(nShots, TEST_NAME);                            \
    auto globalRegResults = cudaq::sample_result{                              \
        cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};     \
    std::cout << #TEST_NAME << ":\n";                                          \
    globalRegResults.dump();                                                   \
  } while (false)

int main() {
  const int nShots = 1000;

  // Check that the bits show up in record order, not allocation order.
  auto test1 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    cudaq::record(b, a);
  };
  SAMPLE_AND_PRINT_GLOBAL_REG(test1);
  // CHECK: test1:
  // CHECK: { 01:1000 }

  // Check that only the recorded qubits appear in the global register.
  auto test2 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    cudaq::record(b);
  };
  SAMPLE_AND_PRINT_GLOBAL_REG(test2);
  // CHECK: test2:
  // CHECK: { 0:1000 }

  auto test3 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    x(b);
    cudaq::record(a, b);
  };
  SAMPLE_AND_PRINT_GLOBAL_REG(test3);
  // CHECK: test3:
  // CHECK: { 11:1000 }

  // Check that mapping introduced qubits (and their corresponding hidden swaps)
  // are managed correctly and distinctly from user swaps.
  auto test4 = []() __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    swap(q[0], q[1]);
    cudaq::record(q);
  };
  SAMPLE_AND_PRINT_GLOBAL_REG(test4);
  // CHECK: test4:
  // CHECK: { 01:1000 }

  return 0;
}
