/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std -DNO_ADAPTIVE --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s
// RUN: nvq++ %cpp_std               --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std                                             %s -o %t && %t | FileCheck %s
// RUN: nvq++ -std=c++17 %s --enable-mlir -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <map>

struct test_adaptive {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    auto bit = mz(b);
    if (!bit) {
      x(b); // note that this is not allowed in base profile programs
    }
    return {mz(a), mz(b)};
  }
};

#define SAMPLE_AND_PRINT_GLOBAL_REG(TEST_NAME)                                 \
  do {                                                                         \
    auto result = cudaq::sample(nShots, TEST_NAME);                            \
    auto globalRegResults = cudaq::sample_result{                              \
        cudaq::ExecutionResult{result.to_map(cudaq::GlobalRegisterName)}};     \
    std::cout << #TEST_NAME << ":\n";                                          \
    globalRegResults.dump();                                                   \
  } while (false)

#define RUN_AND_PRINT_RESULT_DICT(TEST_NAME)                                   \
  do {                                                                         \
    auto results = cudaq::run(nShots, TEST_NAME);                              \
    std::map<std::string, std::size_t> bitstring_counts;                       \
    for (const auto &result : results) {                                       \
      std::string bits =                                                       \
          std::to_string(result[0]) + std::to_string(result[1]);               \
      bitstring_counts[bits]++;                                                \
    }                                                                          \
    std::cout << #TEST_NAME << ":\n";                                          \
    std::cout << "{";                                                          \
    for (const auto &[bits, count] : bitstring_counts) {                       \
      std::cout << " " << bits << ":" << count << " ";                         \
    }                                                                          \
    std::cout << "}\n";                                                        \
  } while (false)

int main() {
  const int nShots = 1000;

  // Check that qubits show up in the return order
  auto test1 = []() __qpu__ -> std::vector<bool> {
    cudaq::qubit a, b;
    x(a);
    return {mz(a), mz(b)};
  };
  RUN_AND_PRINT_RESULT_DICT(test1);
  // CHECK: test1:
  // CHECK: { 10:1000 }

  // Check that measurements will be implicitly added to kernels that have no
  // measurements.
  auto test2 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
  };
  SAMPLE_AND_PRINT_GLOBAL_REG(test2);
  // CHECK: test2:
  // CHECK: { 10:1000 }

  // Check that performing a quantum operation after the final measurement makes
  // all qubits appear in the global register.
#ifndef NO_ADAPTIVE
  auto test3 = test_adaptive{};
  RUN_AND_PRINT_RESULT_DICT(test3);
#else
  // Platforms that don't support the adaptive profile will test this instead.
  auto test3 = []() __qpu__ {
    cudaq::qubit a, b;
    x(a);
    x(b);
  };
  SAMPLE_AND_PRINT_GLOBAL_REG(test3);
#endif

  // CHECK: test3:
  // CHECK: { 11:1000 }

  // Check that mapping introduced qubits (and their corresponding hidden swaps)
  // are managed correctly and distinctly from user swaps.
  auto test4 = []() __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    swap(q[0], q[1]);
  };
  SAMPLE_AND_PRINT_GLOBAL_REG(test4);
  // CHECK: test4:
  // CHECK: { 01:1000 }

  return 0;
}
