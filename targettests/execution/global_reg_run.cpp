/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
/// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/3708
// SKIPPED: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
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
  auto test0 = []() __qpu__ -> std::vector<bool> {
    cudaq::qubit a, b;
    x(a);
    return {mz(a), mz(b)};
  };
  RUN_AND_PRINT_RESULT_DICT(test0);
  // CHECK: test0:
  // CHECK: { 10:1000 }

  // Check that performing a quantum operation after the final measurement makes
  // all qubits appear in the global register.
  auto test1 = test_adaptive{};
  RUN_AND_PRINT_RESULT_DICT(test1);
  // CHECK: test1:
  // CHECK: { 11:1000 }

  return 0;
}
