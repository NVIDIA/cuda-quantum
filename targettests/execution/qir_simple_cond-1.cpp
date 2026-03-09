/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target stim --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --enable-mlir %s -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>
#include <map>

struct kernel {
  std::vector<bool> operator()() __qpu__ {
    std::vector<bool> results(2);
    cudaq::qubit q0;
    cudaq::qubit q1;
    h(q0);
    results[0] = mz(q0);
    if (results[0])
      x(q1);
    results[1] = mz(q1); // Every q1 measurement will be the same as q0
    return results;
  }
};

int main() {

  int nShots = 100;
  auto results = cudaq::run(/*shots=*/nShots, kernel{});

  // Count occurrences of each bitstring
  std::map<std::string, std::size_t> bitstring_counts;
  for (const auto &result : results) {
    std::string bits = std::to_string(result[0]) + std::to_string(result[1]);
    bitstring_counts[bits]++;
  }

  std::cout << "Bitstring counts\n";
  for (const auto &[bits, count] : bitstring_counts)
    std::cout << bits << ": " << count << "\n";

  // Assert that all shots contained "00" or "11", exclusively
  if (bitstring_counts.size() != 2 ||
      bitstring_counts.find("00") == bitstring_counts.end() ||
      bitstring_counts.find("11") == bitstring_counts.end()) {
    std::cout << "FAILURE: Unexpected bitstrings found\n";
    return 1;
  }
  
  std::cout << "SUCCESS\n";
  return 0;
}

// CHECK: SUCCESS
