/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: CUDAQ_DEFAULT_SIMULATOR=stim nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --enable-mlir %s -o %t
// clang-format on

#include <cudaq.h>
#include <iostream>

struct kernel {
  void operator()(const int n_iter) __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    for (int i = 0; i < n_iter; i++) {
      h(q0);
      if (mz(q0))
        x(q1); // toggle q1 on every q0 coin toss that lands heads
    }
    auto q1result = mz(q1); // the measured q1 should contain the parity bit for
                            // the q0 measurements
  }
};

int main() {

  int nShots = 100;
  int nIter = 5;
  cudaq::set_random_seed(13);

  // Sample
  auto counts = cudaq::sample(/*shots=*/nShots, kernel{}, nIter);
  counts.dump();

  auto q1result_0 = counts.count("0", "q1result");
  auto q1result_1 = counts.count("1", "q1result");
  if (q1result_0 + q1result_1 != nShots) {
    std::cout << "q1result_0 (" << q1result_0 << ") + q1result_1 ("
              << q1result_1 << ") != nShots (" << nShots << ")\n";
    return 1;
  }
  if (q1result_0 < static_cast<int>(0.3 * nShots) ||
      q1result_0 > static_cast<int>(0.7 * nShots)) {
    std::cout << "q1result_0 (" << q1result_0
              << ") is not within expected range ["
              << static_cast<int>(0.3 * nShots) << ","
              << static_cast<int>(0.7 * nShots) << "]\n";
    return 2;
  }

  auto &platform = cudaq::get_platform();

  // If you run this on quantinuum hardware (i.e. H2-1E), the following parity
  // check will check that the results look reasonable. Skip the parity check on
  // `--emulate` runs because the unnamed measurement is not saved and therefore
  // cannot be compared in a parity check.
  if (!platform.is_emulated()) {

    // Populate binaryQ0results[iteration][shot]
    std::vector<std::vector<int>> binaryQ0results;
    for (int i = 0; i < nIter; i++) {
      auto vecOfStrings = counts.sequential_data("r0000" + std::to_string(i));
      binaryQ0results.push_back({});
      for (int shot = 0; shot < nShots; shot++)
        binaryQ0results[i].push_back(vecOfStrings[shot][0] == '0' ? 0 : 1);
    }

    // Populate binaryQ1results[shot]
    auto q1result = counts.sequential_data("q1result");
    std::vector<int> binaryQ1results(nShots);
    for (int shot = 0; shot < nShots; shot++)
      binaryQ1results[shot] = q1result[shot][0] == '0' ? 0 : 1;

    // For each shot, do the parity check
    int parityCheckSuccessCount = 0;
    for (int shot = 0; shot < nShots; shot++) {
      int parity = 0;
      for (int i = 0; i < nIter; i++)
        parity ^= binaryQ0results[i][shot];
      if (parity == binaryQ1results[shot])
        parityCheckSuccessCount++;
    }

    if (parityCheckSuccessCount != nShots) {
      // Output q0result and q1results for easy viewing
      std::cout << "q1result  : ";
      for (auto ch : q1result)
        std::cout << ch;
      std::cout << '\n';
      for (int i = 0; i < nIter; i++) {
        std::cout << "q0result%" << i << ": ";
        for (auto b : binaryQ0results[i])
          std::cout << b;
        std::cout << '\n';
      }

      std::cout << "parityCheckSuccessCount: " << parityCheckSuccessCount
                << '\n';
      std::cout << "nShots:                  " << nShots << '\n';
      return 3;
    }
  }

  std::cout << "SUCCESS\n";
  return 0;
}

// CHECK: SUCCESS
