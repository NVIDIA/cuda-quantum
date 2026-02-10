/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// RUN: CUDAQ_DEFAULT_SIMULATOR=stim nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

struct kernel {
  std::vector<int> operator()(const int n_iter) __qpu__ {
    cudaq::qubit q0;
    cudaq::qubit q1;
    std::vector<int> allResults(n_iter + 1); // q0 results + q1 result
    for (int i = 0; i < n_iter; i++) {
      h(q0);
      allResults[i] = mz(q0);
      if (allResults[i])
        x(q1); // toggle q1 on every q0 coin toss that lands heads
    }
    allResults[n_iter] = mz(q1); // the measured q1 should contain the parity
                                 // bit for the q0 measurements
    return allResults;
  }
};

int main() {

  int nShots = 100;
  int nIter = 5;
  cudaq::set_random_seed(13);

  auto results = cudaq::run(/*shots=*/nShots, kernel{}, nIter);

  std::size_t q1result_0 = std::ranges::count_if(
      results, [nIter](const auto &r) { return r[nIter] == 0; });
  std::size_t q1result_1 = results.size() - q1result_0;

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

  // For each shot, do the parity check
  int parityCheckSuccessCount = 0;
  for (int shot = 0; shot < nShots; shot++) {
    int parity = 0;
    for (int i = 0; i < nIter; i++)
      parity ^= results[shot][i];
    if (parity == results[shot][nIter])
      parityCheckSuccessCount++;
  }

  if (parityCheckSuccessCount != nShots) {
    // Output results for easy viewing
    for (int shot = 0; shot < nShots; shot++) {
      std::cout << "Shot " << shot << ": q0=[";
      for (int i = 0; i < nIter; i++)
        std::cout << results[shot][i];
      std::cout << "], q1=" << results[shot][nIter] << '\n';
    }
    std::cout << "parityCheckSuccessCount: " << parityCheckSuccessCount << '\n';
    std::cout << "nShots:                  " << nShots << '\n';
    return 3;
  }

  std::cout << "SUCCESS\n";
  return 0;
}

// CHECK: SUCCESS
