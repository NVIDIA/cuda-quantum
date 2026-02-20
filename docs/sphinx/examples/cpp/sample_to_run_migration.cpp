/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cudaq.h>
#include <map>
#include <string>

// [Begin Sample_Works]
__qpu__ void bell() {
  cudaq::qvector q(2);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
}

__qpu__ void reset_pattern() {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q);
  x(q);
}
// [End Sample_Works]

// [Begin Example1]
struct simple_conditional {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    auto r = mz(q[0]);
    if (r) {
      x(q[1]);
    }
    return mz(q[1]);
  }
};
// [End Example1]

// [Begin Example2]
struct multi_measure {
  std::vector<bool> operator()() __qpu__ {
    std::vector<bool> results(3);
    cudaq::qvector q(3);
    h(q);
    results[0] = mz(q[0]);
    results[1] = mz(q[1]);
    if (results[0] && results[1]) {
      x(q[2]);
    }
    results[2] = mz(q[2]);
    return results;
  }
};
// [End Example2]

// [Begin Example3]
struct teleport {
  std::vector<bool> operator()() __qpu__ {
    std::vector<bool> results(3);
    cudaq::qvector q(3);
    x(q[0]);

    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);

    x<cudaq::ctrl>(q[0], q[1]);
    h(q[0]);

    results[0] = mz(q[0]);
    results[1] = mz(q[1]);

    if (results[1])
      x(q[2]);
    if (results[0])
      z(q[2]);

    results[2] = mz(q[2]);
    return results;
  }
};
// [End Example3]

int main() {
  // [Begin Sample_Works_Run]
  printf("Implicit measurements:\n");
  cudaq::sample(bell).dump();

  printf("\nMid-circuit measurement with reset:\n");
  cudaq::sample(reset_pattern).dump();

  cudaq::sample_options options{.explicit_measurements = true};
  printf("\nWith explicit_measurements option:\n");
  cudaq::sample(options, reset_pattern).dump();
  // [End Sample_Works_Run]

  // [Begin Example1Run]
  auto results1 = cudaq::run(100, simple_conditional{});
  std::size_t nOnes = std::ranges::count(results1, true);
  printf("Measured |1> %zu out of %zu shots\n", nOnes, results1.size());
  // [End Example1Run]

  // [Begin Example2Run]
  auto results2 = cudaq::run(100, multi_measure{});
  for (std::size_t i = 0; i < 5 && i < results2.size(); ++i) {
    for (auto b : results2[i])
      printf("%d", (int)b);
    printf("\n");
  }
  // [End Example2Run]

  // [Begin Example3Run]
  auto results3 = cudaq::run(100, teleport{});
  assert(std::ranges::all_of(results3, [](const auto &r) { return r[2]; }));
  printf("Teleportation succeeded on all %zu shots\n", results3.size());
  // [End Example3Run]

  // [Begin Result_Processing]
  auto results = cudaq::run(1000, multi_measure{});
  std::map<std::string, std::size_t> counts;
  for (const auto &shot : results) {
    std::string bits;
    for (auto b : shot)
      bits += b ? '1' : '0';
    counts[bits]++;
  }
  for (const auto &[bits, count] : counts)
    printf("%s : %zu\n", bits.c_str(), count);
  // [End Result_Processing]

  return 0;
}
