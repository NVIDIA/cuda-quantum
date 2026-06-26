/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Benchmark for issue #4230: cost of X-pair expansion of negated controls.
//
// This is a manual performance regression aid, not a correctness or ctest
// assertion. Build with the target of interest and run with representative n.
//
// Disjoint 4-control blocks: gate g controls q[5g..5g+3], target q[5g+4].
// Disjoint qubit sets per gate => inserted X-pairs from consecutive gates can
// NEVER cancel, so the measured time reflects the real X-tax (a fixed-control
// kernel would let the optimizer collapse interior X;X pairs).
//
// Target is fixed at compile time (one binary per target). usage:
//   nvq++ --target nvidia negated_control_perf.cpp -o bench_nvidia
//   ./bench_nvidia 40 neg 5
//   ./bench_nvidia 40 plain 5

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cudaq.h>
#include <string>

struct qrom_neg {
  void operator()(int n, int reps) __qpu__ {
    cudaq::qvector q(n);
    for (int r = 0; r < reps; r++)
      for (int b = 0; b + 4 < n; b += 5)
        x<cudaq::ctrl>(!q[b], !q[b + 1], !q[b + 2], !q[b + 3], q[b + 4]);
  }
};

struct qrom_plain {
  void operator()(int n, int reps) __qpu__ {
    cudaq::qvector q(n);
    for (int r = 0; r < reps; r++)
      for (int b = 0; b + 4 < n; b += 5)
        x<cudaq::ctrl>(q[b], q[b + 1], q[b + 2], q[b + 3], q[b + 4]);
  }
};

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s <n> <neg|plain> <reps>\n", argv[0]);
    return 1;
  }
  int n = std::atoi(argv[1]);
  std::string variant = argv[2];
  int reps = std::atoi(argv[3]);

  auto run = [&]() {
    if (variant == "neg")
      return cudaq::get_state(qrom_neg{}, n, reps);
    return cudaq::get_state(qrom_plain{}, n, reps);
  };

  run(); // warm-up (context init, JIT)
  auto t0 = std::chrono::high_resolution_clock::now();
  auto s = run();
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("%-12s n=%2d %-6s reps=%d : %9.2f ms\n", argv[0], n, variant.c_str(),
         reps, ms);
  return 0;
}
