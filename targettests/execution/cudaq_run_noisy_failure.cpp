/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target stim %s -o %t && %t | FileCheck %s
// clang-format on

// A `cudaq::run` whose launch throws must not leave its noise model installed
// for later launches in the same process.

#include <cstdio>
#include <cudaq.h>

struct flip {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    return mz(q);
  }
};

// Stim cannot simulate `t`, so this launch throws.
struct non_clifford {
  bool operator()() __qpu__ {
    cudaq::qubit q;
    t(q);
    return mz(q);
  }
};

int main() {
  // Stim's noise sampling asserts on a channel probability of 1.
  cudaq::noise_model noise;
  noise.add_channel("x", {0}, cudaq::bit_flip_channel(0.5));
  // A typed handler misses this in static C++ runtime builds, so check the
  // message that `CUDAQ_ERROR` logs before throwing instead.
  try {
    cudaq::run(1, noise, non_clifford{});
  } catch (...) {
    std::printf("caught\n");
  }
  std::size_t numTrue = 0;
  for (bool result : cudaq::run(100, flip{}))
    numTrue += result;
  std::printf("noiseless run: %zu of 100 true\n", numTrue);
  return 0;
}

// CHECK: [error] {{.*}}Gate not supported by Stim simulator: t
// CHECK: caught
// CHECK: noiseless run: 100 of 100 true
