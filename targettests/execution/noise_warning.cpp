/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target=qpp-cpu %s -o %t && CUDAQ_LOG_LEVEL=warn %t 2>&1 | FileCheck %s


#include <cudaq.h>
#include <cstdio>

struct bell_sample {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    mz(q);
  }
};

int main() {
  std::printf("=== unsupported noise ===\n");
  {
    cudaq::noise_model noise;
    cudaq::depolarization2 depol2(0.1);
    noise.add_channel<cudaq::types::x>({0, 1}, depol2);
    auto counts =
        cudaq::sample(cudaq::sample_options{.shots = 10, .noise = noise},
                      bell_sample{});
    counts.dump();
  }

  std::printf("=== no noise ===\n");
  {
    auto counts = cudaq::sample(/*shots=*/10, bell_sample{});
    counts.dump();
  }

  std::printf("=== default noise options ===\n");
  {
    // Default-constructed noise_model has registered channel types but no gate
    // mappings, so get_channels() returns empty for h/x/mz in the kernel.
    cudaq::noise_model default_noise{};
    auto counts =
        cudaq::sample(cudaq::sample_options{.shots = 10, .noise = default_noise},
                      bell_sample{});
    counts.dump();
  }
  return 0;
}

// CHECK-LABEL: === unsupported noise ===
// CHECK: Applying noise is not supported on qpp simulator.
// CHECK-LABEL: === no noise ===
// CHECK-NOT: Applying noise is not supported
// CHECK-LABEL: === default noise options ===
// CHECK-NOT: Applying noise is not supported
