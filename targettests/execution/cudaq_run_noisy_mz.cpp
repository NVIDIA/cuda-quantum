/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target density-matrix-cpu  %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target stim                %s -o %t && %t | FileCheck %s
/// FIXME: The second test crashes on tensorner with following error:
/// `runtime/nvqir/cutensornet/simulator_cutensornet.inc:384: 
/// bool nvqir::SimulatorTensorNetBase<ScalarType>::measureQubit(std::size_t) 
/// [with ScalarType = double; std::size_t = long unsigned int]: 
/// Assertion `std::abs(1.0 - (prob0 + prob1)) < 1e-3' failed.`
// SKIPPED: nvq++ --target tensornet           %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>

#define ASSERT_NEAR(val1, val2, tol) assert(std::fabs((val1) - (val2)) <= (tol))

struct kernel1 {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    return mz(q);
  }
};

struct kernel2 {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    x(q);
    return mz(q);
  }
};

int main() {
  cudaq::set_random_seed(13);
  constexpr double bitFlipRate = 0.25;
  cudaq::bit_flip_channel bf(bitFlipRate);
  cudaq::noise_model noise;
  // 25% bit flipping during measurement
  noise.add_channel("mz", {0}, bf);
  cudaq::set_noise(noise);
  std::size_t numShots = 100;
  {
    auto results = cudaq::run(numShots, kernel1{});
    // Count occurrences of each bitstring
    std::map<std::string, std::size_t> bitstring_counts;
    for (const auto &result : results) {
      std::string bits = std::to_string(result);
      bitstring_counts[bits]++;
    }
    auto bitstring_0_probability =
        static_cast<double>(bitstring_counts["0"]) / numShots;
    auto bitstring_1_probability =
        static_cast<double>(bitstring_counts["1"]) / numShots;

    printf("Bitstring '0' frequency: %.3f\n", bitstring_0_probability);
    printf("Bitstring '1' frequency: %.3f\n", bitstring_1_probability);

    // Due to measurement errors, we have both 0/1 results.
    assert(2 == bitstring_counts.size());
    ASSERT_NEAR(bitstring_0_probability, bitFlipRate, 0.1);
    ASSERT_NEAR(bitstring_1_probability, 1.0 - bitFlipRate, 0.1);

    printf("success!\n");
  }
  {
    auto results = cudaq::run(numShots, kernel2{});
    // Count occurrences of each bitstring
    std::map<std::string, std::size_t> bitstring_counts;
    for (const auto &result : results) {
      std::string bits = std::to_string(result[0]) + std::to_string(result[1]);
      bitstring_counts[bits]++;
    }

    auto bitstring_01_probability =
        static_cast<double>(bitstring_counts["01"]) / numShots;
    auto bitstring_11_probability =
        static_cast<double>(bitstring_counts["11"]) / numShots;

    printf("Bitstring '01' frequency: %.3f\n", bitstring_01_probability);
    printf("Bitstring '11' frequency: %.3f\n", bitstring_11_probability);

    // We only have measurement noise on the first qubit.
    assert(2 == bitstring_counts.size());
    ASSERT_NEAR(bitstring_01_probability, bitFlipRate, 0.1);
    ASSERT_NEAR(bitstring_11_probability, 1.0 - bitFlipRate, 0.1);

    printf("success!\n");
  }
  cudaq::unset_noise(); // clear for subsequent tests
  return 0;
}

// CHECK: success!
// CHECK: success!
