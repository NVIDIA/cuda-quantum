/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

struct ghz {
  auto operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; ++i) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

// Test a bug with state reuse where the mgpu is working in the single-process
// mode due to small problem size.
CUDAQ_TEST(MGPUTester, checkStateReuse) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 20; // Below the threshold for distribution
  constexpr int numRuns = 100;
  // Run multiple times => state reuse should kick in.
  for (int i = 0; i < numRuns; ++i) {
    auto counts = cudaq::sample(ghz{}, numQubits);
    const std::string allZero(numQubits, '0');
    const std::string allOne(numQubits, '1');
    for (auto &[bits, count] : counts) {
      EXPECT_TRUE(bits == allZero || bits == allOne);
    }
  }
}

struct reorderState {
  void operator()() __qpu__ {
    cudaq::qvector q(6);
    h(q[5]);
    h(q[5]);
    mz(q);
  }
};

struct noisyCustomState {
  void operator()(const std::vector<cudaq::complex> &initialState) __qpu__ {
    cudaq::qvector q(initialState);
    h(q[0]);
    h(q[0]);
    mz(q);
  }
};

CUDAQ_TEST(MGPUTester, checkNoisyCustomStateReuse) {
  constexpr std::size_t numQubits = 6;
  constexpr std::size_t stateSize = std::size_t{1} << numQubits;
  constexpr int shots = 8;
  std::vector<cudaq::complex> zeroState(stateSize);
  std::vector<cudaq::complex> oneState(stateSize);
  zeroState.front() = 1.0;
  oneState.back() = 1.0;

  // Warm up the reusable descriptor with logically cancelling global-wire
  // gates, which may leave a non-identity physical wire ordering.
  cudaq::sample(shots, reorderState{});

  // The two local gates cancel. Tiny noise keeps trajectory replay active while
  // leaving each supplied basis state unchanged with overwhelming probability.
  cudaq::depolarization_channel depolarization(1e-6);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::h>({0}, depolarization);
  cudaq::set_noise(noise);

  const auto zeroCounts = cudaq::sample(shots, noisyCustomState{}, zeroState);
  EXPECT_EQ(zeroCounts.size(), 1);
  EXPECT_EQ(zeroCounts.most_probable(), std::string(numQubits, '0'));

  const auto oneCounts = cudaq::sample(shots, noisyCustomState{}, oneState);
  EXPECT_EQ(oneCounts.size(), 1);
  EXPECT_EQ(oneCounts.most_probable(), std::string(numQubits, '1'));
  cudaq::unset_noise();
}

struct noisyCustomStateObserve {
  void operator()(const std::vector<cudaq::complex> &initialState) __qpu__ {
    cudaq::qvector q(initialState);
    h(q[0]);
  }
};

CUDAQ_TEST(MGPUTester, checkNoisyCustomStateMultiTermObserve) {
  constexpr std::size_t numQubits = 6;
  constexpr std::size_t stateSize = std::size_t{1} << numQubits;
  std::vector<cudaq::complex> initialState(stateSize);
  initialState.front() = 1.0;

  cudaq::depolarization_channel depolarization(1e-6);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::h>({0}, depolarization);
  cudaq::set_noise(noise);

  const auto hamiltonian = cudaq::spin_op::x(0) + cudaq::spin_op::z(0);
  constexpr int shots = 4096;
  auto result = cudaq::observe(shots, noisyCustomStateObserve{}, hamiltonian,
                               initialState);
  EXPECT_NEAR(result.expectation(), 1.0, 0.1);
  cudaq::unset_noise();
}
