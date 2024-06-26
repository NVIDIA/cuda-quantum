/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <algorithm>
#include <cudaq/algorithm.h>

std::vector<cudaq::complex> randomState(int numQubits) {
  std::vector<cudaq::complex> stateVec(1ULL << numQubits);
  std::generate(stateVec.begin(), stateVec.end(), []() -> cudaq::complex {
    thread_local std::default_random_engine
        generator; // thread_local so we don't have to do any locking
    thread_local std::normal_distribution<double> distribution(
        0.0, 1.0); // mean = 0.0, stddev = 1.0
    return cudaq::complex(distribution(generator), distribution(generator));
  });

  const double norm =
      std::sqrt(std::accumulate(stateVec.begin(), stateVec.end(), 0.0,
                                [](double accumulatedNorm, cudaq::complex val) {
                                  return accumulatedNorm + std::norm(val);
                                }));
  std::transform(stateVec.begin(), stateVec.end(), stateVec.begin(),
                 [norm](std::complex<double> x) { return x / norm; });
  return stateVec;
}

struct test_state_vector_init {
  void operator()(const std::vector<cudaq::complex> &stateVec) __qpu__ {
    cudaq::qvector q(stateVec);
    mz(q);
  }
};

CUDAQ_TEST(AllocationTester, checkAllocationFromStateVecGeneral) {
  constexpr int numQubits = 5;
  // Large number of shots
  constexpr int numShots = 1000000;
  const auto stateVec = randomState(numQubits);
  cudaq::set_random_seed(13); // set for repeatability
  auto counts = cudaq::sample(numShots, test_state_vector_init{}, stateVec);
  counts.dump();
  for (const auto &[bitStrOrg, count] : counts) {
    auto bitStr = bitStrOrg;
    std::reverse(bitStr.begin(), bitStr.end());
    const int val = std::stoi(bitStr, nullptr, 2);
    const double prob = 1.0 * count / numShots;
    const double expectedProb = std::norm(stateVec[val]);
    if (expectedProb > 1e-6) {
      const double relError = std::abs(expectedProb - prob) / expectedProb;
      // Less than 10% difference (relative)
      EXPECT_LT(relError, 0.1);
    }
  }
}

// Same as test_state_vector_init with some dummy gates
struct test_state_vector_init_gate {
  void operator()(const std::vector<cudaq::complex> &stateVec) __qpu__ {
    cudaq::qvector q(stateVec);
    // Identity
    cudaq::exp_pauli(1.0, q, "XXXXX");
    cudaq::exp_pauli(-1.0, q, "XXXXX");
    mz(q);
  }
};

CUDAQ_TEST(AllocationTester, checkAllocationFromStateVecWithGate) {
  constexpr int numQubits = 5;
  // Large number of shots
  constexpr int numShots = 1000000;
  const auto stateVec = randomState(numQubits);
  cudaq::set_random_seed(13); // set for repeatability
  auto counts =
      cudaq::sample(numShots, test_state_vector_init_gate{}, stateVec);
  counts.dump();
  for (const auto &[bitStrOrg, count] : counts) {
    auto bitStr = bitStrOrg;
    std::reverse(bitStr.begin(), bitStr.end());
    const int val = std::stoi(bitStr, nullptr, 2);
    const double prob = 1.0 * count / numShots;
    const double expectedProb = std::norm(stateVec[val]);
    if (expectedProb > 1e-6) {
      const double relError = std::abs(expectedProb - prob) / expectedProb;
      // Less than 20% difference (relative)
      EXPECT_LT(relError, 0.2);
    }
  }
}

struct test_allocation {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    cudaq::qubit r;

    cnot(q, r);
    mz(q);
    mz(r);
  }
};

struct test_resizing {
  void operator()() __qpu__ {
    // Start with an initial allocation of 2 qubits.
    cudaq::qvector q(2);
    cudaq::x(q);
    auto result = mz(q[0]);
    auto result1 = mz(q[1]);
    if (result && result1) {
      // Allocate two more qubits mid-circuit.
      cudaq::qvector q2(2);
      auto result2 = mz(q2);
    }
  }
};

struct test_bell_init {
  void operator()() __qpu__ {
    // Start with an initial allocation of 2 qubits in a specific state.
    cudaq::qvector q({M_SQRT1_2, 0.0, 0.0, M_SQRT1_2});
    mz(q);
  }
};

struct test_state_expand_init {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    x(q);
    // Add 2 more qubits in Bell state
    cudaq::qvector q1({M_SQRT1_2, 0.0, 0.0, M_SQRT1_2});
    mz(q);
    mz(q1);
  }
};

CUDAQ_TEST(AllocationTester, checkSimple) {
  test_allocation{}();

  auto counts = cudaq::sample(test_allocation{});
  EXPECT_EQ(2, counts.size());
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
  EXPECT_EQ(c, 1000);
}

CUDAQ_TEST(AllocationTester, checkSetState) {
  auto counts = cudaq::sample(test_bell_init{});
  EXPECT_EQ(2, counts.size());
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
  EXPECT_EQ(c, 1000);
}

CUDAQ_TEST(AllocationTester, checkSetStateExpandRegister) {
  auto counts = cudaq::sample(test_state_expand_init{});
  EXPECT_EQ(2, counts.size());
  counts.dump();
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "1100" || bits == "1111");
  }
  EXPECT_EQ(c, 1000);
}

#ifdef CUDAQ_BACKEND_DM
// Tests for a previous bug in the density simulator, where
// the qubit ordering flipped after resizing the density matrix
// with new qubits.
CUDAQ_TEST(AllocationTester, checkDensityOrderingBug) {
  test_resizing{}();

  auto counts = cudaq::sample(100, test_resizing{});
  counts.dump();
  EXPECT_EQ(1, counts.size());
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "1100");
  }
  EXPECT_EQ(c, 100);
}
#endif

struct test_allocation_from_state {
  void operator()(cudaq::state state) __qpu__ {
    cudaq::qvector q(state);
    mz(q);
  }
};

CUDAQ_TEST(AllocationTester, checkAllocationFromRetrievedState) {
  auto bellState = cudaq::get_state([]() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    cx(q[0], q[1]);
  });
  auto counts = cudaq::sample(test_allocation_from_state{}, bellState);
  counts.dump();
  EXPECT_EQ(2, counts.size());
  int c = 0;
  for (auto &[bits, count] : counts) {
    c += count;
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
  EXPECT_EQ(c, 1000);
}

CUDAQ_TEST(AllocationTester, checkChainingGetState) {
  auto state1 = cudaq::get_state([]() __qpu__ {
    cudaq::qvector q(2);
    // First half of the circuit
    h(q[0]);
  });
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(std::abs(state1.amplitude({0, 0})), 0.5, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 0})), 0.5, 1e-6);
#else
  EXPECT_NEAR(std::abs(state1.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 0})), M_SQRT1_2, 1e-6);
#endif
  EXPECT_NEAR(std::abs(state1.amplitude({1, 1})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state1.amplitude({0, 1})), 0.0, 1e-9);

  // Second half of the circuit
  auto state2 = cudaq::get_state(
      [](cudaq::state state) __qpu__ {
        cudaq::qvector q(state);
        cx(q[0], q[1]);
      },
      state1);
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(std::abs(state2.amplitude({0, 0})), 0.5, 1e-6);
  EXPECT_NEAR(std::abs(state2.amplitude({1, 1})), 0.5, 1e-6);
#else
  EXPECT_NEAR(std::abs(state2.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state2.amplitude({1, 1})), M_SQRT1_2, 1e-6);
#endif
  EXPECT_NEAR(std::abs(state2.amplitude({1, 0})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state2.amplitude({0, 1})), 0.0, 1e-9);

  // Both states should remain valid, hence we can use the two states for
  // computation, e.g., overlap
  const auto overlap = state1.overlap(state2);
// Expected: 0.5 for state vec and 0.25 for density matrix
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(std::abs(overlap), 0.25, 1e-6);
#else
  EXPECT_NEAR(std::abs(overlap), 0.5, 1e-6);
#endif
}

#ifdef CUDAQ_BACKEND_TENSORNET_MPS
CUDAQ_TEST(AllocationTester, checkStateFromMpsData) {
  {
    const std::vector<std::complex<double>> mps1{1.0, 0.0};
    const std::vector<std::size_t> mpsExtent1{2, 1};
    const std::vector<std::complex<double>> mps2{1.0, 0.0};
    const std::vector<std::size_t> mpsExtent2{1, 2};
    cudaq::TensorStateData initData{{mps1.data(), mpsExtent1},
                                    {mps2.data(), mpsExtent2}};
    auto state = cudaq::state::from_data(initData);
    state.dump();
    std::vector<std::complex<double>> stateVec(4);
    state.to_host(stateVec.data(), stateVec.size());
    EXPECT_NEAR(std::abs(stateVec[0] - 1.0), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(stateVec[1]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(stateVec[2]), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(stateVec[3]), 0.0, 1e-12);
  }
  {
    const std::vector<std::complex<double>> mps1{1.0, 0.0};
    const std::vector<std::size_t> mpsExtent1{2, 1};
    const std::vector<std::complex<double>> mps2{1.0, 0.0};
    const std::vector<std::size_t> mpsExtent2{1, 2};
    cudaq::TensorStateData initData{{mps1.data(), mpsExtent1},
                                    {mps2.data(), mpsExtent2}};
    auto state = cudaq::state::from_data(initData);
    state.dump();
    auto state2 = cudaq::get_state(
        [](cudaq::state state) __qpu__ {
          cudaq::qvector q(state);
          h(q[0]);
          cx(q[0], q[1]);
        },
        state);
    state2.dump();
    EXPECT_NEAR(std::abs(state2.amplitude({0, 0})), M_SQRT1_2, 1e-6);
    EXPECT_NEAR(std::abs(state2.amplitude({1, 1})), M_SQRT1_2, 1e-6);
  }
  {
    constexpr int numQubits = 10;
    auto state1 = cudaq::get_state([]() __qpu__ {
      cudaq::qvector q(10);
      // First half of the circuit
      h(q[0]);
    });
    EXPECT_EQ(state1.get_num_tensors(), 10);
    cudaq::TensorStateData tensors;
    // Unpack the state to get the MPS tensors.
    for (const auto &tensor : state1.get_tensors())
      tensors.emplace_back(std::pair<const void *, std::vector<std::size_t>>{
          tensor.data, tensor.extents});
    auto reconstructedState = cudaq::state::from_data(tensors);
    // Second half of the bell circuit
    auto state2 = cudaq::get_state(
        [](cudaq::state state) __qpu__ {
          cudaq::qvector q(state);
          for (std::size_t i = 0; i < q.size() - 1; ++i)
            cx(q[i], q[i + 1]);
        },
        reconstructedState);
    const std::vector<int> allZero(numQubits, 0);
    const std::vector<int> allOne(numQubits, 1);
    EXPECT_NEAR(std::abs(state2.amplitude(allZero)), M_SQRT1_2, 1e-6);
    EXPECT_NEAR(std::abs(state2.amplitude(allOne)), M_SQRT1_2, 1e-6);
    const auto overlap = state1.overlap(state2);
    EXPECT_NEAR(std::abs(overlap), 0.5, 1e-6);
  }
}
#endif
