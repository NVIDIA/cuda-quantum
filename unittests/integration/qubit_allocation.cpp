/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

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
  void operator()(cudaq::state &&state) __qpu__ {
    cudaq::qvector q(std::move(state));
    mz(q);
  }
};

CUDAQ_TEST(AllocationTester, checkAllocationFromRetrievedState) {
  auto bellState = cudaq::get_state([]() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    cx(q[0], q[1]);
  });
  auto counts =
      cudaq::sample(test_allocation_from_state{}, std::move(bellState));
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
  EXPECT_NEAR(std::abs(state1.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 1})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state1.amplitude({0, 1})), 0.0, 1e-9);
  const auto *ptr = state1.get_tensor().data;

  // Second half of the circuit
  auto state2 = cudaq::get_state(
      [](cudaq::state &&state) __qpu__ {
        cudaq::qvector q(std::move(state));
        cx(q[0], q[1]);
      },
      std::move(state1));
  EXPECT_NEAR(std::abs(state2.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state2.amplitude({1, 1})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state2.amplitude({1, 0})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state2.amplitude({0, 1})), 0.0, 1e-9);
  // Check that we're operating on the same piece of memory
  EXPECT_EQ(ptr, state2.get_tensor().data);
}

CUDAQ_TEST(AllocationTester, checkQvecAllocByRef) {
  auto state1 = cudaq::get_state([]() __qpu__ {
    cudaq::qvector q(2);
    // First half of the circuit
    h(q[0]);
  });
  EXPECT_NEAR(std::abs(state1.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 1})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state1.amplitude({0, 1})), 0.0, 1e-9);

  // Second half of the circuit
  // FIXME: how to run the circuit without a cudaq:: API?
  auto state2 = cudaq::get_state(
      [](cudaq::state &state) __qpu__ {
        cudaq::qvector q(state);
        cx(q[0], q[1]);
      },
      state1);
  // The original is updated...
  EXPECT_NEAR(std::abs(state1.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 1})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 0})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state1.amplitude({0, 1})), 0.0, 1e-9);
}

CUDAQ_TEST(AllocationTester, checkQvecAllocByConstRef) {
  auto state1 = cudaq::get_state([]() __qpu__ {
    cudaq::qvector q(2);
    // First half of the circuit
    h(q[0]);
  });
  EXPECT_NEAR(std::abs(state1.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 1})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state1.amplitude({0, 1})), 0.0, 1e-9);

  // Second half of the circuit
  // FIXME: how to run the circuit without a cudaq:: API?
  auto state2 = cudaq::get_state(
      [](const cudaq::state &state) __qpu__ {
        cudaq::qvector q(state);
        cx(q[0], q[1]);
      },
      state1);
  // The original stay the same
  EXPECT_NEAR(std::abs(state1.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state1.amplitude({1, 1})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state1.amplitude({0, 1})), 0.0, 1e-9);
  // The new state is returned
  EXPECT_NEAR(std::abs(state2.amplitude({0, 0})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state2.amplitude({1, 1})), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state2.amplitude({1, 0})), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(state2.amplitude({0, 1})), 0.0, 1e-9);
  // We can thus use the two states for computation, e.g., overlap
  const auto overlap = state1.overlap(state2);
  // Expected: 0.5
  EXPECT_NEAR(std::abs(overlap), 0.5, 1e-6);
}