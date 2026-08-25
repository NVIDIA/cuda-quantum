/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

#include <cudaq.h>
#include <iostream>

TEST(MeasureResetTester, checkBug980) {
  auto foo = []() __qpu__ {
    cudaq::qubit a;
    cudaq::mz(a);
    cudaq::reset(a); // properly reset the qubit!
    cudaq::h(a);
    cudaq::mz(a);
  };

  auto bar = []() __qpu__ {
    cudaq::qubit a;
    cudaq::x(a);
    [[maybe_unused]] auto a0 = cudaq::mz(a);
    cudaq::reset(a); // properly reset the qubit!
    cudaq::h(a);
    [[maybe_unused]] auto a1 = cudaq::mz(a);
  };

  std::cout << "Foo:\n";
  auto result = cudaq::sample(foo);
  result.dump();
  EXPECT_EQ(2, result.size());
  EXPECT_TRUE(result.count("0") > 0);
  EXPECT_TRUE(result.count("1") > 0);

  std::cout << "Bar:\n";
  result = cudaq::sample(bar);
  result.dump();
  EXPECT_EQ(2, result.size());
  EXPECT_TRUE(result.count("0") > 0);
  EXPECT_TRUE(result.count("1") > 0);
}

TEST(MeasureResetTester, checkBug981) {

  auto bar = []() __qpu__ {
    cudaq::qubit a;
    cudaq::x(a);
    [[maybe_unused]] auto a0 = cudaq::mz(a);
    cudaq::reset(a);
    [[maybe_unused]] auto a1 = cudaq::mz(a);
  };

  std::cout << "Bar:\n";
  auto result = cudaq::sample(/*shots=*/10, bar);
  result.dump();
  EXPECT_EQ(1, result.size());
  EXPECT_TRUE(result.count("0") > 0);
}

TEST(MeasureResetTester, checkLibModeOrdering) {
  auto kernel = [](bool switchFlag) __qpu__ {
    cudaq::qubit a, b;
    x(a);
    if (switchFlag) {
      mz(b);
      mz(a);
    } else {
      mz(a);
      mz(b);
    }
  };

  // Bit string ordered according to qubit allocation order.
  auto counts = cudaq::sample(kernel, true);
  counts.dump();
  EXPECT_EQ("10", counts.begin()->first);
  counts = cudaq::sample(kernel, false);
  counts.dump();
  EXPECT_EQ("10", counts.begin()->first);
}

TEST(MeasureResetTester, checkMixedBasisOrderingAndPreservation) {
  constexpr std::size_t shots = 100;

  auto kernel = []() __qpu__ {
    cudaq::qvector q(7);

    // Prepare a non-palindromic deterministic pattern over measured bits.
    // q0=0 (mz), q1=1 (mz), q2=1 (mx), q3=? (my), q4=0 (mz), q5=0 (mx),
    // q6=1 (mz) -> 011?001 in allocation order.
    x(q[1]);
    x(q[2]);
    h(q[2]);
    h(q[5]);
    x(q[6]);

    // Mix measurement bases and execution order.
    mz(q[4]);
    mx(q[2]);
    my(q[3]);
    mz(q[0]);
    mx(q[5]);
    mz(q[6]);
    mz(q[1]);
  };

  auto counts = cudaq::sample(shots, kernel);
  std::size_t totalCounts = 0;
  for (const auto &[bits, count] : counts) {
    if (bits.size() != 7u) {
      ADD_FAILURE() << "Expected 7-bit string in default mode, got '" << bits
                    << "'";
      continue;
    }
    EXPECT_EQ(bits[0], '0');
    EXPECT_EQ(bits[1], '1');
    EXPECT_EQ(bits[2], '1');
    EXPECT_EQ(bits[4], '0');
    EXPECT_EQ(bits[5], '0');
    EXPECT_EQ(bits[6], '1');
    totalCounts += count;
  }
  EXPECT_EQ(totalCounts, shots);

  cudaq::sample_options options{};
  options.shots = shots;
  options.explicit_measurements = true;
  counts = cudaq::sample(options, kernel);
  totalCounts = 0;

  // Execution order was q4, q2, q3, q0, q5, q6, q1 => 01?0011.
  for (const auto &[bits, count] : counts) {
    if (bits.size() != 7u) {
      ADD_FAILURE() << "Expected 7-bit string in explicit mode, got '" << bits
                    << "'";
      continue;
    }
    EXPECT_EQ(bits[0], '0');
    EXPECT_EQ(bits[1], '1');
    EXPECT_EQ(bits[3], '0');
    EXPECT_EQ(bits[4], '0');
    EXPECT_EQ(bits[5], '1');
    EXPECT_EQ(bits[6], '1');
    totalCounts += count;
  }
  EXPECT_EQ(totalCounts, shots);
}
