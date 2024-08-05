/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
