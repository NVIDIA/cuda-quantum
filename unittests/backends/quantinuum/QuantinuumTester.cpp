/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"
#include <fstream>
#include <gtest/gtest.h>

bool isValidExpVal(double value) {
  // give us some wiggle room while keep the tests fast
  return value < -1.1 && value > -2.3;
}

CUDAQ_TEST(QuantinuumTester, checkSampleSync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QuantinuumTester, checkSampleAsync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QuantinuumTester, checkSampleAsyncLoadFromFile) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  // Can sample asynchronously and get a future
  auto future = cudaq::sample_async(kernel);

  // Future can be persisted for later
  {
    std::ofstream out("saveMe.json");
    out << future;
  }

  // Later you can come back and read it in
  cudaq::async_result<cudaq::sample_result> readIn;
  std::ifstream in("saveMe.json");
  in >> readIn;

  // Get the results of the read in future.
  auto counts = readIn.get();
  EXPECT_EQ(counts.size(), 2);

  std::remove("saveMe.json");
}

CUDAQ_TEST(QuantinuumTester, checkObserveSync) {
  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  auto result = cudaq::observe(10000, kernel, h, .59);
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(QuantinuumTester, checkObserveAsync) {
  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  auto future = cudaq::observe_async(kernel, h, .59);

  auto result = future.get();
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(QuantinuumTester, checkObserveAsyncLoadFromFile) {
  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  auto future = cudaq::observe_async(kernel, h, .59);

  {
    std::ofstream out("saveMeObserve.json");
    out << future;
  }

  // Later you can come back and read it in
  cudaq::async_result<cudaq::observe_result> readIn(&h);
  std::ifstream in("saveMeObserve.json");
  in >> readIn;

  // Get the results of the read in future.
  auto result = readIn.get();

  std::remove("saveMeObserve.json");
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(QuantinuumTester, checkControlledRotations) {
  // rx: pi
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.rx<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);

    std::cout << kernel.to_quake() << "\n";

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should've been rotated to |1>.
    EXPECT_EQ(counts.count("0000111111"), 1000);
  }

  // rx: 0.0
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.rx<cudaq::ctrl>(0.0, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should've stayed in |0>
    EXPECT_EQ(counts.count("0000111110"), 1000);
  }

  // ry: pi
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.ry<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should've been rotated to |1>
    EXPECT_EQ(counts.count("0000111111"), 1000);
  }

  // ry: pi / 2
  {
    cudaq::set_random_seed(4);

    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.ry<cudaq::ctrl>(M_PI_2, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();

    // Target qubit should have a 50/50 mix between |0> and |1>
    EXPECT_TRUE(counts.count("0000111111") < 550);
    EXPECT_TRUE(counts.count("0000111110") > 450);
  }

  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(3);
    auto controls2 = kernel.qalloc(3);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    kernel.x(controls1);
    kernel.x(control3);
    // Should do nothing.
    kernel.x<cudaq::ctrl>(controls1, controls2, control3, target);
    kernel.x(controls2);
    // Should rotate `target`.
    kernel.rx<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);

    auto counts = cudaq::sample(kernel);
    counts.dump();
    EXPECT_EQ(counts.count("00000011111111"), 1000);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
