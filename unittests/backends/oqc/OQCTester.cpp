/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "cudaq/algorithm.h"
#include <fstream>
#include <gtest/gtest.h>

std::string mockPort = "62453";
std::string email = "test@test.com";
std::string password = "password";
std::string backendStringTemplate =
    "oqc;emulate;false;url;http://localhost:{};email;{};password;{}";

bool isValidExpVal(double value) {
  // give us some wiggle room while keep the tests fast
  return value < -1.1 && value > -2.3;
}

CUDAQ_TEST(OQCTester, checkSampleSync) {
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, email, password);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(OQCTester, checkSampleAsync) {
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, email, password);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(OQCTester, checkSampleAsyncLoadFromFile) {
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, email, password);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

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

CUDAQ_TEST(OQCTester, checkObserveSync) {
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, email, password);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  auto result = cudaq::observe(kernel, h, .59);
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(OQCTester, checkObserveAsync) {
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, email, password);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  auto future = cudaq::observe_async(kernel, h, .59);

  auto result = future.get();
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(OQCTester, checkObserveAsyncLoadFromFile) {
  auto backendString = fmt::format(fmt::runtime(backendStringTemplate),
                                   mockPort, email, password);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
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

int main(int argc, char **argv) {
  setenv("OQC_PASSWORD", password.c_str(), 0);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
