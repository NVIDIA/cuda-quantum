/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
#include <regex>

std::string mockPort = "62440";
std::string backendStringTemplate =
    "quantinuum;emulate;false;url;http://localhost:{};credentials;{}";

bool isValidExpVal(double value) {
  // give us some wiggle room while keep the tests fast
  return value < -1.1 && value > -2.3;
}

CUDAQ_TEST(QuantinuumTester, checkSampleSync) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

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

CUDAQ_TEST(QuantinuumTester, checkU3Lowering) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    u3(3.14159, 1.5709, 0.78539, q);
  };

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QuantinuumTester, checkSampleSyncEmulate) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);
  backendString =
      std::regex_replace(backendString, std::regex("false"), "true");

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.x<cudaq::ctrl>(qubit[0], qubit[1]);
  kernel.mz(qubit[0]);
  kernel.mz(qubit[1]);

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QuantinuumTester, checkSampleAsync) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

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

CUDAQ_TEST(QuantinuumTester, checkSampleAsyncEmulate) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);
  backendString =
      std::regex_replace(backendString, std::regex("false"), "true");

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QuantinuumTester, checkSampleAsyncLoadFromFile) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

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

CUDAQ_TEST(QuantinuumTester, checkObserveSync) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

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
  auto result = cudaq::observe(10000, kernel, h, .59);
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(QuantinuumTester, checkObserveSyncEmulate) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);
  backendString =
      std::regex_replace(backendString, std::regex("false"), "true");

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
  auto result = cudaq::observe(100000, kernel, h, .59);
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(QuantinuumTester, checkObserveAsync) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

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

CUDAQ_TEST(QuantinuumTester, checkObserveAsyncEmulate) {
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);
  backendString =
      std::regex_replace(backendString, std::regex("false"), "true");

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
  auto future = cudaq::observe_async(100000, 0, kernel, h, .59);

  auto result = future.get();
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(QuantinuumTester, checkObserveAsyncLoadFromFile) {

  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

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

CUDAQ_TEST(QuantinuumTester, checkControlledRotations) {
  // Checks for more advanced controlled rotations that only
  // work in emulation.
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort, fileName);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

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
  std::string home = std::getenv("HOME");
  std::string fileName = home + "/FakeCppQuantinuum.config";
  std::ofstream out(fileName);
  out << "key: key\nrefresh: refresh\ntime: 0";
  out.close();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  std::remove(fileName.c_str());
  return ret;
}
