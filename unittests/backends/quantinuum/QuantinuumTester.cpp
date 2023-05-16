/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"
#include "common/FmtCore.h"
#include <fstream>
#include <gtest/gtest.h>

std::string mockPort = "62454";
std::string backendStringTemplate =
    "quantinuum;url;http://localhost:{};credentials;{}";

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
  auto result = cudaq::observe(kernel, h, .59);
  result.dump();

  printf("ENERGY: %lf\n", result.exp_val_z());
  EXPECT_NEAR(result.exp_val_z(), -1.7, 1e-1);
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

  printf("ENERGY: %lf\n", result.exp_val_z());
  EXPECT_NEAR(result.exp_val_z(), -1.7, 1e-1);
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

  printf("ENERGY: %lf\n", result.exp_val_z());
  EXPECT_NEAR(result.exp_val_z(), -1.7, 1e-1);
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
