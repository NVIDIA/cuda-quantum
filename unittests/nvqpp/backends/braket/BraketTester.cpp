/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/BraketServerHelper.h"
#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"
#include "cudaq/algorithm.h"
#include <fstream>
#include <gtest/gtest.h>

bool isValidExpVal(double value) {
  // give us some wiggle room while keep the tests fast
  return value < -1.1 && value > -2.3;
}

CUDAQ_TEST(BraketTester, checkSampleSync) {
  GTEST_SKIP() << "Amazon Braket credentials required";

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(BraketTester, checkSampleAsync) {
  GTEST_SKIP() << "Amazon Braket credentials required";

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(BraketTester, checkSampleAsyncLoadFromFile) {
  GTEST_SKIP() << "Fails with: Cannot persist a cudaq::future for a local "
                  "kernel execution.";

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

CUDAQ_TEST(BraketTester, setOutputNamesKeyMatchesProcessResultsJobID) {
  // Verify that the key used by setOutputNames (the task ARN string) is the
  // same key used by processResults when looking up the result map.  In
  // production, BraketExecutor calls setOutputNames(taskArn, ...) and then
  // processResults(..., taskArn), so the two keys must be identical.
  auto rawHelper = cudaq::registry::get<cudaq::ServerHelper>("braket");
  ASSERT_TRUE(rawHelper);
  auto *braketHelper =
      dynamic_cast<cudaq::BraketServerHelper *>(rawHelper.get());
  ASSERT_TRUE(braketHelper);

  cudaq::BackendConfig config;
  config["emulate"] = "true";
  braketHelper->initialize(config);

  const std::string taskArn =
      "arn:aws:braket:us-east-1:123:quantum-task/test-task-id";
  // Each output-location tuple is [qubitNum, registerName, outputPosition].
  braketHelper->setOutputNames(
      taskArn,
      R"([[[0, [1, "r00000", 0]], [1, [2, "r00001", 1]], [2, [0, "r00002", 2]]]])");

  cudaq::ServerMessage response = {{"measurementProbabilities", {{"100", 1.0}}},
                                   {"taskMetadata", {{"shots", 1000}}}};

  std::string jobIdCopy = taskArn;
  auto result = braketHelper->processResults(response, jobIdCopy);

  // If the keying were wrong, processResults would return the raw counts
  // instead of the reordered ones.  A successful reorder here confirms the keys
  // match.
  EXPECT_EQ(result.count("001"), 1000);
  EXPECT_EQ(result.count("1", "r00002"), 1000);
}

CUDAQ_TEST(BraketTester, checkObserveSync) {
  GTEST_SKIP() << "Fails with: Cannot observe kernel with measures in it";

  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);
  kernel.mz(qubit);

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  auto result = cudaq::observe(10000, kernel, h, .59);
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(BraketTester, checkObserveAsync) {
  GTEST_SKIP() << "Fails with: Device requires all qubits in the program to be "
                  "measured. This may be caused by declaring non-contiguous "
                  "qubits or measuring partial qubits";

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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
