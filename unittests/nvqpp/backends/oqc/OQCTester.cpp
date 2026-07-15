/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"
#include "cudaq/algorithm.h"
#include <fstream>
#include <gtest/gtest.h>

bool isValidExpVal(double value) {
  // give us some wiggle room while keep the tests fast
  return value < -1.1 && value > -2.3;
}

CUDAQ_TEST(OQCTester, checkSampleSync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(OQCTester, checkSampleAsync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(OQCTester, checkSampleAsyncLoadFromFile) {
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

// Provider-specific parsing: OQC returns counts under a "results" object with
// a sibling "task_error". Verify that processResults uses enriched output_names
// to reconstruct compact QIR results into user-visible terminal order.
// This matches the default-sampling convention, as validated in
// `test_explicit_measurements.py::test_measurement_order`:
// measured bits follow qubit allocation order, not `mz` execution order.
CUDAQ_TEST(OQCTester, processResultsParsesProviderResponse) {
  auto serverHelper = cudaq::registry::get<cudaq::ServerHelper>("oqc");
  ASSERT_TRUE(serverHelper);

  cudaq::BackendConfig config;
  config["emulate"] = "true";
  config["output_names.oqc-parse-job-id"] =
      R"([[[0, [2, "r00000", 2]], [1, [0, "r00001", 0]], [2, [1, "r00002", 1]]]])";
  serverHelper->initialize(config);

  cudaq::ServerMessage response = {{"results", {{"110", 1000}}},
                                   {"task_error", nullptr}};
  std::string jobId = "oqc-parse-job-id";
  auto result = serverHelper->processResults(response, jobId);

  EXPECT_EQ(result.count("101"), 1000);
  EXPECT_EQ(result.count("1", "r00000"), 1000);
  EXPECT_EQ(result.count("1", "r00001"), 1000);
  EXPECT_EQ(result.count("0", "r00002"), 1000);
}

// The OQC compilation of `x(q[0]); swap(q[0], q[2]); mz(q)` maps virtual
// qubits [0, 1, 2] to physical qubits [0, 2, 1]. The QIR results therefore
// retain physical qubit ids [0, 2, 1], while their enriched output positions
// recover the original virtual-qubit order [0, 1, 2]. Verify that the remote
// provider response path reconstructs the CUDA-Q bit string from that metadata
// without applying the legacy non-identity `reorderIdx` a second time.
CUDAQ_TEST(OQCTester, processResultsMappedTerminalMeasurements) {
  auto serverHelper = cudaq::registry::get<cudaq::ServerHelper>("oqc");
  ASSERT_TRUE(serverHelper);

  cudaq::BackendConfig config;
  config["entry_url"] = "http://localhost:62442";
  config["auth_token"] = "fake_auth_token";
  config["device"] = "qpu:uk:-1:1234567890";
  config["output_names.oqc-mapped-job-id"] =
      R"([[[0, [0, "r00000", 0]], [1, [2, "r00001", 1]], [2, [1, "r00002", 2]]]])";
  config["reorderIdx.oqc-mapped-job-id"] = R"([0, 2, 1])";
  serverHelper->initialize(config);

  cudaq::ServerMessage response = {{"results", {{"001", 1000}}},
                                   {"task_error", nullptr}};
  std::string jobId = "oqc-mapped-job-id";
  auto result = serverHelper->processResults(response, jobId);

  // Although `mapping_v2p` is [0, 2, 1], `mz(q)` assigns QIR result ids in
  // logical measurement order. The corrected output positions [0, 1, 2]
  // therefore preserve the raw provider count "001" as the global result.
  EXPECT_EQ(result.count("001"), 1000);
  EXPECT_EQ(result.count("0", "r00000"), 1000);
  EXPECT_EQ(result.count("0", "r00001"), 1000);
  EXPECT_EQ(result.count("1", "r00002"), 1000);
}

CUDAQ_TEST(OQCTester, processResultsRequiresOutputNames) {
  auto serverHelper = cudaq::registry::get<cudaq::ServerHelper>("oqc");
  ASSERT_TRUE(serverHelper);

  cudaq::BackendConfig config;
  config["emulate"] = "true";
  serverHelper->initialize(config);

  cudaq::ServerMessage response = {{"results", {{"0", 1}}},
                                   {"task_error", nullptr}};
  std::string jobId = "oqc-missing-output-names";

  EXPECT_ANY_THROW(serverHelper->processResults(response, jobId));
}

CUDAQ_TEST(OQCTester, checkObserveSync) {
  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  auto result = cudaq::observe(kernel, h, .59);
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(OQCTester, checkObserveAsync) {
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

CUDAQ_TEST(OQCTester, checkObserveAsyncLoadFromFile) {
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
