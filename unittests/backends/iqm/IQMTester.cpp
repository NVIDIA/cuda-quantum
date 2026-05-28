
/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 IQM Quantum Computers                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CUDAQTestUtils.h"
#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"
#include "cudaq/algorithm.h"

CUDAQ_TEST(IQMTester, executeOneMeasuredQubitProgram) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.y(qubit[0]);
  kernel.z(qubit[0]);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(IQMTester, executeSeveralMeasuredQubitProgram) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  kernel.mz(qubit[1]);

  auto counts = cudaq::sample(kernel);
  EXPECT_GE(counts.size(), 2);
  EXPECT_LE(counts.size(), 4);
}

CUDAQ_TEST(IQMTester, executeLoopOverQubitsProgram) {
  auto N = 5;
  auto kernel = cudaq::make_kernel();

  auto qubit = kernel.qalloc(N);
  kernel.h(qubit[0]);

  kernel.for_loop(
      0, N - 1, [&](auto i) { kernel.x<cudaq::ctrl>(qubit[i], qubit[i + 1]); });

  kernel.mz(qubit[0]);
  auto counts = cudaq::sample(kernel);

  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(IQMTester, executeMultipleMeasuredQubitsProgram) {
  auto N = 2;
  auto kernel = cudaq::make_kernel();

  auto qubit = kernel.qalloc(N);
  kernel.h(qubit[0]);
  kernel.x<cudaq::ctrl>(qubit[0], qubit[1]);

  kernel.mz(qubit);

  auto counts = cudaq::sample(kernel);
  EXPECT_GE(counts.size(), 2);
  EXPECT_LE(counts.size(), 4);
}

// Regression test for https://github.com/NVIDIA/cuda-quantum/issues/4621
// The IQM server returns counts ordered by physical qubit. When the
// qubit-mapping pass picks a non-identity virtual->physical placement, the
// helper must permute the bitstring back to the user's virtual qubit
// allocation order. That permutation is communicated to the server helper
// through `reorderIdx.<jobId>` entries in the backend config, which
// `parseConfigForCommonParams` parses into `ServerHelper::reorderIdx`. If
// `initialize()` forgets to call that, the helper returns physical-ordered
// bitstrings (the bug).
CUDAQ_TEST(IQMTester, processResultsAppliesReorderIdxFromConfig) {
  auto serverHelper = cudaq::registry::get<cudaq::ServerHelper>("iqm");
  ASSERT_TRUE(serverHelper);

  const std::string jobId = "test-job-id";
  // sample_result::reorder semantics: newBits[i] = oldBits[idx[i]].
  // With idx = [1, 2, 0] and physical bitstring "100", the reordered
  // (virtual-order) bitstring is "001".
  cudaq::BackendConfig config;
  config["url"] = "http://localhost:62443";
  config["reorderIdx." + jobId] = "[1, 2, 0]";
  serverHelper->initialize(config);

  cudaq::ServerMessage response = {
      {"status", "ready"},
      {"counts_batch",
       cudaq::ServerMessage::array(
           {{{"counts", {{"100", 1000}}}, {"measurement_keys", {"m_0"}}}})}};

  std::string jobIdCopy = jobId;
  auto result = serverHelper->processResults(response, jobIdCopy);

  // After reordering, the dominant bitstring must be "001".
  std::string mostFrequent;
  std::size_t bestCount = 0;
  for (auto &[bits, n] : result.to_map()) {
    if (n > bestCount) {
      bestCount = n;
      mostFrequent = bits;
    }
  }
  EXPECT_EQ(mostFrequent, "001")
      << "IQMServerHelper did not apply reorderIdx from backend config "
      << "(see GitHub issue #4621). Got \"" << mostFrequent << "\".";
}

// Setting an arbitrary string in the IQM_TOKEN environment variable must
// trigger the response that the authentication failed.
CUDAQ_TEST(IQMTester, invalidTokenFromEnvVariable) {
  char *token = getenv("IQM_TOKEN");

  EXPECT_THAT(
      []() {
        setenv("IQM_TOKEN", "invalid-invalid-invalid", true);
        auto kernel = cudaq::make_kernel();
        auto qubit = kernel.qalloc(1);
        kernel.h(qubit[0]);
        kernel.mz(qubit[0]);
        cudaq::sample(kernel);
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("HTTP GET Error - status code 401")));

  if (token) {
    setenv("IQM_TOKEN", token, true);
  } else {
    unsetenv("IQM_TOKEN");
  }
}

CUDAQ_TEST(IQMTester, iqmServerUrlEnvOverride) {
  char *url = getenv("IQM_SERVER_URL");

  EXPECT_THAT(
      []() {
        setenv("IQM_SERVER_URL", "fake-fake-fake", true);
        auto kernel = cudaq::make_kernel();
        auto qubit = kernel.qalloc(1);
        kernel.h(qubit[0]);
        kernel.mz(qubit[0]);
        cudaq::sample(kernel);
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Could not resolve host: fake-fake-fake")));

  if (url) {
    setenv("IQM_SERVER_URL", url, true);
  } else {
    unsetenv("IQM_SERVER_URL");
  }
}

// Without the IQM_TOKEN environment variable the fallback is to check the
// file pointed to by the IQM_TOKENS_FILE environment variable for tokens.
// If this does not exist an error will be thrown.
CUDAQ_TEST(IQMTester, tokenFilePathEnvOverride) {
  char *token = getenv("IQM_TOKEN");
  char *tfile = getenv("IQM_TOKENS_FILE");

  EXPECT_THAT(
      []() {
        unsetenv("IQM_TOKEN");
        setenv("IQM_TOKENS_FILE", "fake-fake-fake", true);
        auto kernel = cudaq::make_kernel();
        auto qubit = kernel.qalloc(1);
        kernel.h(qubit[0]);
        kernel.mz(qubit[0]);
        cudaq::sample(kernel);
      },
      testing::ThrowsMessage<std::runtime_error>(
          testing::HasSubstr("Unable to open tokens file: fake-fake-fake")));

  if (token) {
    setenv("IQM_TOKEN", token, true);
  } else {
    unsetenv("IQM_TOKEN");
  }
  if (tfile) {
    setenv("IQM_TOKENS_FILE", tfile, true);
  } else {
    unsetenv("IQM_TOKENS_FILE");
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
