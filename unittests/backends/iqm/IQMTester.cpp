
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

// Setting an arbitrary string in the IQM_TOKEN environment variable must
// trigger the response that the authentication failed.
CUDAQ_TEST(IQMTester, invalidTokenFromEnvVariable) {
  char *token = getenv("IQM_TOKEN");

  EXPECT_THAT([]() { setenv("IQM_TOKEN", "invalid-invalid-invalid", true); },
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

  EXPECT_THAT([]() { setenv("IQM_SERVER_URL", "fake-fake-fake", true); },
              testing::ThrowsMessage<std::runtime_error>(testing::HasSubstr(
                  "Could not resolve host: fake-fake-fake")));

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

CUDAQ_TEST(IQMTester, dynamicQuantumArchitectureFile) {
  const char dqa_filename[] = "dqa_mock_qpu_saved.txt";

  unlink(dqa_filename);

  // Test 1: saving dynamic quantum architecture to file
  setenv("IQM_SAVE_QPU_QA", dqa_filename, true);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  kernel.mz(qubit[1]);

  auto counts = cudaq::sample(kernel);

  unsetenv("IQM_SAVE_QPU_QA");

  EXPECT_GE(counts.size(), 2);
  EXPECT_LE(counts.size(), 4);

  // Test 2: use quantum architecture file referenced in environment variable
  setenv("IQM_QPU_QA", dqa_filename, true);

  auto kernel2 = cudaq::make_kernel();
  auto qubit2 = kernel2.qalloc(2);
  kernel2.h(qubit2[0]);
  kernel2.mz(qubit2[0]);
  kernel2.mz(qubit2[1]);

  counts = cudaq::sample(kernel2);

  unsetenv("IQM_QPU_QA");

  EXPECT_GE(counts.size(), 2);
  EXPECT_LE(counts.size(), 4);

  // When IQM tests can run, populate the dqa_mock_qpu.txt
  // with the content of this file.
  unlink(dqa_filename);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
