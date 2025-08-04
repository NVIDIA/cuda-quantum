
/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cudaq/algorithm.h"

#include <fstream>
#include <regex>

std::string backendString = "iqm;emulate;false;url;"
                            "http://localhost:62443";

CUDAQ_TEST(IQMTester, executeOneMeasuredQubitProgram) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

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
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

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
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

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
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

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

  EXPECT_THAT(
      []() {
        setenv("IQM_TOKEN", "invalid-invalid-invalid", true);
        auto &platform = cudaq::get_platform();
        platform.setTargetBackend(backendString);
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
        auto &platform = cudaq::get_platform();
        platform.setTargetBackend(backendString);
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
        auto &platform = cudaq::get_platform();
        platform.setTargetBackend(backendString);
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
