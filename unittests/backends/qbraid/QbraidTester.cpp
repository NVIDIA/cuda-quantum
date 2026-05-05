/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "common/RestClient.h"
#include "cudaq/algorithm.h"
#include <fstream>
#include <gtest/gtest.h>
#include <stdlib.h>

// Update the backend string to match the QBraid format
std::string mockPort = "62454";
std::string backendStringTemplate =
    "qbraid;emulate;false;url;http://localhost:{}";

bool isValidExpVal(double value) {
  // The qbraid mock server doesn't simulate quantum mechanics - X0X1 counts
  // are uniform random per 1000-shot sample (std dev ~0.03), so the
  // expectation value for this VQE Hamiltonian fluctuates around -2.14 by
  // a few hundredths per run. The band below is wide enough (~10 sigma) to
  // be stable across test runs while still catching corrupt / NaN results.
  return value < -1.0 && value > -3.0;
}

CUDAQ_TEST(QbraidTester, checkSampleSync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QbraidTester, checkSampleAsync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QbraidTester, checkSampleAsyncLoadFromFile) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  {
    std::ofstream out("saveMe.json");
    out << future;
  }

  cudaq::async_result<cudaq::sample_result> readIn;
  std::ifstream in("saveMe.json");
  in >> readIn;

  auto counts = readIn.get();
  EXPECT_EQ(counts.size(), 2);

  std::remove("saveMe.json");
}

CUDAQ_TEST(QbraidTester, checkObserveSync) {
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

CUDAQ_TEST(QbraidTester, checkObserveAsync) {
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

CUDAQ_TEST(QbraidTester, checkObserveAsyncLoadFromFile) {
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

  cudaq::async_result<cudaq::observe_result> readIn(&h);
  std::ifstream in("saveMeObserve.json");
  in >> readIn;

  auto result = readIn.get();

  std::remove("saveMeObserve.json");
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

// Every test in this file runs through the backend configured by
// add_backend_unittest_executable in CMakeLists, which passes api_key via the
// target config (BACKEND_CONFIG). QBRAID_API_KEY env var is NOT set by the
// launch script, so a successful sample here exercises the target-arg path.
CUDAQ_TEST(QbraidTester, checkApiKeyFromTarget) {
  ASSERT_EQ(std::getenv("QBRAID_API_KEY"), nullptr)
      << "QBRAID_API_KEY should not be set; this test verifies the "
         "api_key=... target-arg path.";

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  EXPECT_GE(counts.size(), 1u);
}

CUDAQ_TEST(QbraidTester, checkJobFailure) {
  // Arm the mock to fail the next submitted job.
  cudaq::RestClient client;
  nlohmann::json body = nlohmann::json::object();
  std::map<std::string, std::string> headers;
  auto armed = client.post("http://localhost:62454/", "test/fail_next", body,
                           headers, /*enableLogging=*/false);
  ASSERT_TRUE(armed.value("armed", false));

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  EXPECT_ANY_THROW({ (void)cudaq::sample(kernel); });
}

// Arm the mock to make the next N /result calls return "not yet available",
// so processResults must retry. maxRetries is 3, so 2 delays should succeed.
CUDAQ_TEST(QbraidTester, checkResultRetry) {
  cudaq::RestClient client;
  nlohmann::json body = nlohmann::json::object();
  std::map<std::string, std::string> headers;
  auto armed =
      client.post("http://localhost:62454/", "test/delay_next_results/2", body,
                  headers, /*enableLogging=*/false);
  ASSERT_EQ(armed.value("remaining", -1), 2);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  EXPECT_GE(counts.size(), 1u);
}

// Arm enough delays to exhaust the retry budget (maxRetries = 3). Sample must
// throw. Uses 10 so the retry loop can never succeed.
CUDAQ_TEST(QbraidTester, checkResultRetryExhaustion) {
  cudaq::RestClient client;
  nlohmann::json body = nlohmann::json::object();
  std::map<std::string, std::string> headers;
  auto armed =
      client.post("http://localhost:62454/", "test/delay_next_results/10", body,
                  headers, /*enableLogging=*/false);
  ASSERT_EQ(armed.value("remaining", -1), 10);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  EXPECT_ANY_THROW({ (void)cudaq::sample(kernel); });
}

// Helper: arm the mock to return a specific HTTP status on the next /result.
// Resets prior test-hook state first so the test is order-independent.
static void armResultStatus(int code) {
  cudaq::RestClient client;
  nlohmann::json body = nlohmann::json::object();
  std::map<std::string, std::string> headers;
  (void)client.post("http://localhost:62454/", "test/reset", body, headers,
                    /*enableLogging=*/false);
  auto armed =
      client.post("http://localhost:62454/",
                  "test/force_next_result_status/" + std::to_string(code), body,
                  headers, /*enableLogging=*/false);
  ASSERT_EQ(armed.value("armed_status", -1), code);
}

// Helper: match a substring in the exception message.
static ::testing::AssertionResult throwsWithMessage(std::function<void()> fn,
                                                    const std::string &needle) {
  try {
    fn();
  } catch (const std::exception &e) {
    std::string what = e.what();
    if (what.find(needle) != std::string::npos)
      return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
           << "exception message did not contain '" << needle << "'. Actual: '"
           << what << "'";
  }
  return ::testing::AssertionFailure() << "expected exception, none thrown";
}

// 401 on /result -> terminal auth failure, message must name the status.
CUDAQ_TEST(QbraidTester, checkResultAuthFailure) {
  armResultStatus(401);
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  EXPECT_TRUE(throwsWithMessage([&]() { (void)cudaq::sample(kernel); },
                                "authentication failed"));
}

// 403 on /result -> same terminal auth failure translation as 401.
CUDAQ_TEST(QbraidTester, checkResultForbidden) {
  armResultStatus(403);
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  EXPECT_TRUE(throwsWithMessage([&]() { (void)cudaq::sample(kernel); },
                                "authentication failed"));
}

// 404 on /result -> terminal "not found", message must mention the job id.
CUDAQ_TEST(QbraidTester, checkResultNotFound) {
  armResultStatus(404);
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  EXPECT_TRUE(throwsWithMessage([&]() { (void)cudaq::sample(kernel); },
                                "result not found"));
}

// 409 on /result -> terminal. qBraid v2 returns this when the job reached a
// non-success terminal state (FAILED or CANCELLED), so results will never
// appear and the helper must fail fast instead of burning the retry budget.
CUDAQ_TEST(QbraidTester, checkResultConflict) {
  armResultStatus(409);
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  EXPECT_TRUE(throwsWithMessage([&]() { (void)cudaq::sample(kernel); },
                                "did not produce results"));
}

// 500 on /result -> retryable. Force hook fires once then clears, so the
// second attempt succeeds. Sampling must not throw.
CUDAQ_TEST(QbraidTester, checkResultServerErrorRetries) {
  armResultStatus(500);
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  auto counts = cudaq::sample(kernel);
  EXPECT_GE(counts.size(), 1u);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
