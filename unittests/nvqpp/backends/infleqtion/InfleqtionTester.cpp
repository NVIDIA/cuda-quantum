/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/ServerHelper.h"
#include "cudaq/algorithm.h"
#include <fstream>
#include <functional>
#include <gtest/gtest.h>
#include <stdexcept>
#include <stdlib.h>
#include <string>

bool isValidExpVal(double value) {
  // give us some wiggle room while keep the tests fast
  return value < -1.1 && value > -2.3;
}

CUDAQ_TEST(InfleqtionTester, checkSampleSync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(InfleqtionTester, checkSampleAsync) {
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto future = cudaq::sample_async(kernel);
  auto counts = future.get();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(InfleqtionTester, checkSampleAsyncLoadFromFile) {
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

namespace {
// Sets (value != nullptr) or unsets an environment variable for the lifetime
// of the object and restores the previous state afterwards.
struct ScopedEnv {
  ScopedEnv(const char *name, const char *value) : name(name) {
    if (const char *old = std::getenv(name)) {
      hadOld = true;
      oldValue = old;
    }
    if (value)
      ::setenv(name, value, 1);
    else
      ::unsetenv(name);
  }
  ~ScopedEnv() {
    if (hadOld)
      ::setenv(name, oldValue.c_str(), 1);
    else
      ::unsetenv(name);
  }
  const char *name;
  bool hadOld = false;
  std::string oldValue;
};

// Returns the message of the `std::runtime_error` thrown by `fn`, or
// "<no throw>".
std::string thrownMessage(const std::function<void()> &fn) {
  try {
    fn();
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "<no throw>";
}
} // namespace

// The tests below drive the helper through the registry to pin how
// `initialize` resolves the API token and config values. No network access
// happens during `initialize`.
CUDAQ_TEST(InfleqtionTester, checkMissingApiKeyThrows) {
  ScopedEnv env("SUPERSTAQ_API_KEY", nullptr);
  auto helper = cudaq::registry::get<cudaq::ServerHelper>("infleqtion");
  ASSERT_TRUE(helper);
  cudaq::BackendConfig config{{"emulate", "false"}};
  // Infleqtion has its own wording, distinct from the generic
  // "<name> environment variable is not set." used by other backends.
  EXPECT_EQ(thrownMessage([&]() { helper->initialize(config); }),
            "Environment variable SUPERSTAQ_API_KEY is required but not set.");
}

CUDAQ_TEST(InfleqtionTester, checkEmulateDoesNotRequireApiKey) {
  ScopedEnv env("SUPERSTAQ_API_KEY", nullptr);
  auto helper = cudaq::registry::get<cudaq::ServerHelper>("infleqtion");
  ASSERT_TRUE(helper);
  helper->initialize({{"emulate", "true"}});
  EXPECT_EQ(helper->getConfig()["token"], "0");
}

CUDAQ_TEST(InfleqtionTester, checkTokenPrecedenceAndConfigOverrides) {
  ScopedEnv env("SUPERSTAQ_API_KEY", "env-key");
  auto helper = cudaq::registry::get<cudaq::ServerHelper>("infleqtion");
  ASSERT_TRUE(helper);
  helper->initialize({{"emulate", "false"}});
  auto config = helper->getConfig();
  EXPECT_EQ(config["token"], "env-key");
  EXPECT_EQ(config["url"], "https://superstaq.infleqtion.com");
  EXPECT_EQ(config["machine"], "cq_sqale_simulator");

  // A token in the config takes precedence over the environment.
  helper->initialize({{"emulate", "false"},
                      {"token", "cfg-key"},
                      {"url", "http://localhost:1"},
                      {"machine", "cq_sqale_qpu"}});
  config = helper->getConfig();
  EXPECT_EQ(config["token"], "cfg-key");
  EXPECT_EQ(config["url"], "http://localhost:1");
  EXPECT_EQ(config["machine"], "cq_sqale_qpu");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
