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
#include <cstdlib>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

bool result_maps_are_matching(
    const std::unordered_map<std::string, std::size_t> &results,
    const std::unordered_map<std::string, std::size_t> &expected) {
  for (const auto &[key, value] : expected) {
    auto it = results.find(key);
    if (it == results.end() || it->second != value) {
      return false;
    }
  }
  return true;
}

CUDAQ_TEST(TiiTester, checkSimpleCircuitSync) {
  // Create a simple circuit
  auto kernel = cudaq::make_kernel();
  auto qubits = kernel.qalloc(2);
  kernel.h(qubits[0]);
  kernel.x<cudaq::ctrl>(qubits[0], qubits[1]);
  kernel.mz(qubits);

  // Execute the circuit
  auto counts = cudaq::sample(kernel);
  counts.dump();
  // Check results
  EXPECT_EQ(counts.size(), 2);
  std::unordered_map<std::string, std::size_t> expected = {{"00", 500},
                                                           {"11", 500}};
  EXPECT_TRUE(result_maps_are_matching(counts.to_map(), expected));
}

CUDAQ_TEST(TiiTester, checkSimpleCircuitAsync) {
  // Create a simple circuit
  auto kernel = cudaq::make_kernel();
  auto qubits = kernel.qalloc(2);
  kernel.h(qubits[0]);
  kernel.x<cudaq::ctrl>(qubits[0], qubits[1]);
  kernel.mz(qubits);

  // Execute the circuit
  auto counts = cudaq::sample_async(kernel).get();
  counts.dump();
  // Check results
  EXPECT_EQ(counts.size(), 2);
  std::unordered_map<std::string, std::size_t> expected = {{"00", 500},
                                                           {"11", 500}};
  EXPECT_TRUE(result_maps_are_matching(counts.to_map(), expected));
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
} // namespace

// The tests below drive the helper through the registry to pin how
// `initialize` resolves the API token and URL. No network access happens
// during `initialize`.
CUDAQ_TEST(TiiTester, checkMissingTokenThrows) {
  ScopedEnv env("TII_API_TOKEN", nullptr);
  auto helper = cudaq::registry::get<cudaq::ServerHelper>("tii");
  ASSERT_TRUE(helper);
  cudaq::BackendConfig config{{"url", "http://localhost:1"}};
  try {
    helper->initialize(config);
    FAIL() << "expected an exception";
  } catch (const std::runtime_error &e) {
    EXPECT_STREQ(e.what(), "TII_API_TOKEN environment variable is not set.");
  }
}

CUDAQ_TEST(TiiTester, checkTokenAndUrlPrecedence) {
  ScopedEnv token("TII_API_TOKEN", "env-token");
  ScopedEnv url("TII_API_URL", "http://env.invalid:1");
  auto helper = cudaq::registry::get<cudaq::ServerHelper>("tii");
  ASSERT_TRUE(helper);

  // Nothing in the config: both come from the environment; the URL gets a
  // trailing slash.
  helper->initialize({});
  auto config = helper->getConfig();
  EXPECT_EQ(config["api_key"], "env-token");
  EXPECT_EQ(config["url"], "http://env.invalid:1/");

  // Values in the config take precedence over the environment.
  helper->initialize(
      {{"api_key", "cfg-token"}, {"url", "http://cfg.invalid:1"}});
  config = helper->getConfig();
  EXPECT_EQ(config["api_key"], "cfg-token");
  EXPECT_EQ(config["url"], "http://cfg.invalid:1/");
}

CUDAQ_TEST(TiiTester, checkDefaultUrlWhenEnvUnset) {
  ScopedEnv url("TII_API_URL", nullptr);
  auto helper = cudaq::registry::get<cudaq::ServerHelper>("tii");
  ASSERT_TRUE(helper);
  helper->initialize({{"api_key", "cfg-token"}});
  EXPECT_EQ(helper->getConfig()["url"], "https://q-cloud.tii.ae/");
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
