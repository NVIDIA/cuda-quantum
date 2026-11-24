/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ConfigUtils.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
constexpr const char *kVar = "CUDAQ_CONFIG_UTILS_TEST_VAR";

// Sets/unsets the test variable and restores it unset on destruction.
struct EnvGuard {
  EnvGuard() { ::unsetenv(kVar); }
  ~EnvGuard() { ::unsetenv(kVar); }
  void set(const char *v) { ::setenv(kVar, v, 1); }
};

std::string messageOf(const std::string &key, bool required,
                      const std::string &msg = "") {
  try {
    cudaq::getEnvVar(key, "dflt", required, msg);
  } catch (const std::runtime_error &e) {
    return e.what();
  }
  return "<no throw>";
}
} // namespace

TEST(ConfigUtilsTester, envVarSet) {
  EnvGuard g;
  g.set("value");
  EXPECT_EQ(cudaq::getEnvVar(kVar, "dflt", false), "value");
  EXPECT_EQ(cudaq::getEnvVar(kVar, "dflt", true), "value");
}

TEST(ConfigUtilsTester, envVarUnsetUsesDefault) {
  EnvGuard g;
  EXPECT_EQ(cudaq::getEnvVar(kVar, "dflt", false), "dflt");
}

TEST(ConfigUtilsTester, envVarUnsetRequiredThrows) {
  EnvGuard g;
  EXPECT_THROW(cudaq::getEnvVar(kVar, "dflt", true), std::runtime_error);
  EXPECT_EQ(messageOf(kVar, true),
            std::string(kVar) + " environment variable is not set.");
}

TEST(ConfigUtilsTester, envVarCustomMessage) {
  EnvGuard g;
  EXPECT_EQ(messageOf(kVar, true, "custom message"), "custom message");
  // Not required: message is unused, default returned.
  EXPECT_EQ(messageOf(kVar, false, "custom message"), "<no throw>");
}

TEST(ConfigUtilsTester, envVarEmptyCountsAsSet) {
  EnvGuard g;
  g.set("");
  EXPECT_EQ(cudaq::getEnvVar(kVar, "dflt", false), "");
  EXPECT_EQ(cudaq::getEnvVar(kVar, "dflt", true), "");
}

TEST(ConfigUtilsTester, envVarNotTrimmed) {
  EnvGuard g;
  g.set("  padded \n");
  EXPECT_EQ(cudaq::getEnvVar(kVar, "dflt", false), "  padded \n");
}

TEST(ConfigUtilsTester, valueOrDefault) {
  std::map<std::string, std::string> config{{"present", "v"}, {"empty", ""}};
  EXPECT_EQ(cudaq::getValueOrDefault(config, "present", "d"), "v");
  EXPECT_EQ(cudaq::getValueOrDefault(config, "missing", "d"), "d");
  // A present-but-empty value is returned, not replaced by the default.
  EXPECT_EQ(cudaq::getValueOrDefault(config, "empty", "d"), "");
  // The config is not modified by lookups.
  EXPECT_EQ(config.size(), 2u);
}

TEST(ConfigUtilsTester, valueOrDefaultOtherMapTypes) {
  std::unordered_map<std::string, std::string> config{{"k", "v"}};
  EXPECT_EQ(cudaq::getValueOrDefault(config, "k", "d"), "v");
  EXPECT_EQ(cudaq::getValueOrDefault(config, "x", "d"), "d");
}
