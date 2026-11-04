/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *                                                                             *
 ******************************************************************************/

// Emulation needs a server helper for target compilation, but must not start
// remote connections as a side effect of constructing the target's executor.
#include "common/Executor.h"
#include "common/ServerHelper.h"
#include "cudaq/platform/qpu_utils.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {
int remoteConstructions = 0;
int remoteInitializations = 0;

class TestServerHelper : public cudaq::ServerHelper {
public:
  const std::string name() const override { return "executor-test"; }
  void initialize(cudaq::BackendConfig config) override {
    backendConfig = std::move(config);
  }
  cudaq::RestHeaders getHeaders() override { return {}; }
  cudaq::ServerJobPayload
  createJob(std::vector<cudaq::KernelExecution> &) override {
    return {};
  }
  std::string extractJobId(cudaq::ServerMessage &) override { return {}; }
  std::string constructGetJobPath(std::string &) override { return {}; }
  std::string constructGetJobPath(cudaq::ServerMessage &) override {
    return {};
  }
  bool jobIsDone(cudaq::ServerMessage &) override { return true; }
  cudaq::sample_result processResults(cudaq::ServerMessage &,
                                      std::string &) override {
    return {};
  }
};

class TestRemoteExecutor : public cudaq::Executor {
public:
  TestRemoteExecutor() { ++remoteConstructions; }
  void setServerHelper(cudaq::ServerHelper *helper) override {
    ++remoteInitializations;
    cudaq::Executor::setServerHelper(helper);
  }
};

cudaq::ServerHelper::RegistryType::Add<TestServerHelper>
    helperRegistration("executor-test");
cudaq::Executor::RegistryType::Add<TestRemoteExecutor>
    executorRegistration("executor-test");
cudaq::ServerHelper::RegistryType::Add<TestServerHelper>
    genericHelperRegistration("generic-executor-test");

class RemoteExecutorTest : public ::testing::Test {
protected:
  cudaq::owning_ptr<cudaq::ServerHelper> helper;
  std::unique_ptr<cudaq::Executor> executor;

  void SetUp() override {
    remoteConstructions = 0;
    remoteInitializations = 0;
  }

  void initialize(cudaq::BackendConfig config,
                  const std::string &name = "executor-test") {
    cudaq::detail::initServerHelperAndExecutor(name, config, {}, helper,
                                               executor);
    ASSERT_NE(helper, nullptr);
    ASSERT_NE(executor, nullptr);
    EXPECT_EQ(helper->getConfig(), config);
  }
};

TEST_F(RemoteExecutorTest, EmulationDoesNotConstructRemoteExecutor) {
  initialize({{"emulate", "true"}, {"machine", "test-device"}});
  EXPECT_EQ(remoteConstructions, 0);
  EXPECT_EQ(remoteInitializations, 0);
}

TEST_F(RemoteExecutorTest, RemoteExecutionUsesRegisteredExecutor) {
  initialize({{"emulate", "false"}});
  EXPECT_EQ(remoteConstructions, 1);
  EXPECT_EQ(remoteInitializations, 1);
}

TEST_F(RemoteExecutorTest, RemoteExecutionIsTheDefault) {
  initialize({});
  EXPECT_EQ(remoteConstructions, 1);
  EXPECT_EQ(remoteInitializations, 1);
}

TEST_F(RemoteExecutorTest, UnregisteredExecutorUsesGenericImplementation) {
  initialize({}, "generic-executor-test");
  EXPECT_EQ(remoteConstructions, 0);
  EXPECT_EQ(remoteInitializations, 0);
}
} // namespace
