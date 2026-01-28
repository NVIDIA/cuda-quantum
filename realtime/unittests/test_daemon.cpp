/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "test_utils/loopback_channel.h"
#include "cudaq/nvqlink/daemon/daemon.h"
#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/serialization/output_stream.h"

#include <gtest/gtest.h>
#include <thread>

using namespace cudaq::nvqlink;
using namespace cudaq::nvqlink::test;

// Simple test functions
int add(int a, int b) { return a + b; }
int multiply(int a, int b) { return a * b; }
void increment(int &value) { value++; }

class DaemonTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Create loopback channel for testing
    channel = std::make_unique<LoopbackChannel>();
    channel->initialize();
  }

  void TearDown() override {
    if (channel)
      channel->cleanup();
  }

  std::unique_ptr<LoopbackChannel> channel;
};

TEST_F(DaemonTest, Construction) {
  DaemonConfig config;
  config.id = "test_daemon";
  config.datapath_mode = DatapathMode::CPU;
  config.compute.cpu_cores = {0};

  ASSERT_TRUE(config.is_valid());

  // Create daemon with loopback channel
  auto daemon = std::make_unique<Daemon>(config, std::move(channel));

  EXPECT_FALSE(daemon->is_running());
}

TEST_F(DaemonTest, RegisterFunction) {
  DaemonConfig config;
  config.id = "test_daemon";
  config.datapath_mode = DatapathMode::CPU;
  config.compute.cpu_cores = {0};

  auto daemon = std::make_unique<Daemon>(config, std::move(channel));

  // Register functions using the simplified API
  EXPECT_NO_THROW(daemon->register_function(NVQLINK_RPC_HANDLE(add)));
  EXPECT_NO_THROW(daemon->register_function(NVQLINK_RPC_HANDLE(multiply)));
}

TEST_F(DaemonTest, RegisterDuplicateFunction) {
  DaemonConfig config;
  config.id = "test_daemon";
  config.datapath_mode = DatapathMode::CPU;
  config.compute.cpu_cores = {0};

  auto daemon = std::make_unique<Daemon>(config, std::move(channel));

  // Register function once
  daemon->register_function(NVQLINK_RPC_HANDLE(add));

  // Registering the same function again should throw (hash collision)
  EXPECT_THROW(daemon->register_function(NVQLINK_RPC_HANDLE(add)),
               std::runtime_error);
}

TEST_F(DaemonTest, StartStop) {
  DaemonConfig config;
  config.id = "test_daemon";
  config.datapath_mode = DatapathMode::CPU;
  config.compute.cpu_cores = {0};

  auto daemon = std::make_unique<Daemon>(config, std::move(channel));
  daemon->register_function(NVQLINK_RPC_HANDLE(add));

  EXPECT_FALSE(daemon->is_running());

  daemon->start();
  EXPECT_TRUE(daemon->is_running());

  // Give it a moment to actually start
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  daemon->stop();
  EXPECT_FALSE(daemon->is_running());
}

TEST_F(DaemonTest, InvalidConfig) {
  DaemonConfig config;
  // Invalid: empty id
  config.datapath_mode = DatapathMode::CPU;
  config.compute.cpu_cores = {0};

  EXPECT_FALSE(config.is_valid());

  // Note: Daemon constructor may or may not validate config strictly
  // The is_valid() check is advisory - implementation may allow invalid configs
  // for testing or graceful degradation
}

TEST_F(DaemonTest, CPUModeNoCores) {
  DaemonConfig config;
  config.id = "test";
  config.datapath_mode = DatapathMode::CPU;
  // Invalid: no CPU cores specified for CPU mode

  EXPECT_FALSE(config.is_valid());
}

TEST_F(DaemonTest, DaemonConfigBuilder) {
  auto config = DaemonConfigBuilder()
                    .set_id("test_builder")
                    .set_datapath_mode(DatapathMode::CPU)
                    .set_cpu_cores({0, 1})
                    .build();

  EXPECT_EQ(config.id, "test_builder");
  EXPECT_EQ(config.datapath_mode, DatapathMode::CPU);
  EXPECT_EQ(config.compute.cpu_cores.size(), 2);
  EXPECT_TRUE(config.is_valid());
}

TEST_F(DaemonTest, GetStats) {
  DaemonConfig config;
  config.id = "test_daemon";
  config.datapath_mode = DatapathMode::CPU;
  config.compute.cpu_cores = {0};

  auto daemon = std::make_unique<Daemon>(config, std::move(channel));
  daemon->register_function(NVQLINK_RPC_HANDLE(add));

  // Get initial stats
  auto stats = daemon->get_stats();
  EXPECT_EQ(stats.packets_received, 0);
  EXPECT_EQ(stats.packets_sent, 0);
  EXPECT_EQ(stats.errors, 0);
}

// Note: Full end-to-end RPC tests would require:
// 1. Starting the daemon
// 2. Injecting properly formatted RPC packets via LoopbackChannel
// 3. Verifying responses
// These are more complex integration tests and may be added later
