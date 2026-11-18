/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_cpu_roce_bridge_config.cpp
/// @brief Reject invalid CPU RoCE modes and ring geometry before NIC access.
///
/// Deliberately use a nonexistent RDMA device: argument errors must be reported
/// before transport setup, with no live bridge returned to the caller.
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

// Invalid configurations must fail before touching hardware. Loading by soname
// also exercises provider discovery through libcudaq-realtime's own RPATH.
void rejects(std::vector<std::string> options) {
  options.emplace_back("--device=not-a-real-rdma-device");
  options.emplace_back("--qp_config=hsb_fpga");
  options.emplace_back("--peer-ip=192.0.2.1");
  std::vector<char *> args;
  for (auto &option : options)
    args.push_back(option.data());
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  EXPECT_EQ(CUDAQ_ERR_INVALID_ARG,
            cudaq_bridge_create_from_library(
                &bridge, "libcudaq-realtime-bridge-cpu-roce.so",
                static_cast<int>(args.size()), args.data()));
  // A failed create must not expose a usable handle. Still clean it up if a
  // regression returns one, so later cases do not inherit leaked resources.
  EXPECT_EQ(nullptr, bridge);
  if (bridge)
    cudaq_bridge_destroy(bridge);
}

TEST(CpuRoceBridgeConfig, RejectsForwardWithUnified) {
  rejects({"--forward", "--unified"});
}

// HSB addressing needs a nonzero power-of-two slot count within its limit.
TEST(CpuRoceBridgeConfig, RejectsUnaddressableHsbRing) {
  rejects({"--num-slots=0"});
  rejects({"--num-slots=3"});
  rejects({"--num-slots=8192"});
}

TEST(CpuRoceBridgeConfig, RejectsFrameLargerThanSlot) {
  rejects({"--slot-size=32", "--frame-size=64"});
  rejects({"--slot-size=0"});
}

} // namespace
