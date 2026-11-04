/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/LaunchContext.h"
#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>

using cudaq::LaunchContext;
using cudaq::detail::claimLaunchContext;
using cudaq::detail::LaunchContextHandoff;

TEST(LaunchContextTester, claimTakesTheHandedOverContextOnce) {
  LaunchContext ctx;
  ctx.qpuId = 3;
  LaunchContextHandoff handoff{ctx};
  EXPECT_EQ(claimLaunchContext(), &ctx);
  EXPECT_EQ(claimLaunchContext(), nullptr);
}

TEST(LaunchContextTester, claimWithoutHandoffReturnsNull) {
  EXPECT_EQ(claimLaunchContext(), nullptr);
}

TEST(LaunchContextTester, unclaimedHandoffIsWithdrawnOnException) {
  LaunchContext ctx;
  EXPECT_THROW(
      {
        LaunchContextHandoff handoff{ctx};
        throw std::runtime_error("kernel failed before its launch");
      },
      std::runtime_error);
  EXPECT_EQ(claimLaunchContext(), nullptr);
}

TEST(LaunchContextTester, nestedHandoffIsRejected) {
  LaunchContext outer, inner;
  LaunchContextHandoff handoff{outer};
  EXPECT_THROW(
      { [[maybe_unused]] LaunchContextHandoff nested{inner}; },
      std::logic_error);
  // The outer launch context is still pending.
  EXPECT_EQ(claimLaunchContext(), &outer);
}

TEST(LaunchContextTester, handoffAfterClaimIsAccepted) {
  LaunchContext outer, inner;
  LaunchContextHandoff handoff{outer};
  ASSERT_EQ(claimLaunchContext(), &outer);
  // A launch made from within the claimed launch hands over its own context.
  {
    LaunchContextHandoff innerHandoff{inner};
    EXPECT_EQ(claimLaunchContext(), &inner);
  }
  EXPECT_EQ(claimLaunchContext(), nullptr);
}

TEST(LaunchContextTester, handoffIsThreadLocal) {
  LaunchContext ctx;
  LaunchContextHandoff handoff{ctx};
  LaunchContext *claimedElsewhere = &ctx;
  std::thread([&] { claimedElsewhere = claimLaunchContext(); }).join();
  EXPECT_EQ(claimedElsewhere, nullptr);
  EXPECT_EQ(claimLaunchContext(), &ctx);
}
