/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/InvocationContext.h"
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <thread>

using namespace cudaq;

TEST(InvocationContextTester, claimTakesTheHandedOverContextOnce) {
  InvocationContext ctx;
  ctx.launchOptions.qpuId = 3;
  auto live = pushInvocationContext(ctx);
  EXPECT_EQ(popInvocationContext()->launchOptions.qpuId, 3);
  EXPECT_EQ(popInvocationContext(), std::nullopt);
}

TEST(InvocationContextTester, claimWithoutHandoffReturnsNull) {
  EXPECT_EQ(popInvocationContext(), std::nullopt);
}

TEST(InvocationContextTester, unclaimedHandoffIsWithdrawnOnException) {
  InvocationContext ctx;
  EXPECT_THROW(
      {
        auto live = pushInvocationContext(ctx);
        throw std::runtime_error("kernel failed before its launch");
      },
      std::runtime_error);
  EXPECT_EQ(popInvocationContext(), std::nullopt);
}

TEST(InvocationContextTester, nestedHandoffIsRejected) {
  std::size_t outerQpuId = 99;
  std::size_t innerQpuId = 123;
  InvocationContext outer, inner;
  outer.launchOptions.qpuId = outerQpuId;
  inner.launchOptions.qpuId = innerQpuId;
  auto live = pushInvocationContext(outer);
  EXPECT_THROW(
      { [[maybe_unused]] auto live = pushInvocationContext(inner); },
      std::logic_error);
  // The outer launch context is still pending.
  EXPECT_EQ(popInvocationContext()->launchOptions.qpuId, outerQpuId);
}

TEST(InvocationContextTester, handoffAfterClaimIsAccepted) {
  std::size_t outerQpuId = 99;
  std::size_t innerQpuId = 123;
  InvocationContext outer, inner;
  outer.launchOptions.qpuId = outerQpuId;
  inner.launchOptions.qpuId = innerQpuId;
  auto live = pushInvocationContext(outer);
  ASSERT_EQ(popInvocationContext()->launchOptions.qpuId, outerQpuId);
  // A launch made from within the claimed launch hands over its own context.
  {
    auto live = pushInvocationContext(inner);
    EXPECT_EQ(popInvocationContext()->launchOptions.qpuId, innerQpuId);
  }
  EXPECT_EQ(popInvocationContext(), std::nullopt);
}

TEST(InvocationContextTester, handoffIsThreadLocal) {
  InvocationContext ctx;
  auto live = pushInvocationContext(ctx);
  std::optional<InvocationContext> claimedElsewhere = ctx;
  std::thread([&] { claimedElsewhere = popInvocationContext(); }).join();
  EXPECT_EQ(claimedElsewhere, std::nullopt);
  EXPECT_TRUE(popInvocationContext().has_value());
}
