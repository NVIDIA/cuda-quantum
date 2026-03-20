/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Tests that cc::IfOp::getSuccessorRegions and getRegionInvocationBounds
// correctly report only the live region when the condition is a constant.

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <gtest/gtest.h>

using namespace mlir;

struct IfOpConstantFoldTest : public ::testing::Test {
  IfOpConstantFoldTest() {
    context.loadDialect<cudaq::cc::CCDialect, arith::ArithDialect>();
  }

  // Build a cc.if with the given constant bool condition and (optionally) an
  // else region, then return the op.
  cudaq::cc::IfOp buildIfOp(OpBuilder &builder, bool condValue, bool withElse) {
    auto loc = builder.getUnknownLoc();
    Value cond = builder.create<arith::ConstantIntOp>(loc, condValue ? 1 : 0,
                                                      /*width=*/1);
    auto regionBuilder = [&](OpBuilder &b, Location l, Region &region) {
      auto *block = b.createBlock(&region);
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToEnd(block);
      b.create<cudaq::cc::ContinueOp>(l);
    };
    return builder.create<cudaq::cc::IfOp>(
        loc, TypeRange{}, cond,
        /*thenBuilder=*/regionBuilder,
        /*elseBuilder=*/
        withElse ? std::function<void(OpBuilder &, Location, Region &)>(
                       regionBuilder)
                 : std::function<void(OpBuilder &, Location, Region &)>{});
  }

  MLIRContext context;
};

// ── getSuccessorRegions ──────────────────────────────────────────────────────

TEST_F(IfOpConstantFoldTest, SuccessorRegions_ConstTrue_OnlyThen) {
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  auto ifOp = buildIfOp(builder, /*condValue=*/true, /*withElse=*/true);

  // Simulate what the canonicalizer does: fold the condition to a constant.
  SmallVector<Attribute> operands = {IntegerAttr::get(builder.getI1Type(), 1)};

  SmallVector<RegionSuccessor> successors;
  ifOp.getSuccessorRegions(std::nullopt, operands, successors);

  ASSERT_EQ(successors.size(), 1u);
  EXPECT_EQ(successors[0].getSuccessor(), &ifOp.getThenRegion());
}

TEST_F(IfOpConstantFoldTest, SuccessorRegions_ConstFalse_OnlyElse) {
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  auto ifOp = buildIfOp(builder, /*condValue=*/false, /*withElse=*/true);

  SmallVector<Attribute> operands = {IntegerAttr::get(builder.getI1Type(), 0)};

  SmallVector<RegionSuccessor> successors;
  ifOp.getSuccessorRegions(std::nullopt, operands, successors);

  ASSERT_EQ(successors.size(), 1u);
  EXPECT_EQ(successors[0].getSuccessor(), &ifOp.getElseRegion());
}

TEST_F(IfOpConstantFoldTest, SuccessorRegions_NonConst_BothRegions) {
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  auto ifOp = buildIfOp(builder, /*condValue=*/true, /*withElse=*/true);

  // No constant operand — both regions are potential successors.
  SmallVector<Attribute> operands = {Attribute{}};

  SmallVector<RegionSuccessor> successors;
  ifOp.getSuccessorRegions(std::nullopt, operands, successors);

  EXPECT_EQ(successors.size(), 2u);
}

// ── getRegionInvocationBounds ────────────────────────────────────────────────

TEST_F(IfOpConstantFoldTest, InvocationBounds_ConstTrue_ThenOneElseZero) {
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  auto ifOp = buildIfOp(builder, /*condValue=*/true, /*withElse=*/true);

  SmallVector<Attribute> operands = {IntegerAttr::get(builder.getI1Type(), 1)};

  SmallVector<InvocationBounds> bounds;
  ifOp.getRegionInvocationBounds(operands, bounds);

  ASSERT_EQ(bounds.size(), 2u);
  // Then-region: always executes exactly once.
  EXPECT_EQ(bounds[0].getLowerBound(), 1u);
  EXPECT_EQ(bounds[0].getUpperBound(), std::optional<unsigned>(1));
  // Else-region: never executes.
  EXPECT_EQ(bounds[1].getLowerBound(), 0u);
  EXPECT_EQ(bounds[1].getUpperBound(), std::optional<unsigned>(0));
}

TEST_F(IfOpConstantFoldTest, InvocationBounds_ConstFalse_ThenZeroElseOne) {
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  auto ifOp = buildIfOp(builder, /*condValue=*/false, /*withElse=*/true);

  SmallVector<Attribute> operands = {IntegerAttr::get(builder.getI1Type(), 0)};

  SmallVector<InvocationBounds> bounds;
  ifOp.getRegionInvocationBounds(operands, bounds);

  ASSERT_EQ(bounds.size(), 2u);
  // Then-region: never executes.
  EXPECT_EQ(bounds[0].getLowerBound(), 0u);
  EXPECT_EQ(bounds[0].getUpperBound(), std::optional<unsigned>(0));
  // Else-region: always executes exactly once.
  EXPECT_EQ(bounds[1].getLowerBound(), 1u);
  EXPECT_EQ(bounds[1].getUpperBound(), std::optional<unsigned>(1));
}

TEST_F(IfOpConstantFoldTest, InvocationBounds_NonConst_BothZeroOrOne) {
  OpBuilder builder(&context);
  auto module = builder.create<ModuleOp>(builder.getUnknownLoc());
  builder.setInsertionPointToStart(module.getBody());
  auto ifOp = buildIfOp(builder, /*condValue=*/true, /*withElse=*/true);

  SmallVector<Attribute> operands = {Attribute{}};

  SmallVector<InvocationBounds> bounds;
  ifOp.getRegionInvocationBounds(operands, bounds);

  ASSERT_EQ(bounds.size(), 2u);
  EXPECT_EQ(bounds[0].getLowerBound(), 0u);
  EXPECT_EQ(bounds[0].getUpperBound(), std::optional<unsigned>(1));
  EXPECT_EQ(bounds[1].getLowerBound(), 0u);
  EXPECT_EQ(bounds[1].getUpperBound(), std::optional<unsigned>(1));
}
