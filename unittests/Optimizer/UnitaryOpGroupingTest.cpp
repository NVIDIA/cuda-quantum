/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Need to do
// - structs for handles to the ops of the different tests
// - helper functions to create the different test cases
// - helper functions to check the results of the different test cases

#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

struct SimpleHandles {
  mlir::func::FuncOp func;
  mlir::Operation *h = nullptr;
  mlir::Operation *x = nullptr;
  mlir::Operation *mz = nullptr;
  mlir::Operation *z = nullptr;
  mlir::Operation *constant = nullptr;
  mlir::Operation *rx = nullptr;
};

struct NestedIfHandles {
  func::FuncOp func;
  Operation *ifOp = nullptr;
  Operation *h = nullptr;
  Operation *x = nullptr;
  Operation *z = nullptr;
};

// Build the equivalent of:
//
//   func.func @simple(%q0: !quake.ref, %q1: !quake.ref, %theta: f64) attributes
//   {"cudaq-kernel"} {
//     quake.h %q0 : (!quake.ref) -> ()
//     quake.x %q1 : (!quake.ref) -> ()
//     %m = quake.mz %q0 : (!quake.ref) -> !cc.measure_handle
//     quake.z %q0 : (!quake.ref) -> ()
//     %c0 = arith.constant 0 : i64
//     quake.rx (%theta) %q1 : (f64, !quake.ref) -> ()
//     return
//   }
static SimpleHandles buildSimple(mlir::ModuleOp module) {
  OpBuilder builder(module.getContext());
  Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(module.getBody());

  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto funcTy =
      builder.getFunctionType({refTy, refTy, builder.getF64Type()}, {});

  auto func = func::FuncOp::create(builder, loc, "simple", funcTy);
  func->setAttr("cudaq-kernel", builder.getUnitAttr());

  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value q0 = entry->getArgument(0);
  Value q1 = entry->getArgument(1);
  Value theta = entry->getArgument(2);

  SimpleHandles handles;
  handles.func = func;
  handles.h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
  handles.x = cudaq::quake::XOp::create(builder, loc, q1).getOperation();

  auto measureTy = cudaq::cc::MeasureHandleType::get(module.getContext());
  handles.mz = cudaq::quake::MzOp::create(builder, loc, TypeRange{measureTy},
                                          ValueRange{q0}, StringAttr{})
                   .getOperation();

  handles.z = cudaq::quake::ZOp::create(builder, loc, q0).getOperation();
  handles.constant =
      arith::ConstantIntOp::create(builder, loc, 0, 64).getOperation();
  handles.rx = cudaq::quake::RxOp::create(builder, loc, ValueRange{theta},
                                          ValueRange{}, ValueRange{q1})
                   .getOperation();

  func::ReturnOp::create(builder, loc);
  return handles;
}

// Build the equivalent of:
//
//   func.func @nested_if(%q0: !quake.ref, %q1: !quake.ref, %flag: i1)
//   attributes
//   {"cudaq-kernel"} {
//     cc.if(%flag) {
//       quake.h %q0 : (!quake.ref) -> ()
//       quake.x %q1 : (!quake.ref) -> ()
//     } else {
//       quake.z %q0 : (!quake.ref) -> ()
//     }
//     return
//   }
static NestedIfHandles buildNestedIf(ModuleOp module) {
  OpBuilder builder(module.getContext());
  Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(module.getBody());

  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto funcTy =
      builder.getFunctionType({refTy, refTy, builder.getI1Type()}, {});

  auto func = func::FuncOp::create(builder, loc, "nested_if", funcTy);
  func->setAttr("cudaq-kernel", builder.getUnitAttr());

  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value q0 = entry->getArgument(0);
  Value q1 = entry->getArgument(1);
  Value flag = entry->getArgument(2);

  NestedIfHandles handles;
  handles.func = func;

  auto ifOp = cudaq::cc::IfOp::create(
      builder, loc, TypeRange{}, flag,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{});
        handles.h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
        handles.x = cudaq::quake::XOp::create(builder, loc, q1).getOperation();
        cudaq::cc::ContinueOp::create(builder, loc);
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{});
        handles.z = cudaq::quake::ZOp::create(builder, loc, q0).getOperation();
        cudaq::cc::ContinueOp::create(builder, loc);
      });

  handles.ifOp = ifOp.getOperation();

  builder.setInsertionPointAfter(ifOp);
  func::ReturnOp::create(builder, loc);
  return handles;
}

// The simple function has one block with three unitary groups:
//   group 0: quake.h, quake.x
//   group 1: quake.z
//   group 2: quake.rx
//
// The measurement and classical constant break unitary runs, so they should
// not map to any group. All three groups should report the same containing
// block, and querying that block should return all three groups.
TEST(UnitaryOpGroupingAnalysisTest, GroupsSimpleFunction) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();

  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  SimpleHandles ops = buildSimple(*module);

  cudaq::quake::detail::UnitaryOpGroupingAnalysis analysis(ops.func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 3u);

  EXPECT_TRUE(analysis.inSameGroup(ops.h, ops.x));
  EXPECT_FALSE(analysis.inSameGroup(ops.x, ops.z));
  EXPECT_FALSE(analysis.inSameGroup(ops.z, ops.rx));

  EXPECT_EQ(analysis.getGroupContainingOp(ops.mz), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(ops.constant), nullptr);

  EXPECT_EQ(groups[0].ops.size(), 2u);
  EXPECT_EQ(groups[0].ops[0], ops.h);
  EXPECT_EQ(groups[0].ops[1], ops.x);

  EXPECT_EQ(groups[1].ops.size(), 1u);
  EXPECT_EQ(groups[1].ops[0], ops.z);

  EXPECT_EQ(groups[2].ops.size(), 1u);
  EXPECT_EQ(groups[2].ops[0], ops.rx);

  EXPECT_EQ(analysis.getGroupsIn(groups[0].block).size(), 3u);
}

// The nested-if function has two unitary groups, each in a different nested
// region block:
//   group 0: quake.h, quake.x in the then block
//   group 1: quake.z in the else block
//
// The parent cc.if operation is not unitary and should not map to a group.
// The then-block operations should be in the same group, but the then and else
// operations should not be in the same group because groups never cross block
// or region boundaries.
TEST(UnitaryOpGroupingAnalysisTest, GroupsNestedIfRegionsSeparately) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();

  OwningOpRef<ModuleOp> module(ModuleOp::create(UnknownLoc::get(&context)));
  NestedIfHandles ops = buildNestedIf(*module);

  cudaq::quake::detail::UnitaryOpGroupingAnalysis analysis(ops.func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);

  EXPECT_EQ(analysis.getGroupContainingOp(ops.ifOp), nullptr);
  EXPECT_TRUE(analysis.inSameGroup(ops.h, ops.x));
  EXPECT_FALSE(analysis.inSameGroup(ops.h, ops.z));

  EXPECT_EQ(groups[0].ops.size(), 2u);
  EXPECT_EQ(groups[0].ops[0], ops.h);
  EXPECT_EQ(groups[0].ops[1], ops.x);

  EXPECT_EQ(groups[1].ops.size(), 1u);
  EXPECT_EQ(groups[1].ops[0], ops.z);

  EXPECT_NE(groups[0].block, groups[1].block);
  EXPECT_EQ(analysis.getGroupsIn(groups[0].block).size(), 1u);
  EXPECT_EQ(analysis.getGroupsIn(groups[1].block).size(), 1u);
}
