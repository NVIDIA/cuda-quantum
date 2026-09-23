/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/NewUnitaryOpGrouping.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

using cudaq::quake::detail::NewUnitaryOpGroupingAnalysis;

static void loadTestDialects(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
}

static func::FuncOp createKernel(ModuleOp module, OpBuilder &builder,
                                 llvm::StringRef name,
                                 ArrayRef<Type> inputTypes = {}) {
  Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(module.getBody());

  auto funcTy = builder.getFunctionType(inputTypes, {});
  auto func = func::FuncOp::create(builder, loc, name, funcTy);
  func->setAttr("cudaq-kernel", builder.getUnitAttr());
  func.addEntryBlock();
  builder.setInsertionPointToStart(&func.front());
  return func;
}

class BuilderNewUnitaryOpGroupingAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    loadTestDialects(context);
    module = OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
  }

  func::FuncOp createKernel(llvm::StringRef name,
                            ArrayRef<Type> inputTypes = {}) {
    OpBuilder builder(&context);
    return ::createKernel(*module, builder, name, inputTypes);
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

// Expected MLIR:
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
//
// Expected analysis:
//   groups.size() == 3
//   group 0: quake.h, quake.x
//   group 1: quake.z
//   group 2: quake.rx
//   inSameGroup(h, x) == true
//   inSameGroup(x, z) == false
//   inSameGroup(z, rx) == false
//   quake.mz and arith.constant do not belong to a group.
//   getGroupsIn(group 0 block).size() == 3
TEST_F(BuilderNewUnitaryOpGroupingAnalysisTest, GroupsSimpleFunction) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("simple", {refTy, refTy, builder.getF64Type()});
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = func.getArgument(0);
  Value q1 = func.getArgument(1);
  Value theta = func.getArgument(2);

  auto *h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q1).getOperation();
  auto measureTy = cudaq::cc::MeasureHandleType::get(&context);
  auto *mz = cudaq::quake::MzOp::create(builder, loc, TypeRange{measureTy},
                                        ValueRange{q0}, StringAttr{})
                 .getOperation();
  auto *z = cudaq::quake::ZOp::create(builder, loc, q0).getOperation();
  auto *constant =
      arith::ConstantIntOp::create(builder, loc, 0, 64).getOperation();
  auto *rx = cudaq::quake::RxOp::create(builder, loc, ValueRange{theta},
                                        ValueRange{}, ValueRange{q1})
                 .getOperation();
  func::ReturnOp::create(builder, loc);

  NewUnitaryOpGroupingAnalysis analysis(func);

  // ASSERT_EQ(groups.size(), 3u);
  // EXPECT_TRUE(analysis.inSameGroup(h, x));
  // EXPECT_FALSE(analysis.inSameGroup(x, z));
  // EXPECT_FALSE(analysis.inSameGroup(z, rx));
  // EXPECT_EQ(analysis.getGroupContainingOp(mz), nullptr);
  // EXPECT_EQ(analysis.getGroupContainingOp(constant), nullptr);
  // expectGroupOps(groups[0], {h, x});
  // expectGroupOps(groups[1], {z});
  // expectGroupOps(groups[2], {rx});
  // EXPECT_EQ(analysis.getGroupsIn(groups[0].block).size(), 3u);
}

// Expected MLIR:
//
//   func.func @nested_if(%q0: !quake.ref, %q1: !quake.ref, %flag: i1)
//   attributes {"cudaq-kernel"} {
//     cc.if(%flag) {
//       quake.h %q0 : (!quake.ref) -> ()
//       quake.x %q1 : (!quake.ref) -> ()
//     } else {
//       quake.z %q0 : (!quake.ref) -> ()
//     }
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h, quake.x in the then block
//   group 1: quake.z in the else block
//   cc.if does not belong to a group.
//   inSameGroup(h, x) == true
//   inSameGroup(h, z) == false
//   group 0 and group 1 have different blocks.
//   getGroupsIn(group 0 block).size() == 1
//   getGroupsIn(group 1 block).size() == 1
TEST_F(BuilderNewUnitaryOpGroupingAnalysisTest,
       GroupsNestedIfRegionsSeparately) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("nested_if", {refTy, refTy, builder.getI1Type()});
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = func.getArgument(0);
  Value q1 = func.getArgument(1);
  Value flag = func.getArgument(2);

  Operation *h = nullptr;
  Operation *x = nullptr;
  Operation *z = nullptr;
  auto ifOp = cudaq::cc::IfOp::create(
      builder, loc, TypeRange{}, flag,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{});
        h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
        x = cudaq::quake::XOp::create(builder, loc, q1).getOperation();
        cudaq::cc::ContinueOp::create(builder, loc);
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{});
        z = cudaq::quake::ZOp::create(builder, loc, q0).getOperation();
        cudaq::cc::ContinueOp::create(builder, loc);
      });
  builder.setInsertionPointAfter(ifOp);
  func::ReturnOp::create(builder, loc);

  NewUnitaryOpGroupingAnalysis analysis(func);
}
