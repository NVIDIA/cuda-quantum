/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstddef>
#include <initializer_list>

using namespace mlir;

using cudaq::quake::detail::UnitaryOpGroup;
using cudaq::quake::detail::UnitaryOpGroupingAnalysis;

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

static void expectGroupOps(const UnitaryOpGroup &group,
                           std::initializer_list<Operation *> expected) {
  ASSERT_EQ(group.ops.size(), expected.size());
  std::size_t index = 0;
  for (Operation *op : expected)
    EXPECT_EQ(group.ops[index++], op);
}

class BuilderUnitaryOpGroupingAnalysisTest : public ::testing::Test {
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
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, GroupsSimpleFunction) {
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

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 3u);
  EXPECT_TRUE(analysis.inSameGroup(h, x));
  EXPECT_FALSE(analysis.inSameGroup(x, z));
  EXPECT_FALSE(analysis.inSameGroup(z, rx));
  EXPECT_EQ(analysis.getGroupContainingOp(mz), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(constant), nullptr);
  expectGroupOps(groups[0], {h, x});
  expectGroupOps(groups[1], {z});
  expectGroupOps(groups[2], {rx});
  EXPECT_EQ(analysis.getGroupsIn(groups[0].block).size(), 3u);
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
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, GroupsNestedIfRegionsSeparately) {
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

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  EXPECT_EQ(analysis.getGroupContainingOp(ifOp.getOperation()), nullptr);
  EXPECT_TRUE(analysis.inSameGroup(h, x));
  EXPECT_FALSE(analysis.inSameGroup(h, z));
  expectGroupOps(groups[0], {h, x});
  expectGroupOps(groups[1], {z});
  EXPECT_NE(groups[0].block, groups[1].block);
  EXPECT_EQ(analysis.getGroupsIn(groups[0].block).size(), 1u);
  EXPECT_EQ(analysis.getGroupsIn(groups[1].block).size(), 1u);
}

// Expected MLIR:
//
//   func.func @empty() attributes {"cudaq-kernel"} {
//     return
//   }
//
// Expected analysis:
//   groups.empty() == true
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, EmptyFunctionHasNoGroups) {
  auto func = createKernel("empty");
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  builder.setInsertionPointToEnd(&func.front());
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  EXPECT_TRUE(analysis.getGroups().empty());
}

// Expected MLIR:
//
//   func.func @alloca_veq_break(%q: !quake.ref) attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %v = quake.alloca !quake.veq<2>
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.alloca does not belong to a group.
//   inSameGroup(h, x) == false
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, AllocaVeqBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veqTy = cudaq::quake::VeqType::get(&context, 2);
  auto func = createKernel("alloca_veq_break", {refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q = func.getArgument(0);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *alloca =
      cudaq::quake::AllocaOp::create(builder, loc, veqTy).getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(alloca), nullptr);
  EXPECT_FALSE(analysis.inSameGroup(h, x));
}

// Expected MLIR:
//
//   func.func @extract_ref_break(%vec: !quake.veq<2>, %q: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %r = quake.extract_ref %vec[0] : (!quake.veq<2>) -> !quake.ref
//     quake.x %r : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.extract_ref does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, ExtractRefBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veqTy = cudaq::quake::VeqType::get(&context, 2);
  auto func = createKernel("extract_ref_break", {veqTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value vec = func.getArgument(0);
  Value q = func.getArgument(1);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto extract = cudaq::quake::ExtractRefOp::create(builder, loc, vec, 0u);
  auto *x = cudaq::quake::XOp::create(builder, loc, extract.getResult())
                .getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(extract.getOperation()), nullptr);
}

// Expected MLIR:
//
//   func.func @dynamic_extract_ref_break(%vec: !quake.veq<?>, %i: i64,
//                                        %q: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %r = quake.extract_ref %vec[%i] : (!quake.veq<?>, i64) -> !quake.ref
//     quake.y %r : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.y
//   quake.extract_ref does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       DynamicExtractRefBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veqTy = cudaq::quake::VeqType::getUnsized(&context);
  auto i64Ty = builder.getI64Type();
  auto func = createKernel("dynamic_extract_ref_break", {veqTy, i64Ty, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value vec = func.getArgument(0);
  Value index = func.getArgument(1);
  Value q = func.getArgument(2);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto extract = cudaq::quake::ExtractRefOp::create(builder, loc, vec, index);
  auto *y = cudaq::quake::YOp::create(builder, loc, extract.getResult())
                .getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {y});
  EXPECT_EQ(analysis.getGroupContainingOp(extract.getOperation()), nullptr);
}

// Expected MLIR:
//
//   func.func @subveq_break(%vec: !quake.veq<4>, %q: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %sub = quake.subveq %vec, 1, 2 : (!quake.veq<4>) -> !quake.veq<2>
//     %r = quake.extract_ref %sub[0] : (!quake.veq<2>) -> !quake.ref
//     quake.x %r : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.subveq and quake.extract_ref do not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, SubVeqBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veq4Ty = cudaq::quake::VeqType::get(&context, 4);
  auto veq2Ty = cudaq::quake::VeqType::get(&context, 2);
  auto func = createKernel("subveq_break", {veq4Ty, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value vec = func.getArgument(0);
  Value q = func.getArgument(1);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto subveq = cudaq::quake::SubVeqOp::create(builder, loc, veq2Ty, vec, 1, 2);
  auto extract =
      cudaq::quake::ExtractRefOp::create(builder, loc, subveq.getResult(), 0u);
  auto *x = cudaq::quake::XOp::create(builder, loc, extract.getResult())
                .getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(subveq.getOperation()), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(extract.getOperation()), nullptr);
}

// Expected MLIR:
//
//   func.func @relax_size_break(%vec: !quake.veq<3>, %q: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %relaxed = quake.relax_size %vec : (!quake.veq<3>) -> !quake.veq<?>
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.relax_size does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, RelaxSizeBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veq3Ty = cudaq::quake::VeqType::get(&context, 3);
  auto unsizedVeqTy = cudaq::quake::VeqType::getUnsized(&context);
  auto func = createKernel("relax_size_break", {veq3Ty, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value vec = func.getArgument(0);
  Value q = func.getArgument(1);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto relax =
      cudaq::quake::RelaxSizeOp::create(builder, loc, unsizedVeqTy, vec);
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(relax.getOperation()), nullptr);
}

// Expected MLIR:
//
//   func.func @concat_break(%q0: !quake.ref, %q1: !quake.ref,
//                           %vec: !quake.veq<2>) attributes {"cudaq-kernel"} {
//     quake.h %q0 : (!quake.ref) -> ()
//     %merged = quake.concat %q1, %vec : (!quake.ref, !quake.veq<2>) ->
//     !quake.veq<3> quake.x %q0 : (!quake.ref) -> () return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.concat does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, ConcatBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veq2Ty = cudaq::quake::VeqType::get(&context, 2);
  auto veq3Ty = cudaq::quake::VeqType::get(&context, 3);
  auto func = createKernel("concat_break", {refTy, refTy, veq2Ty});
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = func.getArgument(0);
  Value q1 = func.getArgument(1);
  Value vec = func.getArgument(2);
  auto *h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
  auto concat =
      cudaq::quake::ConcatOp::create(builder, loc, veq3Ty, ValueRange{q1, vec});
  auto *x = cudaq::quake::XOp::create(builder, loc, q0).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(concat.getOperation()), nullptr);
}

// Expected MLIR:
//
//   func.func @veq_size_break(%vec: !quake.veq<?>, %q: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %size = quake.veq_size %vec : (!quake.veq<?>) -> i64
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.veq_size does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, VeqSizeBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veqTy = cudaq::quake::VeqType::getUnsized(&context);
  auto func = createKernel("veq_size_break", {veqTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value vec = func.getArgument(0);
  Value q = func.getArgument(1);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto veqSize =
      cudaq::quake::VeqSizeOp::create(builder, loc, builder.getI64Type(), vec);
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(veqSize.getOperation()), nullptr);
}

// Expected MLIR:
//
//   func.func @veq_measurement_break(%vec: !quake.veq<3>, %q: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %m = quake.mz %vec : (!quake.veq<3>) -> !cc.stdvec<!cc.measure_handle>
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.mz does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       VectorMeasurementBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veqTy = cudaq::quake::VeqType::get(&context, 3);
  auto func = createKernel("veq_measurement_break", {veqTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value vec = func.getArgument(0);
  Value q = func.getArgument(1);
  Type measureVecTy =
      cudaq::cc::StdvecType::get(cudaq::cc::MeasureHandleType::get(&context));
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *mz = cudaq::quake::MzOp::create(builder, loc, TypeRange{measureVecTy},
                                        ValueRange{vec}, StringAttr{})
                 .getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(mz), nullptr);
}

// Expected MLIR:
//
//   func.func @mx_my_measurement_break(%q: !quake.ref) attributes
//   {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     %mx = quake.mx %q : (!quake.ref) -> !cc.measure_handle
//     quake.x %q : (!quake.ref) -> ()
//     %my = quake.my %q : (!quake.ref) -> !cc.measure_handle
//     quake.z %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 3
//   group 0: quake.h
//   group 1: quake.x
//   group 2: quake.z
//   quake.mx and quake.my do not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       MxAndMyMeasurementsBreakBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("mx_my_measurement_break", {refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q = func.getArgument(0);
  auto measureTy = cudaq::cc::MeasureHandleType::get(&context);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *mx = cudaq::quake::MxOp::create(builder, loc, TypeRange{measureTy},
                                        ValueRange{q}, StringAttr{})
                 .getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  auto *my = cudaq::quake::MyOp::create(builder, loc, TypeRange{measureTy},
                                        ValueRange{q}, StringAttr{})
                 .getOperation();
  auto *z = cudaq::quake::ZOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 3u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  expectGroupOps(groups[2], {z});
  EXPECT_EQ(analysis.getGroupContainingOp(mx), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(my), nullptr);
}

// Expected MLIR:
//
//   func.func @reset_ref_break(%q: !quake.ref) attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     quake.reset %q : (!quake.ref) -> ()
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.reset does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       ResetRefIsExcludedFromUnitaryGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("reset_ref_break", {refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q = func.getArgument(0);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *reset = cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, q)
                    .getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(reset), nullptr);
}

// Expected MLIR:
//
//   func.func @controlled_veq_group(%ctrl: !quake.veq<2>, %target: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.x [%ctrl] %target : (!quake.veq<2>, !quake.ref) -> ()
//     quake.y [%ctrl] %target : (!quake.veq<2>, !quake.ref) -> ()
//     quake.z [%ctrl] %target : (!quake.veq<2>, !quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 1
//   group 0: quake.x, quake.y, quake.z
//   inSameGroup(x, z) == true
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       ControlledGatesWithVeqControlGroupTogether) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veqTy = cudaq::quake::VeqType::get(&context, 2);
  auto func = createKernel("controlled_veq_group", {veqTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value ctrl = func.getArgument(0);
  Value target = func.getArgument(1);
  auto *x = cudaq::quake::XOp::create(builder, loc, ValueRange{ctrl},
                                      ValueRange{target})
                .getOperation();
  auto *y = cudaq::quake::YOp::create(builder, loc, ValueRange{ctrl},
                                      ValueRange{target})
                .getOperation();
  auto *z = cudaq::quake::ZOp::create(builder, loc, ValueRange{ctrl},
                                      ValueRange{target})
                .getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 1u);
  expectGroupOps(groups[0], {x, y, z});
  EXPECT_TRUE(analysis.inSameGroup(x, z));
}

// Expected MLIR:
//
//   func.func @parameterized_gate_group(%theta: f64, %phi: f64, %lambda: f64,
//                                       %q0: !quake.ref, %q1: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.r1 (%theta) %q0 : (f64, !quake.ref) -> ()
//     quake.rx (%theta) %q0 : (f64, !quake.ref) -> ()
//     quake.phased_rx (%theta, %phi) %q0 : (f64, f64, !quake.ref) -> ()
//     quake.ry (%phi) %q0 : (f64, !quake.ref) -> ()
//     quake.rz (%lambda) %q0 : (f64, !quake.ref) -> ()
//     quake.u2 (%theta, %phi) %q0 : (f64, f64, !quake.ref) -> ()
//     quake.u3 (%theta, %phi, %lambda) %q0 : (f64, f64, f64, !quake.ref) -> ()
//     quake.swap %q0, %q1 : (!quake.ref, !quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 1
//   group 0: quake.r1, quake.rx, quake.phased_rx, quake.ry, quake.rz,
//            quake.u2, quake.u3, quake.swap
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       ParameterizedAndMultiTargetGatesGroupTogether) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto f64Ty = builder.getF64Type();
  auto func = createKernel("parameterized_gate_group",
                           {f64Ty, f64Ty, f64Ty, refTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value theta = func.getArgument(0);
  Value phi = func.getArgument(1);
  Value lambda = func.getArgument(2);
  Value q0 = func.getArgument(3);
  Value q1 = func.getArgument(4);
  auto *r1 = cudaq::quake::R1Op::create(builder, loc, ValueRange{theta},
                                        ValueRange{}, ValueRange{q0})
                 .getOperation();
  auto *rx = cudaq::quake::RxOp::create(builder, loc, ValueRange{theta},
                                        ValueRange{}, ValueRange{q0})
                 .getOperation();
  auto *phasedRx =
      cudaq::quake::PhasedRxOp::create(builder, loc, ValueRange{theta, phi},
                                       ValueRange{}, ValueRange{q0})
          .getOperation();
  auto *ry = cudaq::quake::RyOp::create(builder, loc, ValueRange{phi},
                                        ValueRange{}, ValueRange{q0})
                 .getOperation();
  auto *rz = cudaq::quake::RzOp::create(builder, loc, ValueRange{lambda},
                                        ValueRange{}, ValueRange{q0})
                 .getOperation();
  auto *u2 = cudaq::quake::U2Op::create(builder, loc, ValueRange{theta, phi},
                                        ValueRange{}, ValueRange{q0})
                 .getOperation();
  auto *u3 =
      cudaq::quake::U3Op::create(builder, loc, ValueRange{theta, phi, lambda},
                                 ValueRange{}, ValueRange{q0})
          .getOperation();
  auto *swap = cudaq::quake::SwapOp::create(builder, loc, ValueRange{},
                                            ValueRange{q0, q1})
                   .getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 1u);
  expectGroupOps(groups[0], {r1, rx, phasedRx, ry, rz, u2, u3, swap});
}

// Expected MLIR:
//
//   func.func @exp_pauli_group(%theta: f64, %vec: !quake.veq<3>, %q:
//   !quake.ref) attributes {"cudaq-kernel"} {
//     quake.h %q : (!quake.ref) -> ()
//     quake.exp_pauli (%theta) %vec to "XYZ" : (f64, !quake.veq<3>) -> ()
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 1
//   group 0: quake.h, quake.exp_pauli, quake.x
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, ExpPauliWithVeqTargetIsUnitary) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto veqTy = cudaq::quake::VeqType::get(&context, 3);
  auto f64Ty = builder.getF64Type();
  auto func = createKernel("exp_pauli_group", {f64Ty, veqTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value theta = func.getArgument(0);
  Value vec = func.getArgument(1);
  Value q = func.getArgument(2);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *expPauli = cudaq::quake::ExpPauliOp::create(
                       builder, loc, ValueRange{theta}, ValueRange{},
                       ValueRange{vec}, llvm::StringRef("XYZ"))
                       .getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 1u);
  expectGroupOps(groups[0], {h, expPauli, x});
}

// Expected MLIR:
//
//   func.func @compute_action_break(%q: !quake.ref) attributes {"cudaq-kernel"}
//   {
//     %compute = cc.undef !cc.callable<() -> ()>
//     %action = cc.undef !cc.callable<() -> ()>
//     quake.h %q : (!quake.ref) -> ()
//     quake.compute_action %compute, %action : !cc.callable<() -> ()>,
//                                             !cc.callable<() -> ()>
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   quake.compute_action does not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, ComputeActionBreaksBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("compute_action_break", {refTy});
  builder.setInsertionPointToEnd(&func.front());

  auto callableTy = cudaq::cc::CallableType::get(
      &context, builder.getFunctionType(TypeRange{}, TypeRange{}));
  Value q = func.getArgument(0);
  Value compute = cudaq::cc::UndefOp::create(builder, loc, callableTy);
  Value action = cudaq::cc::UndefOp::create(builder, loc, callableTy);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *computeAction = cudaq::quake::ComputeActionOp::create(
                            builder, loc, /*is_dagger=*/false, compute, action)
                            .getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(computeAction), nullptr);
}

// Expected MLIR:
//
//   func.func @cc_scope_boundary(%q0: !quake.ref, %q1: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     quake.h %q0 : (!quake.ref) -> ()
//     cc.scope {
//       quake.x %q0 : (!quake.ref) -> ()
//       quake.y %q1 : (!quake.ref) -> ()
//       cc.continue
//     }
//     quake.z %q0 : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 3
//   group 0: quake.h in the parent block
//   group 1: quake.x, quake.y in the cc.scope block
//   group 2: quake.z in the parent block
//   cc.scope does not belong to a group.
//   group 0 and group 1 have different blocks.
//   group 1 and group 2 have different blocks.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       CCScopeRegionDoesNotMergeWithParentBlock) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("cc_scope_boundary", {refTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = func.getArgument(0);
  Value q1 = func.getArgument(1);
  auto *h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
  Operation *x = nullptr;
  Operation *y = nullptr;
  auto scope = cudaq::cc::ScopeOp::create(
      builder, loc, [&](OpBuilder &builder, Location loc) {
        x = cudaq::quake::XOp::create(builder, loc, q0).getOperation();
        y = cudaq::quake::YOp::create(builder, loc, q1).getOperation();
        cudaq::cc::ContinueOp::create(builder, loc);
      });
  auto *z = cudaq::quake::ZOp::create(builder, loc, q0).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 3u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x, y});
  expectGroupOps(groups[2], {z});
  EXPECT_EQ(analysis.getGroupContainingOp(scope.getOperation()), nullptr);
  EXPECT_NE(groups[0].block, groups[1].block);
  EXPECT_NE(groups[1].block, groups[2].block);
}

// Expected MLIR:
//
//   func.func @cc_memory_break(%value: i32, %q: !quake.ref)
//   attributes {"cudaq-kernel"} {
//     %ptr = cc.alloca i32
//     quake.h %q : (!quake.ref) -> ()
//     cc.store %value, %ptr : !cc.ptr<i32>
//     %loaded = cc.load %ptr : !cc.ptr<i32>
//     %wide = cc.cast signed %loaded : (i32) -> i64
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   cc.store, cc.load, and cc.cast do not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, CCMemoryOpsBreakBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto i32Ty = builder.getI32Type();
  auto i64Ty = builder.getI64Type();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("cc_memory_break", {i32Ty, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value value = func.getArgument(0);
  Value q = func.getArgument(1);
  auto alloca = cudaq::cc::AllocaOp::create(builder, loc, i32Ty);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *store =
      cudaq::cc::StoreOp::create(builder, loc, value, alloca).getOperation();
  auto load = cudaq::cc::LoadOp::create(builder, loc, alloca);
  auto cast = cudaq::cc::CastOp::create(builder, loc, i64Ty, load,
                                        cudaq::cc::CastOpMode::Signed);
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(store), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(load.getOperation()), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(cast.getOperation()), nullptr);
}

// Expected MLIR:
//
//   func.func private @callee(i64)
//   func.func @arith_and_call_break(%q: !quake.ref) attributes {"cudaq-kernel"}
//   {
//     %c0 = arith.constant 0 : i64
//     quake.h %q : (!quake.ref) -> ()
//     %c1 = arith.constant 1 : i64
//     %sum = arith.addi %c0, %c1 : i64
//     call @callee(%sum) : (i64) -> ()
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h
//   group 1: quake.x
//   arith.constant, arith.addi, and func.call do not belong to a group.
TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       ArithAndFuncCallOpsBreakBetweenGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto i64Ty = builder.getI64Type();
  auto refTy = builder.getType<cudaq::quake::RefType>();

  builder.setInsertionPointToEnd(module->getBody());
  func::FuncOp::create(builder, loc, "callee",
                       builder.getFunctionType({i64Ty}, {}));

  auto func = createKernel("arith_and_call_break", {refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q = func.getArgument(0);
  auto *c0 = arith::ConstantIntOp::create(builder, loc, 0, 64).getOperation();
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto c1 = arith::ConstantIntOp::create(builder, loc, 1, 64);
  auto add = arith::AddIOp::create(builder, loc, c0->getResult(0), c1);
  auto call = func::CallOp::create(builder, loc, "callee", TypeRange{},
                                   ValueRange{add.getResult()});
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroupOps(groups[0], {h});
  expectGroupOps(groups[1], {x});
  EXPECT_EQ(analysis.getGroupContainingOp(c0), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(c1.getOperation()), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(add.getOperation()), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(call.getOperation()), nullptr);
}
