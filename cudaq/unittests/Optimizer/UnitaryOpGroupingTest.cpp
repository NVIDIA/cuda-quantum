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
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <cstddef>
#include <initializer_list>

using namespace mlir;

using cudaq::quake::detail::UnitaryOpGroup;
using cudaq::quake::detail::UnitaryOpGroupingAnalysis;

static void expectOperations(llvm::ArrayRef<Operation *> actual,
                             std::initializer_list<Operation *> expected) {
  ASSERT_EQ(actual.size(), expected.size());

  std::size_t index = 0;
  for (Operation *op : expected)
    EXPECT_EQ(actual[index++], op);
}

static void
expectGroup(const UnitaryOpGroup &group, const Block *expectedBlock,
            std::initializer_list<Operation *> expectedUnitaryOps,
            std::initializer_list<Operation *> expectedDelimiters = {}) {
  EXPECT_EQ(group.block, expectedBlock);
  expectOperations(group.ops, expectedUnitaryOps);
  expectOperations(group.trailingDelimiterOps, expectedDelimiters);
}

static void expectGroupOps(const UnitaryOpGroup &group,
                           std::initializer_list<Operation *> expected) {
  expectOperations(group.ops, expected);
}

static void expectGroupIndex(const UnitaryOpGroupingAnalysis &analysis,
                             Operation *op, std::optional<unsigned> expected) {
  auto actual = analysis.getGroupIndexForOp(op);
  ASSERT_EQ(actual.has_value(), expected.has_value());
  if (expected)
    EXPECT_EQ(*actual, *expected);
}

static Value createNullWire(OpBuilder &builder, Location loc) {
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  return cudaq::quake::NullWireOp::create(builder, loc, wireTy);
}

template <typename GateOp>
static GateOp createWireGate(OpBuilder &builder, Location loc, Value target) {
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  return GateOp::create(builder, loc, TypeRange{wireTy}, /*is_adj=*/false,
                        ValueRange{}, ValueRange{}, ValueRange{target},
                        DenseBoolArrayAttr{});
}

template <typename GateOp>
static GateOp createWireGate(OpBuilder &builder, Location loc,
                             ValueRange controls, ValueRange targets) {
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  SmallVector<Type> resultTypes(controls.size() + targets.size(), wireTy);
  return GateOp::create(builder, loc, resultTypes, /*is_adj=*/false,
                        ValueRange{}, controls, targets, DenseBoolArrayAttr{});
}

template <typename MeasurementOp>
static MeasurementOp createWireMeasurement(OpBuilder &builder, Location loc,
                                           Value target) {
  auto measureTy = cudaq::quake::MeasureType::get(builder.getContext());
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  return MeasurementOp::create(builder, loc, TypeRange{measureTy, wireTy},
                               ValueRange{target}, StringAttr{});
}

template <typename MeasurementOp>
static MeasurementOp createRefMeasurement(OpBuilder &builder, Location loc,
                                          Value target) {
  auto measureTy = cudaq::cc::MeasureHandleType::get(builder.getContext());
  return MeasurementOp::create(builder, loc, TypeRange{measureTy},
                               ValueRange{target}, StringAttr{});
}

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
//   group 0: quake.h, quake.x; trailing delimiter: quake.mz
//   group 1: quake.z
//   group 2: quake.rx
//   inSameGroup(h, x) == true
//   inSameGroup(x, mz) == true
//   inSameGroup(x, z) == false
//   inSameGroup(z, rx) == false
//   arith.constant does not belong to a group.
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
  expectGroup(groups[0], &func.front(), {h, x}, {mz});
  expectGroup(groups[1], &func.front(), {z});
  expectGroup(groups[2], &func.front(), {rx});

  expectGroupIndex(analysis, h, 0u);
  expectGroupIndex(analysis, mz, 0u);
  expectGroupIndex(analysis, z, 1u);
  expectGroupIndex(analysis, rx, 2u);
  expectGroupIndex(analysis, constant, std::nullopt);
  expectGroupIndex(analysis, nullptr, std::nullopt);
  EXPECT_TRUE(analysis.inSameGroup(h, x));
  EXPECT_TRUE(analysis.inSameGroup(x, mz));
  EXPECT_FALSE(analysis.inSameGroup(x, z));
  EXPECT_FALSE(analysis.inSameGroup(z, rx));
  EXPECT_EQ(analysis.getGroupContainingOp(mz), &groups[0]);
  EXPECT_EQ(analysis.getGroupContainingOp(constant), nullptr);
  EXPECT_EQ(analysis.getBlockForGroup(groups[0]), &func.front());

  auto groupsInBlock = analysis.getGroupsIn(&func.front());
  ASSERT_EQ(groupsInBlock.size(), 3u);
  EXPECT_EQ(groupsInBlock[0], &groups[0]);
  EXPECT_EQ(groupsInBlock[1], &groups[1]);
  EXPECT_EQ(groupsInBlock[2], &groups[2]);
  EXPECT_TRUE(analysis.getGroupsIn(nullptr).empty());
}

// Expected MLIR:
//
//   func.func @nested_if(%q0: !quake.ref, %q1: !quake.ref, %flag: i1)
//   attributes {"cudaq-kernel"} {
//     cc.if(%flag) {
//       quake.h %q0 : (!quake.ref) -> ()
//       quake.x %q1 : (!quake.ref) -> ()
//       %m = quake.mz %q0 : (!quake.ref) -> !cc.measure_handle
//     } else {
//       quake.z %q0 : (!quake.ref) -> ()
//       quake.reset %q0 : (!quake.ref) -> ()
//     }
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h, quake.x; trailing delimiter: quake.mz in the then block
//   group 1: quake.z; trailing delimiter: quake.reset in the else block
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
  Operation *mz = nullptr;
  Operation *z = nullptr;
  Operation *reset = nullptr;
  auto ifOp = cudaq::cc::IfOp::create(
      builder, loc, TypeRange{}, flag,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{});
        h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
        x = cudaq::quake::XOp::create(builder, loc, q1).getOperation();
        mz = createRefMeasurement<cudaq::quake::MzOp>(builder, loc, q0)
                 .getOperation();
        cudaq::cc::ContinueOp::create(builder, loc);
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{});
        z = cudaq::quake::ZOp::create(builder, loc, q0).getOperation();
        reset = cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, q0)
                    .getOperation();
        cudaq::cc::ContinueOp::create(builder, loc);
      });
  builder.setInsertionPointAfter(ifOp);
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroup(groups[0], h->getBlock(), {h, x}, {mz});
  expectGroup(groups[1], z->getBlock(), {z}, {reset});
  EXPECT_NE(groups[0].block, groups[1].block);

  EXPECT_TRUE(analysis.inSameGroup(h, x));
  EXPECT_TRUE(analysis.inSameGroup(x, mz));
  EXPECT_TRUE(analysis.inSameGroup(z, reset));
  EXPECT_FALSE(analysis.inSameGroup(h, z));
  EXPECT_EQ(analysis.getGroupContainingOp(ifOp.getOperation()), nullptr);
  EXPECT_TRUE(analysis.getGroupsIn(&func.front()).empty());

  auto thenGroups = analysis.getGroupsIn(h->getBlock());
  ASSERT_EQ(thenGroups.size(), 1u);
  EXPECT_EQ(thenGroups[0], &groups[0]);

  auto elseGroups = analysis.getGroupsIn(z->getBlock());
  ASSERT_EQ(elseGroups.size(), 1u);
  EXPECT_EQ(elseGroups[0], &groups[1]);
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       EmptyAndNonFunctionInputsProduceNoGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto func = createKernel("empty");
  builder.setInsertionPointToEnd(&func.front());
  auto returnOp = func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis functionAnalysis(func);
  EXPECT_TRUE(functionAnalysis.getGroups().empty());
  expectGroupIndex(functionAnalysis, returnOp.getOperation(), std::nullopt);
  EXPECT_EQ(functionAnalysis.getGroupContainingOp(nullptr), nullptr);
  EXPECT_FALSE(functionAnalysis.inSameGroup(nullptr, nullptr));

  UnitaryOpGroupingAnalysis moduleAnalysis(module->getOperation());
  EXPECT_TRUE(moduleAnalysis.getGroups().empty());
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       CoalescesConsecutiveMeasurementAndResetDelimiters) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("consecutive_delimiters", {refTy, refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = func.getArgument(0);
  Value q1 = func.getArgument(1);
  auto *mx =
      createRefMeasurement<cudaq::quake::MxOp>(builder, loc, q0).getOperation();
  auto *reset0 = cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, q0)
                     .getOperation();
  auto *my =
      createRefMeasurement<cudaq::quake::MyOp>(builder, loc, q1).getOperation();
  auto *h = cudaq::quake::HOp::create(builder, loc, q0).getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q1).getOperation();
  auto *mz =
      createRefMeasurement<cudaq::quake::MzOp>(builder, loc, q0).getOperation();
  auto *reset1 = cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, q1)
                     .getOperation();
  auto *z = cudaq::quake::ZOp::create(builder, loc, q0).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 3u);
  expectGroup(groups[0], &func.front(), {}, {mx, reset0, my});
  expectGroup(groups[1], &func.front(), {h, x}, {mz, reset1});
  expectGroup(groups[2], &func.front(), {z});
  EXPECT_TRUE(analysis.inSameGroup(mx, my));
  EXPECT_FALSE(analysis.inSameGroup(my, h));
  EXPECT_TRUE(analysis.inSameGroup(h, reset1));
  EXPECT_FALSE(analysis.inSameGroup(reset1, z));
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       ConsecutiveHardBoundariesDoNotCreateEmptyGroups) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("consecutive_hard_boundaries", {refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q = func.getArgument(0);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *constant0 =
      arith::ConstantIntOp::create(builder, loc, 0, 64).getOperation();
  auto *constant1 =
      arith::ConstantIntOp::create(builder, loc, 1, 64).getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroup(groups[0], &func.front(), {h});
  expectGroup(groups[1], &func.front(), {x});
  EXPECT_EQ(analysis.getGroupContainingOp(constant0), nullptr);
  EXPECT_EQ(analysis.getGroupContainingOp(constant1), nullptr);
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       UnknownWireIdentityFallsBackToTextualOrder) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  auto func = createKernel("unknown_wire_identity", {wireTy, wireTy});
  builder.setInsertionPointToEnd(&func.front());

  auto mz = createWireMeasurement<cudaq::quake::MzOp>(builder, loc,
                                                      func.getArgument(0));
  auto x = createWireGate<cudaq::quake::XOp>(builder, loc, func.getArgument(1));
  auto *mzOp = mz.getOperation();
  auto *xOp = x.getOperation();
  cudaq::quake::SinkOp::create(builder, loc, mz.getWires().front());
  cudaq::quake::SinkOp::create(builder, loc, x.getResult(0));
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroup(groups[0], &func.front(), {}, {mzOp});
  expectGroup(groups[1], &func.front(), {xOp});
  EXPECT_FALSE(analysis.inSameGroup(mzOp, xOp));
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       WireModePrioritizesUnitariesAndUsesOriginalOrderForTies) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  auto func = createKernel("wire_ready_queues");
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = createNullWire(builder, loc);
  Value q1 = createNullWire(builder, loc);
  Value q2 = createNullWire(builder, loc);
  Value q3 = createNullWire(builder, loc);
  auto mz = createWireMeasurement<cudaq::quake::MzOp>(builder, loc, q0);
  auto reset =
      cudaq::quake::ResetOp::create(builder, loc, TypeRange{wireTy}, q1);
  auto z = createWireGate<cudaq::quake::ZOp>(builder, loc, q2);
  auto h = createWireGate<cudaq::quake::HOp>(builder, loc, q3);
  auto *mzOp = mz.getOperation();
  auto *resetOp = reset.getOperation();
  auto *zOp = z.getOperation();
  auto *hOp = h.getOperation();
  cudaq::quake::SinkOp::create(builder, loc, mz.getWires().front());
  cudaq::quake::SinkOp::create(builder, loc, reset.getResult(0));
  cudaq::quake::SinkOp::create(builder, loc, z.getResult(0));
  cudaq::quake::SinkOp::create(builder, loc, h.getResult(0));
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 1u);
  expectGroup(groups[0], &func.front(), {zOp, hOp}, {mzOp, resetOp});
  expectGroupIndex(analysis, zOp, 0u);
  expectGroupIndex(analysis, resetOp, 0u);
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       UnitaryWaitsForMeasurementPredecessor) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto func = createKernel("measurement_predecessor");
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = createNullWire(builder, loc);
  Value q1 = createNullWire(builder, loc);
  auto mz = createWireMeasurement<cudaq::quake::MzOp>(builder, loc, q0);
  auto x =
      createWireGate<cudaq::quake::XOp>(builder, loc, mz.getWires().front());
  auto z = createWireGate<cudaq::quake::ZOp>(builder, loc, q1);
  auto *mzOp = mz.getOperation();
  auto *xOp = x.getOperation();
  auto *zOp = z.getOperation();
  cudaq::quake::SinkOp::create(builder, loc, x.getResult(0));
  cudaq::quake::SinkOp::create(builder, loc, z.getResult(0));
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroup(groups[0], &func.front(), {zOp}, {mzOp});
  expectGroup(groups[1], &func.front(), {xOp});
  EXPECT_FALSE(analysis.inSameGroup(mzOp, xOp));
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       JoinWaitsForAllMeasurementPredecessors) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto func = createKernel("multiple_predecessors");
  builder.setInsertionPointToEnd(&func.front());

  Value q0 = createNullWire(builder, loc);
  Value q1 = createNullWire(builder, loc);
  Value q2 = createNullWire(builder, loc);
  auto mz0 = createWireMeasurement<cudaq::quake::MzOp>(builder, loc, q0);
  auto mz1 = createWireMeasurement<cudaq::quake::MzOp>(builder, loc, q1);
  auto cx = createWireGate<cudaq::quake::XOp>(
      builder, loc, ValueRange{mz0.getWires().front()},
      ValueRange{mz1.getWires().front()});
  auto z = createWireGate<cudaq::quake::ZOp>(builder, loc, q2);
  auto *mz0Op = mz0.getOperation();
  auto *mz1Op = mz1.getOperation();
  auto *cxOp = cx.getOperation();
  auto *zOp = z.getOperation();
  for (Value wire : cx.getWires())
    cudaq::quake::SinkOp::create(builder, loc, wire);
  cudaq::quake::SinkOp::create(builder, loc, z.getResult(0));
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroup(groups[0], &func.front(), {zOp}, {mz0Op, mz1Op});
  expectGroup(groups[1], &func.front(), {cxOp});
  EXPECT_TRUE(analysis.inSameGroup(mz0Op, mz1Op));
  EXPECT_FALSE(analysis.inSameGroup(mz1Op, cxOp));
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       QubitIdentityOrdersDistinctSsaRoots) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  auto func = createKernel("qubit_identity_edges");
  builder.setInsertionPointToEnd(&func.front());

  auto allocation = cudaq::quake::AllocaOp::create(builder, loc, refTy);
  Value wireA = cudaq::quake::UnwrapOp::create(builder, loc, wireTy,
                                               allocation.getRefOrVec());
  Value wireB = cudaq::quake::UnwrapOp::create(builder, loc, wireTy,
                                               allocation.getRefOrVec());
  Value independentWire = createNullWire(builder, loc);
  auto h = createWireGate<cudaq::quake::HOp>(builder, loc, wireA);
  auto mz =
      createWireMeasurement<cudaq::quake::MzOp>(builder, loc, h.getResult(0));
  auto z = createWireGate<cudaq::quake::ZOp>(builder, loc, independentWire);
  auto x = createWireGate<cudaq::quake::XOp>(builder, loc, wireB);
  auto *hOp = h.getOperation();
  auto *mzOp = mz.getOperation();
  auto *zOp = z.getOperation();
  auto *xOp = x.getOperation();
  cudaq::quake::SinkOp::create(builder, loc, mz.getWires().front());
  cudaq::quake::SinkOp::create(builder, loc, z.getResult(0));
  cudaq::quake::SinkOp::create(builder, loc, x.getResult(0));
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroup(groups[0], &func.front(), {hOp, zOp}, {mzOp});
  expectGroup(groups[1], &func.front(), {xOp});
  EXPECT_TRUE(analysis.inSameGroup(hOp, zOp));
  EXPECT_FALSE(analysis.inSameGroup(mzOp, xOp));
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       RepeatedQubitIdentityWithinOneOpDoesNotCreateSelfEdge) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  auto func = createKernel("repeated_qubit_identity");
  builder.setInsertionPointToEnd(&func.front());

  auto allocation = cudaq::quake::AllocaOp::create(builder, loc, refTy);
  Value wireA = cudaq::quake::UnwrapOp::create(builder, loc, wireTy,
                                               allocation.getRefOrVec());
  Value wireB = cudaq::quake::UnwrapOp::create(builder, loc, wireTy,
                                               allocation.getRefOrVec());
  auto controlledX = createWireGate<cudaq::quake::XOp>(
      builder, loc, ValueRange{wireA}, ValueRange{wireB});
  auto *controlledXOp = controlledX.getOperation();
  for (Value wire : controlledX.getWires())
    cudaq::quake::SinkOp::create(builder, loc, wire);
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 1u);
  expectGroup(groups[0], &func.front(), {controlledXOp});
}

TEST_F(BuilderUnitaryOpGroupingAnalysisTest,
       FlushesUnterminatedNestedBlockAtEnd) {
  OpBuilder builder(&context);
  Location loc = builder.getUnknownLoc();
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("unterminated_nested_block", {refTy});
  builder.setInsertionPointToEnd(&func.front());

  Value q = func.getArgument(0);
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  Operation *x = nullptr;
  Operation *mz = nullptr;
  auto scope = cudaq::cc::ScopeOp::create(
      builder, loc, [&](OpBuilder &builder, Location loc) {
        x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
        mz = createRefMeasurement<cudaq::quake::MzOp>(builder, loc, q)
                 .getOperation();
        // Intentionally omit cc.continue to exercise the end-of-block flush.
      });
  builder.setInsertionPointAfter(scope);
  auto *z = cudaq::quake::ZOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 3u);
  expectGroup(groups[0], &func.front(), {h});
  expectGroup(groups[1], x->getBlock(), {x}, {mz});
  expectGroup(groups[2], &func.front(), {z});
  EXPECT_EQ(analysis.getGroupContainingOp(scope.getOperation()), nullptr);

  auto parentGroups = analysis.getGroupsIn(&func.front());
  ASSERT_EQ(parentGroups.size(), 2u);
  EXPECT_EQ(parentGroups[0], &groups[0]);
  EXPECT_EQ(parentGroups[1], &groups[2]);
  auto nestedGroups = analysis.getGroupsIn(x->getBlock());
  ASSERT_EQ(nestedGroups.size(), 1u);
  EXPECT_EQ(nestedGroups[0], &groups[1]);
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
//     %m = quake.mz %vec : (!quake.veq<3>) -> !cc.sequence<!cc.measure_handle>
//     quake.x %q : (!quake.ref) -> ()
//     return
//   }
//
// Expected analysis:
//   groups.size() == 2
//   group 0: quake.h; trailing delimiter: quake.mz
//   group 1: quake.x
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
      cudaq::cc::SequenceType::get(cudaq::cc::MeasureHandleType::get(&context));
  auto *h = cudaq::quake::HOp::create(builder, loc, q).getOperation();
  auto *mz = cudaq::quake::MzOp::create(builder, loc, TypeRange{measureVecTy},
                                        ValueRange{vec}, StringAttr{})
                 .getOperation();
  auto *x = cudaq::quake::XOp::create(builder, loc, q).getOperation();
  func::ReturnOp::create(builder, loc);

  UnitaryOpGroupingAnalysis analysis(func);
  const auto &groups = analysis.getGroups();

  ASSERT_EQ(groups.size(), 2u);
  expectGroup(groups[0], &func.front(), {h}, {mz});
  expectGroup(groups[1], &func.front(), {x});
  EXPECT_EQ(analysis.getGroupContainingOp(mz), &groups[0]);
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
//   group 0: quake.h; trailing delimiter: quake.mx
//   group 1: quake.x; trailing delimiter: quake.my
//   group 2: quake.z
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
  expectGroup(groups[0], &func.front(), {h}, {mx});
  expectGroup(groups[1], &func.front(), {x}, {my});
  expectGroup(groups[2], &func.front(), {z});
  EXPECT_EQ(analysis.getGroupContainingOp(mx), &groups[0]);
  EXPECT_EQ(analysis.getGroupContainingOp(my), &groups[1]);
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
//   group 0: quake.h; trailing delimiter: quake.reset
//   group 1: quake.x
TEST_F(BuilderUnitaryOpGroupingAnalysisTest, ResetRefIsATrailingDelimiter) {
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
  expectGroup(groups[0], &func.front(), {h}, {reset});
  expectGroup(groups[1], &func.front(), {x});
  EXPECT_EQ(analysis.getGroupContainingOp(reset), &groups[0]);
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
