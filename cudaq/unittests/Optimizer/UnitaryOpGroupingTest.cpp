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

/// Assert that \p actual contains the expected operation pointers in the same
/// order.
static void expectOperations(llvm::ArrayRef<Operation *> actual,
                             std::initializer_list<Operation *> expected) {
  ASSERT_EQ(actual.size(), expected.size());

  std::size_t index = 0;
  for (Operation *op : expected)
    EXPECT_EQ(actual[index++], op);
}

/// Assert a group's containing block, unitary run, and trailing delimiter run.
static void
expectGroup(const UnitaryOpGroup &group, const Block *expectedBlock,
            std::initializer_list<Operation *> expectedUnitaryOps,
            std::initializer_list<Operation *> expectedDelimiters = {}) {
  EXPECT_EQ(group.block, expectedBlock);
  expectOperations(group.ops, expectedUnitaryOps);
  expectOperations(group.trailingDelimiterOps, expectedDelimiters);
}

/// Assert only the ordered unitary run of a group.
static void expectGroupOps(const UnitaryOpGroup &group,
                           std::initializer_list<Operation *> expected) {
  expectOperations(group.ops, expected);
}

/// Assert both the presence and value of an operation-to-group lookup.
static void expectGroupIndex(const UnitaryOpGroupingAnalysis &analysis,
                             Operation *op, std::optional<unsigned> expected) {
  auto actual = analysis.getGroupIndexForOp(op);
  ASSERT_EQ(actual.has_value(), expected.has_value());
  if (expected)
    EXPECT_EQ(*actual, *expected);
}

/// Create a scalar wire root with a fresh, known logical-qubit identity.
static Value createNullWire(OpBuilder &builder, Location loc) {
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  return cudaq::quake::NullWireOp::create(builder, loc, wireTy);
}

/// Create a non-adjoint scalar-wire gate that threads one target through its
/// result.
template <typename GateOp>
static GateOp createWireGate(OpBuilder &builder, Location loc, Value target) {
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  return GateOp::create(builder, loc, TypeRange{wireTy}, /*is_adj=*/false,
                        ValueRange{}, ValueRange{}, ValueRange{target},
                        DenseBoolArrayAttr{});
}

/// Create a non-adjoint scalar-wire gate with one result for every control and
/// target.
template <typename GateOp>
static GateOp createWireGate(OpBuilder &builder, Location loc,
                             ValueRange controls, ValueRange targets) {
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  SmallVector<Type> resultTypes(controls.size() + targets.size(), wireTy);
  return GateOp::create(builder, loc, resultTypes, /*is_adj=*/false,
                        ValueRange{}, controls, targets, DenseBoolArrayAttr{});
}

/// Create a wire-semantics measurement with classical and threaded-wire
/// results.
template <typename MeasurementOp>
static MeasurementOp createWireMeasurement(OpBuilder &builder, Location loc,
                                           Value target) {
  auto measureTy = cudaq::quake::MeasureType::get(builder.getContext());
  auto wireTy = builder.getType<cudaq::quake::WireType>();
  return MeasurementOp::create(builder, loc, TypeRange{measureTy, wireTy},
                               ValueRange{target}, StringAttr{});
}

/// Create a reference-semantics measurement with a classical-handle result.
template <typename MeasurementOp>
static MeasurementOp createRefMeasurement(OpBuilder &builder, Location loc,
                                          Value target) {
  auto measureTy = cudaq::cc::MeasureHandleType::get(builder.getContext());
  return MeasurementOp::create(builder, loc, TypeRange{measureTy},
                               ValueRange{target}, StringAttr{});
}

/// Load every dialect used to construct the test IR.
static void loadTestDialects(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
}

/// Create a CUDA-Q kernel with one entry block and leave \p builder at the
/// start of that block.
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

/// Fixture providing a fresh module and a module-bound kernel constructor.
class BuilderUnitaryOpGroupingAnalysisTest : public ::testing::Test {
protected:
  void SetUp() override {
    loadTestDialects(context);
    module = OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
  }

  /// Create a kernel in the fixture module using its MLIR context.
  func::FuncOp createKernel(llvm::StringRef name,
                            ArrayRef<Type> inputTypes = {}) {
    OpBuilder builder(&context);
    return ::createKernel(*module, builder, name, inputTypes);
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

// Expected MLIR sections are schematic and omit types, sinks, and terminators
// unless they are material to the behavior under test.

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
//   group 0: unitaries [h, x], delimiters [mz]
//   group 1: unitaries [z], delimiters []
//   group 2: unitaries [rx], delimiters []
//   arith.constant is an unmapped hard boundary.
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
//       cc.continue
//     } else {
//       quake.z %q0 : (!quake.ref) -> ()
//       quake.reset %q0 : (!quake.ref) -> ()
//       cc.continue
//     }
//     return
//   }
//
// Expected analysis:
//   group 0 (then block): unitaries [h, x], delimiters [mz]
//   group 1 (else block): unitaries [z], delimiters [reset]
//   cc.if is an unmapped hard boundary.
//   The parent block has no groups; each branch contains one local group.
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

// Expected MLIR:
//
//   func.func @empty() attributes {"cudaq-kernel"} {
//     return
//   }
//
// Expected analysis:
//   The empty function produces no groups. Constructing the analysis with the
//   module operation instead of a func.func also produces no groups.
//   The return operation and null operation queries are unmapped.
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

// Expected MLIR:
//
//   quake.mx %q0
//   quake.reset %q0
//   quake.my %q1
//   quake.h %q0
//   quake.x %q1
//   quake.mz %q0
//   quake.reset %q1
//   quake.z %q0
//
// Expected analysis:
//   group 0: unitaries [], delimiters [mx, reset0, my]
//   group 1: unitaries [h, x], delimiters [mz, reset1]
//   group 2: unitaries [z], delimiters []
//   Consecutive leading delimiters form one delimiter-only group; consecutive
//   trailing delimiters remain in the group they terminate.
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

// Expected MLIR:
//
//   quake.h %q
//   arith.constant 0 : i64
//   arith.constant 1 : i64
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   Both constants are unmapped hard boundaries, and no empty group is
//   emitted between them.
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

// Expected MLIR:
//
//   ^bb0(%q0: !quake.wire, %q1: !quake.wire):
//   %m, %q0.next = quake.mz %q0
//   %q1.next = quake.x %q1
//
// Expected analysis:
//   ordering mode: textual
//   canonical order: [mz, x]
//   group 0: unitaries [], delimiters [mz]
//   group 1: unitaries [x], delimiters []
//   Wire block arguments have unknown logical identities, so the leading
//   measurement remains before the independent X gate.
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

// Expected MLIR:
//
//   %q0 = quake.null_wire
//   %q1 = quake.null_wire
//   %q2 = quake.null_wire
//   %q3 = quake.null_wire
//   %m, %q0.next = quake.mz %q0
//   %q1.next = quake.reset %q1
//   %q2.next = quake.z %q2
//   %q3.next = quake.h %q3
//
// Expected analysis:
//   segment order: [mz, reset, z, h]
//   canonical order: [z, h, mz, reset]
//   group 0: unitaries [z, h], delimiters [mz, reset]
//   All four operations are initially ready. Unitaries take priority over
//   delimiters, while original order breaks ties within each role.
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

// Expected MLIR:
//
//   %q0 = quake.null_wire
//   %q1 = quake.null_wire
//   %m, %q0.next = quake.mz %q0
//   %q0.final = quake.x %q0.next
//   %q1.next = quake.z %q1
//
// Expected analysis:
//   dependency: mz -> x
//   canonical order: [z, mz, x]
//   group 0: unitaries [z], delimiters [mz]
//   group 1: unitaries [x], delimiters []
//   The independent Z may move first, but X remains after its measurement
//   predecessor.
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

// Expected MLIR:
//
//   %q0 = quake.null_wire
//   %q1 = quake.null_wire
//   %q2 = quake.null_wire
//   %m0, %q0.next = quake.mz %q0
//   %m1, %q1.next = quake.mz %q1
//   %q0.final, %q1.final = quake.x [%q0.next] %q1.next
//   %q2.next = quake.z %q2
//
// Expected analysis:
//   dependencies: mz0 -> cx, mz1 -> cx
//   canonical order: [z, mz0, mz1, cx]
//   group 0: unitaries [z], delimiters [mz0, mz1]
//   group 1: unitaries [cx], delimiters []
//   The controlled X becomes ready only after both measurement predecessors
//   have been emitted.
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

// Expected MLIR:
//
//   %r = quake.alloca !quake.ref
//   %wireA = quake.unwrap %r
//   %wireB = quake.unwrap %r
//   %independent = quake.null_wire
//   %h = quake.h %wireA
//   %m, %measured = quake.mz %h
//   %z = quake.z %independent
//   %x = quake.x %wireB
//
// Expected analysis:
//   dependencies include h -> mz and mz -> x
//   canonical order: [h, z, mz, x]
//   group 0: unitaries [h, z], delimiters [mz]
//   group 1: unitaries [x], delimiters []
//   The identity edge mz -> x orders distinct SSA roots for the same logical
//   qubit. The alloca, unwrap, and null_wire setup operations are unmapped hard
//   boundaries.
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

// Expected MLIR:
//
//   %r = quake.alloca !quake.ref
//   %wireA = quake.unwrap %r
//   %wireB = quake.unwrap %r
//   %control, %target = quake.x [%wireA] %wireB
//
// Expected analysis:
//   canonical order: [controlledX]
//   group 0: unitaries [controlledX], delimiters []
//   Both operands have the same logical-qubit identity. Per-operation touch
//   deduplication prevents a self-edge and dependency-graph cycle.
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

// Expected MLIR:
//
//   quake.h %q
//   cc.scope {
//     quake.x %q
//     quake.mz %q
//     // Intentionally no cc.continue.
//   }
//   quake.z %q
//
// Expected analysis:
//   group 0 (parent block): unitaries [h], delimiters []
//   group 1 (scope block): unitaries [x], delimiters [mz]
//   group 2 (parent block): unitaries [z], delimiters []
//   cc.scope is an unmapped hard boundary. The nested block's final segment is
//   flushed at block end even without a terminator.
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
//   quake.h %q
//   %vec = quake.alloca !quake.veq<2>
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   quake.alloca is an unmapped hard boundary, so h and x cannot share a
//   group.
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
//   quake.h %q
//   %extracted = quake.extract_ref %vec[0]
//   quake.x %extracted
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   quake.extract_ref is an unmapped hard boundary.
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
//   quake.h %q
//   %extracted = quake.extract_ref %vec[%index]
//   quake.y %extracted
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [y], delimiters []
//   Dynamically indexed quake.extract_ref is an unmapped hard boundary.
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
//   quake.h %q
//   %sub = quake.subveq %vec, 1, 2
//   %extracted = quake.extract_ref %sub[0]
//   quake.x %extracted
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   quake.subveq and quake.extract_ref are unmapped hard boundaries.
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
//   quake.h %q
//   %relaxed = quake.relax_size %vec
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   quake.relax_size is an unmapped hard boundary.
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
//   quake.h %q0
//   %merged = quake.concat %q1, %vec
//   quake.x %q0
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   quake.concat is an unmapped hard boundary.
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
//   quake.h %q
//   %size = quake.veq_size %vec
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   quake.veq_size is an unmapped hard boundary.
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
//   quake.h %q
//   %results = quake.mz %vec
//   quake.x %q
//
// Expected analysis:
//   ordering mode: textual
//   group 0: unitaries [h], delimiters [mz]
//   group 1: unitaries [x], delimiters []
//   The non-scalar measurement remains a delimiter even though its segment
//   cannot use scalar-wire ordering.
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
//   quake.h %q
//   quake.mx %q
//   quake.x %q
//   quake.my %q
//   quake.z %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters [mx]
//   group 1: unitaries [x], delimiters [my]
//   group 2: unitaries [z], delimiters []
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
//   quake.h %q
//   quake.reset %q
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters [reset]
//   group 1: unitaries [x], delimiters []
//   Reset is a group member but is classified as a delimiter, not a unitary.
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
//   quake.x [%ctrl] %target
//   quake.y [%ctrl] %target
//   quake.z [%ctrl] %target
//
// Expected analysis:
//   ordering mode: textual
//   group 0: unitaries [x, y, z], delimiters []
//   Gates with vector controls remain unitary even though the segment cannot
//   use scalar-wire ordering.
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
//   quake.r1 (%theta) %q0
//   quake.rx (%theta) %q0
//   quake.phased_rx (%theta, %phi) %q0
//   quake.ry (%phi) %q0
//   quake.rz (%lambda) %q0
//   quake.u2 (%theta, %phi) %q0
//   quake.u3 (%theta, %phi, %lambda) %q0
//   quake.swap %q0, %q1
//
// Expected analysis:
//   group 0: unitaries [r1, rx, phasedRx, ry, rz, u2, u3, swap],
//            delimiters []
//   Classical parameters and multiple targets do not split a unitary run.
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
//   quake.h %q
//   quake.exp_pauli (%theta) %vec to "XYZ"
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h, expPauli, x], delimiters []
//   ExpPauli remains unitary when its target is a vector.
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
//   quake.h %q
//   quake.compute_action %compute, %action
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   quake.compute_action is an unmapped hard boundary.
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
//   quake.h %q0
//   cc.scope {
//     quake.x %q0
//     quake.y %q1
//     cc.continue
//   }
//   quake.z %q0
//
// Expected analysis:
//   group 0 (parent block): unitaries [h], delimiters []
//   group 1 (scope block): unitaries [x, y], delimiters []
//   group 2 (parent block): unitaries [z], delimiters []
//   cc.scope is an unmapped hard boundary; groups do not cross into or out of
//   its nested block.
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
//   %slot = cc.alloca i32
//   quake.h %q
//   cc.store %value, %slot
//   %loaded = cc.load %slot
//   %extended = cc.cast signed %loaded : (i32) -> i64
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   The cc.alloca, cc.store, cc.load, and cc.cast operations are unmapped hard
//   boundaries.
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
//   %c0 = arith.constant 0 : i64
//   quake.h %q
//   %c1 = arith.constant 1 : i64
//   %sum = arith.addi %c0, %c1 : i64
//   func.call @callee(%sum) : (i64) -> ()
//   quake.x %q
//
// Expected analysis:
//   group 0: unitaries [h], delimiters []
//   group 1: unitaries [x], delimiters []
//   The arithmetic and call operations are unmapped hard boundaries.
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
