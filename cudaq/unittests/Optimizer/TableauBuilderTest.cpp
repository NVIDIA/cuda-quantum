/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "gtest/gtest.h"
#include "cudaq/Optimizer/Analysis/CircuitValidation.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <cmath>

using namespace mlir;

using cudaq::opt::compareTableaux;

namespace {

class TableauBuilderTest : public ::testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect>();
    context.loadDialect<func::FuncDialect>();
    context.loadDialect<cudaq::cc::CCDialect>();
    context.loadDialect<cudaq::quake::QuakeDialect>();
    module = OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
  }

  func::FuncOp createKernel(llvm::StringRef name, ArrayRef<Type> inputTypes,
                            OpBuilder &builder) {
    Location loc = builder.getUnknownLoc();
    builder.setInsertionPointToEnd(module->getBody());
    auto funcTy = builder.getFunctionType(inputTypes, {});
    auto func = func::FuncOp::create(builder, loc, name, funcTy);
    func->setAttr("cudaq-kernel", builder.getUnitAttr());
    func.addEntryBlock();
    builder.setInsertionPointToStart(&func.front());
    return func;
  }

  MLIRContext context;
  OwningOpRef<ModuleOp> module;
};

} // namespace

// Two identical Clifford kernels have identical tableaux.
TEST_F(TableauBuilderTest, IdenticalCliffordEquivalent) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy, refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, base.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, ValueRange{base.getArgument(0)},
                            ValueRange{base.getArgument(1)});
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy, refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, ValueRange{cand.getArgument(0)},
                            ValueRange{cand.getArgument(1)});
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.equivalent);
}

// H·Z·H = X: a non-trivial Clifford identity is recognized as equivalent.
TEST_F(TableauBuilderTest, HZHEqualsX) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::ZOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::HOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.equivalent);
}

// rz(pi/2) folds to S, so the two kernels are equivalent.
TEST_F(TableauBuilderTest, RzHalfPiEqualsS) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::SOp::create(builder, loc, /*is_adj=*/false, ValueRange{},
                            ValueRange{}, ValueRange{base.getArgument(0)});
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy}, builder);
  Value halfPi = arith::ConstantFloatOp::create(
      builder, loc, builder.getF64Type(), llvm::APFloat(M_PI_2));
  cudaq::quake::RzOp::create(builder, loc, halfPi, ValueRange{},
                             ValueRange{cand.getArgument(0)});
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.equivalent);
}

// A global-phase difference (Z·X·Z·X = -I vs identity) is invisible to a
// stabilizer tableau, so the two kernels compare equal (the same
// up-to-global-phase acceptance signal as the dense-unitary oracle).
TEST_F(TableauBuilderTest, GlobalPhaseDifferenceIsEquivalent) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  func::ReturnOp::create(builder, loc); // identity on one qubit

  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::ZOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::ZOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.equivalent);
}

// Genuinely different Clifford circuits (H vs X) are not equivalent, but the
// comparison still succeeds.
TEST_F(TableauBuilderTest, DifferentCircuitsNotEquivalent) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_FALSE(result.equivalent);
}

// Kernels on different numbers of qubits are compared by padding the narrower
// one with identity. A wider kernel that leaves the extra qubit alone is the
// narrower one tensored with I, and certifies at the borrowed guarantee.
TEST_F(TableauBuilderTest, WidthMismatchComparedAsTensorWithIdentity) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy, refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.error.empty());
  EXPECT_TRUE(result.equivalent);
  EXPECT_EQ(result.guarantee,
            cudaq::opt::EquivalenceGuarantee::BorrowedAncilla);
}

// The same padding must not wave through a kernel that actually uses its extra
// qubit: entangling it with the system is not a tensor with anything.
TEST_F(TableauBuilderTest, WidthMismatchRejectsUsedExtraQubit) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy, refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, ValueRange{cand.getArgument(0)},
                            ValueRange{cand.getArgument(1)});
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_FALSE(result.equivalent);
}

// A negated control is the control conjugated by X, so `x [neg] q0, q1` must
// build the same tableau as X(q0); CX(q0, q1); X(q0).
TEST_F(TableauBuilderTest, NegatedControlConjugatedByX) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy, refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, ValueRange{base.getArgument(0)},
                            ValueRange{base.getArgument(1)});
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy, refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, /*is_adj=*/false, ValueRange{},
                            ValueRange{cand.getArgument(0)},
                            ValueRange{cand.getArgument(1)},
                            builder.getDenseBoolArrayAttr({true}));
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.equivalent);
}

// The negation is not silently dropped: a negated-control CX differs from a
// plain CX.
TEST_F(TableauBuilderTest, NegatedControlDiffersFromPlainControl) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy, refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, ValueRange{base.getArgument(0)},
                            ValueRange{base.getArgument(1)});
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy, refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, /*is_adj=*/false, ValueRange{},
                            ValueRange{cand.getArgument(0)},
                            ValueRange{cand.getArgument(1)},
                            builder.getDenseBoolArrayAttr({true}));
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_FALSE(result.equivalent);
}

// An op that touches a qubit but has no tableau encoding (here quake.reset) is
// reported as a build failure rather than being dropped from the circuit.
TEST_F(TableauBuilderTest, UnsupportedQuantumOpFailsBuild) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareTableaux(base, cand);
  EXPECT_FALSE(result.computed);
  EXPECT_FALSE(result.error.empty());
}

// Build an n-qubit GHZ-style Clifford circuit (H on qubit 0, then a CX chain)
// into the given kernel.
static void buildGhz(func::FuncOp func, OpBuilder &builder, unsigned n,
                     unsigned cxChainLen) {
  Location loc = builder.getUnknownLoc();
  Value veq = func.getArgument(0);
  Value q0 = cudaq::quake::ExtractRefOp::create(builder, loc, veq,
                                                static_cast<std::size_t>(0));
  cudaq::quake::HOp::create(builder, loc, q0);
  for (unsigned i = 0; i < cxChainLen; ++i) {
    Value ci = cudaq::quake::ExtractRefOp::create(builder, loc, veq,
                                                  static_cast<std::size_t>(i));
    Value ti = cudaq::quake::ExtractRefOp::create(
        builder, loc, veq, static_cast<std::size_t>(i + 1));
    cudaq::quake::XOp::create(builder, loc, ValueRange{ci}, ValueRange{ti});
  }
  func::ReturnOp::create(builder, loc);
}

// A 30-qubit Clifford circuit is far past the dense-unitary bound (a 2^30 x
// 2^30 matrix is infeasible), yet the tableau oracle compares it exactly. This
// is the reason the Clifford oracle exists.
TEST_F(TableauBuilderTest, LargeCliffordBeyondDenseBound) {
  OpBuilder builder(&context);
  auto veqTy = cudaq::quake::VeqType::get(&context, 30);

  auto base = createKernel("base", {veqTy}, builder);
  buildGhz(base, builder, 30, /*cxChainLen=*/29);

  auto cand = createKernel("cand", {veqTy}, builder);
  buildGhz(cand, builder, 30, /*cxChainLen=*/29);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.equivalent);
}

// The oracle also detects inequivalence at scale: dropping one CX from the
// candidate's chain makes the 30-qubit circuits differ.
TEST_F(TableauBuilderTest, LargeCliffordDetectsDifference) {
  OpBuilder builder(&context);
  auto veqTy = cudaq::quake::VeqType::get(&context, 30);

  auto base = createKernel("base", {veqTy}, builder);
  buildGhz(base, builder, 30, /*cxChainLen=*/29);

  auto cand = createKernel("cand", {veqTy}, builder);
  buildGhz(cand, builder, 30, /*cxChainLen=*/28);

  auto result = compareTableaux(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_FALSE(result.equivalent);
}
