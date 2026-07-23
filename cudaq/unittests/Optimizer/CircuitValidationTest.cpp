/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CircuitValidation.h"
#include "gtest/gtest.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include <cmath>

using namespace mlir;

using cudaq::opt::BoundedUnitaryDomainStatus;
using cudaq::opt::checkBoundedUnitaryDomain;
using cudaq::opt::compareUnitaries;
using cudaq::opt::DomainRejectionKind;

static void loadTestDialects(MLIRContext &context) {
  context.loadDialect<arith::ArithDialect>();
  context.loadDialect<func::FuncDialect>();
  context.loadDialect<cudaq::cc::CCDialect>();
  context.loadDialect<cudaq::quake::QuakeDialect>();
}

static bool hasKind(const BoundedUnitaryDomainStatus &status,
                    DomainRejectionKind kind) {
  for (const auto &r : status.rejections)
    if (r.kind == kind)
      return true;
  return false;
}

class CircuitValidationTest : public ::testing::Test {
protected:
  void SetUp() override {
    loadTestDialects(context);
    module = OwningOpRef<ModuleOp>(ModuleOp::create(UnknownLoc::get(&context)));
  }

  /// Create a `func.func` kernel and return a builder positioned at its entry.
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

// A straight-line unitary kernel (h, x, rx) on two qubits is in the domain.
TEST_F(CircuitValidationTest, AcceptsStraightLineUnitary) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func =
      createKernel("kern", {refTy, refTy, builder.getF64Type()}, builder);
  Location loc = builder.getUnknownLoc();
  Value q0 = func.getArgument(0);
  Value q1 = func.getArgument(1);
  Value theta = func.getArgument(2);
  cudaq::quake::HOp::create(builder, loc, q0);
  cudaq::quake::XOp::create(builder, loc, q1);
  cudaq::quake::RxOp::create(builder, loc, ValueRange{theta}, ValueRange{},
                             ValueRange{q1});
  func::ReturnOp::create(builder, loc);

  auto status = checkBoundedUnitaryDomain(*module);
  EXPECT_TRUE(status.supported);
  EXPECT_TRUE(status.rejections.empty());
  EXPECT_EQ(status.maxQubits, 2u);
}

// A measurement pushes the kernel out of the bounded-unitary domain.
TEST_F(CircuitValidationTest, RejectsMeasurement) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("kern", {refTy}, builder);
  Location loc = builder.getUnknownLoc();
  Value q0 = func.getArgument(0);
  cudaq::quake::HOp::create(builder, loc, q0);
  auto measureTy = cudaq::cc::MeasureHandleType::get(&context);
  cudaq::quake::MzOp::create(builder, loc, TypeRange{measureTy}, ValueRange{q0},
                             StringAttr{});
  func::ReturnOp::create(builder, loc);

  auto status = checkBoundedUnitaryDomain(*module);
  EXPECT_FALSE(status.supported);
  EXPECT_TRUE(hasKind(status, DomainRejectionKind::Measurement));
}

// A reset is not a unitary operation.
TEST_F(CircuitValidationTest, RejectsReset) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  auto func = createKernel("kern", {refTy}, builder);
  Location loc = builder.getUnknownLoc();
  Value q0 = func.getArgument(0);
  cudaq::quake::ResetOp::create(builder, loc, TypeRange{}, q0);
  func::ReturnOp::create(builder, loc);

  auto status = checkBoundedUnitaryDomain(*module);
  EXPECT_FALSE(status.supported);
  EXPECT_TRUE(hasKind(status, DomainRejectionKind::Reset));
}

// A dynamically-sized register has an unknowable qubit count.
TEST_F(CircuitValidationTest, RejectsDynamicRegister) {
  OpBuilder builder(&context);
  auto veqTy = cudaq::quake::VeqType::getUnsized(&context);
  auto func = createKernel("kern", {veqTy}, builder);
  func::ReturnOp::create(builder, builder.getUnknownLoc());

  auto status = checkBoundedUnitaryDomain(*module);
  EXPECT_FALSE(status.supported);
  EXPECT_TRUE(hasKind(status, DomainRejectionKind::DynamicQubitRegister));
}

// A kernel wider than the exact-unitary bound is rejected. The tally is
// reported regardless.
TEST_F(CircuitValidationTest, RejectsTooManyQubits) {
  OpBuilder builder(&context);
  auto veqTy = cudaq::quake::VeqType::get(&context, 4);
  auto func = createKernel("kern", {veqTy}, builder);
  func::ReturnOp::create(builder, builder.getUnknownLoc());

  auto status = checkBoundedUnitaryDomain(*module, /*exactQubitBound=*/2);
  EXPECT_FALSE(status.supported);
  EXPECT_TRUE(hasKind(status, DomainRejectionKind::TooManyQubits));
  EXPECT_EQ(status.maxQubits, 4u);
}

// Two kernels with the same gate sequence have identical unitaries.
TEST_F(CircuitValidationTest, CompareIdenticalKernels) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareUnitaries(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.strictEqual);
  EXPECT_TRUE(result.equalUpToGlobalPhase);
  EXPECT_TRUE(result.phaseIsZero);
}

// A pair of self-cancelling gates (x·x = I) leaves the unitary unchanged.
TEST_F(CircuitValidationTest, CompareCancellingGatesAreEqual) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareUnitaries(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_TRUE(result.strictEqual);
}

// Two circuits that differ only by a global phase (x·z·x·z = -I vs identity)
// are rejected by the strict oracle but accepted up to a global phase. This is
// exactly the distinction between the two exact-tier oracles, so the phase
// delta (here pi) must be reported.
TEST_F(CircuitValidationTest, CompareGlobalPhaseDifference) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  // base = x·x = I
  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  // cand = x·z·x·z = -I (identity times a global phase of e^{i*pi})
  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::ZOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::ZOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareUnitaries(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_FALSE(result.strictEqual);
  EXPECT_TRUE(result.equalUpToGlobalPhase);
  EXPECT_FALSE(result.phaseIsZero);
  EXPECT_NEAR(std::abs(result.phase), M_PI, 1e-6);
}

// Genuinely different circuits are not equivalent, but the comparison still
// succeeds (computed == true).
TEST_F(CircuitValidationTest, CompareDifferentKernels) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::HOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto result = compareUnitaries(base, cand);
  EXPECT_TRUE(result.computed);
  EXPECT_FALSE(result.strictEqual);
  EXPECT_FALSE(result.equalUpToGlobalPhase);
}

// Kernels on different numbers of qubits cannot be compared; the result reports
// computed == false rather than a false equivalence.
TEST_F(CircuitValidationTest, CompareDimensionMismatch) {
  OpBuilder builder(&context);
  auto refTy = builder.getType<cudaq::quake::RefType>();
  Location loc = builder.getUnknownLoc();

  auto base = createKernel("base", {refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, base.getArgument(0));
  func::ReturnOp::create(builder, loc);

  auto cand = createKernel("cand", {refTy, refTy}, builder);
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(0));
  cudaq::quake::XOp::create(builder, loc, cand.getArgument(1));
  func::ReturnOp::create(builder, loc);

  auto result = compareUnitaries(base, cand);
  EXPECT_FALSE(result.computed);
  EXPECT_FALSE(result.error.empty());
}
