/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>

using namespace mlir;

namespace {

// Row-major 2x2 complex matrix.
struct M2 {
  std::complex<double> a, b, c, d;
};

M2 mul(const M2 &L, const M2 &R) {
  return {L.a * R.a + L.b * R.c, L.a * R.b + L.b * R.d, L.c * R.a + L.d * R.c,
          L.c * R.b + L.d * R.d};
}

const std::complex<double> kI{0.0, 1.0};
const double kInvSqrt2 = 1.0 / std::sqrt(2.0);

const M2 kH = {kInvSqrt2, kInvSqrt2, kInvSqrt2, -kInvSqrt2};
const M2 kS = {1.0, 0.0, 0.0, kI};
const M2 kSdg = {1.0, 0.0, 0.0, std::conj(kI)};
const M2 kT = {1.0, 0.0, 0.0, std::polar(1.0, M_PI / 4.0)};
const M2 kTdg = {1.0, 0.0, 0.0, std::polar(1.0, -M_PI / 4.0)};
const M2 kX = {0.0, 1.0, 1.0, 0.0};

// Create a module containing `func.func @rz_test() { %q = quake.alloca;
// %a = arith.constant theta; quake.rz(%a) %q; return }`.
OwningOpRef<ModuleOp> buildRzModule(MLIRContext *context, double theta) {
  OpBuilder builder(context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToEnd(module.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto func = func::FuncOp::create(builder, loc, "rz_test", funcType);
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto refType = cudaq::quake::RefType::get(context);
  Value q = cudaq::quake::AllocaOp::create(builder, loc, refType);
  Value angle = cudaq::opt::factory::createFloatConstant(loc, builder, theta,
                                                         builder.getF64Type());

  cudaq::quake::RzOp::create(builder, loc, /*isAdj=*/false, ValueRange{angle},
                             /*controls=*/ValueRange{}, q);

  func::ReturnOp::create(builder, loc);
  return OwningOpRef<ModuleOp>(module);
}

M2 reconstructUnitary(ModuleOp module) {
  M2 U = {1.0, 0.0, 0.0, 1.0};
  module.walk([&](Operation *op) {
    if (isa<cudaq::quake::HOp>(op)) {
      U = mul(kH, U);
    } else if (auto s = dyn_cast<cudaq::quake::SOp>(op)) {
      U = mul(s.isAdj() ? kSdg : kS, U);
    } else if (auto t = dyn_cast<cudaq::quake::TOp>(op)) {
      U = mul(t.isAdj() ? kTdg : kT, U);
    } else if (isa<cudaq::quake::XOp>(op)) {
      U = mul(kX, U);
    }
  });
  return U;
}

class CliffordTSynthesisTester : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    context->loadDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                         func::FuncDialect, cudaq::quake::QuakeDialect>();
  }

  std::unique_ptr<MLIRContext> context;
};

TEST_F(CliffordTSynthesisTester, RzRoundTripMatchesIdealUpToGlobalPhase) {
  const double theta = M_PI / 4.0;
  auto module = buildRzModule(context.get(), theta);

  PassManager pm(context.get());
  cudaq::opt::CliffordTSynthesisOptions opts;
  opts.epsilon = 1e-8;
  pm.addPass(cudaq::opt::createCliffordTSynthesis(opts));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  // No rotation ops should survive.
  module->walk([](Operation *op) {
    EXPECT_FALSE((isa<cudaq::quake::RxOp, cudaq::quake::RyOp,
                      cudaq::quake::RzOp, cudaq::quake::R1Op>(op)))
        << "rotation gate survived synthesis: "
        << op->getName().getStringRef().str();
  });

  M2 U = reconstructUnitary(*module);

  // Expected R_z(theta) = diag(e^{-i theta/2}, e^{i theta/2}).
  const std::complex<double> expectedA = std::polar(1.0, -theta / 2.0);
  const std::complex<double> expectedD = std::polar(1.0, theta / 2.0);

  const std::complex<double> globalPhase = U.a / expectedA;
  const std::complex<double> normalized = globalPhase / std::abs(globalPhase);

  const std::complex<double> alignedA = U.a / normalized;
  const std::complex<double> alignedB = U.b / normalized;
  const std::complex<double> alignedC = U.c / normalized;
  const std::complex<double> alignedD = U.d / normalized;

  const double tol = 1e-6;
  EXPECT_LT(std::abs(alignedA - expectedA), tol);
  EXPECT_LT(std::abs(alignedB), tol);
  EXPECT_LT(std::abs(alignedC), tol);
  EXPECT_LT(std::abs(alignedD - expectedD), tol);
}

} // namespace
