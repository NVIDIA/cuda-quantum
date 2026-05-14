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
#include <algorithm>
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

enum class Rot { Rx, Ry, Rz, R1 };

std::string nameOf(Rot gate) {
  switch (gate) {
  case Rot::Rx:
    return "Rx";
  case Rot::Ry:
    return "Ry";
  case Rot::Rz:
    return "Rz";
  case Rot::R1:
    return "R1";
  }
  __builtin_unreachable();
}

// Ideal 2x2 unitary for each single-qubit rotation.
M2 idealUnitary(Rot gate, double theta) {
  const double cs = std::cos(theta / 2.0);
  const double sn = std::sin(theta / 2.0);
  switch (gate) {
  case Rot::Rx:
    return {cs, std::complex<double>(0.0, -sn), std::complex<double>(0.0, -sn),
            cs};
  case Rot::Ry:
    return {cs, -sn, sn, cs};
  case Rot::Rz:
    return {std::polar(1.0, -theta / 2.0), 0.0, 0.0,
            std::polar(1.0, theta / 2.0)};
  case Rot::R1:
    return {1.0, 0.0, 0.0, std::polar(1.0, theta)};
  }
  __builtin_unreachable();
}

// Build a module containing one rotation op on a single qubit.
OwningOpRef<ModuleOp> buildRotationModule(MLIRContext *context, Rot gate,
                                          double theta) {
  OpBuilder builder(context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToEnd(module.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto func = func::FuncOp::create(builder, loc, "rot_test", funcType);
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto refType = cudaq::quake::RefType::get(context);
  Value q = cudaq::quake::AllocaOp::create(builder, loc, refType);
  Value angle = cudaq::opt::factory::createFloatConstant(loc, builder, theta,
                                                         builder.getF64Type());
  switch (gate) {
  case Rot::Rx:
    cudaq::quake::RxOp::create(builder, loc, /*isAdj=*/false, ValueRange{angle},
                               ValueRange{}, q);
    break;
  case Rot::Ry:
    cudaq::quake::RyOp::create(builder, loc, /*isAdj=*/false, ValueRange{angle},
                               ValueRange{}, q);
    break;
  case Rot::Rz:
    cudaq::quake::RzOp::create(builder, loc, /*isAdj=*/false, ValueRange{angle},
                               ValueRange{}, q);
    break;
  case Rot::R1:
    cudaq::quake::R1Op::create(builder, loc, /*isAdj=*/false, ValueRange{angle},
                               ValueRange{}, q);
    break;
  }

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

struct RotationCase {
  Rot gate;
  double theta;
};

class CliffordTSynthesisRotationTest
    : public ::testing::TestWithParam<RotationCase> {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    context->loadDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                         func::FuncDialect, cudaq::quake::QuakeDialect>();
  }

  std::unique_ptr<MLIRContext> context;
};

TEST_P(CliffordTSynthesisRotationTest, RoundTripMatchesIdealUpToGlobalPhase) {
  const auto &param = GetParam();
  auto module = buildRotationModule(context.get(), param.gate, param.theta);

  PassManager pm(context.get());
  cudaq::opt::CliffordTSynthesisOptions opts;
  opts.epsilon = 1e-8;
  pm.addPass(cudaq::opt::createCliffordTSynthesis(opts));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  module->walk([](Operation *op) {
    EXPECT_FALSE((isa<cudaq::quake::RxOp, cudaq::quake::RyOp,
                      cudaq::quake::RzOp, cudaq::quake::R1Op>(op)))
        << "rotation gate survived synthesis: "
        << op->getName().getStringRef().str();
  });

  const M2 U = reconstructUnitary(*module);
  const M2 expected = idealUnitary(param.gate, param.theta);

  const std::complex<double> inner =
      U.a * std::conj(expected.a) + U.b * std::conj(expected.b) +
      U.c * std::conj(expected.c) + U.d * std::conj(expected.d);
  EXPECT_NEAR(std::abs(inner), 2.0, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(
    Rotations, CliffordTSynthesisRotationTest,
    ::testing::Values(RotationCase{Rot::Rz, M_PI / 4.0},
                      RotationCase{Rot::Rz, M_PI / 3.0},
                      RotationCase{Rot::Rx, M_PI / 4.0},
                      RotationCase{Rot::Ry, M_PI / 4.0},
                      RotationCase{Rot::R1, 2.0 * M_PI / 7.0}),
    [](const ::testing::TestParamInfo<RotationCase> &info) {
      std::string theta = std::to_string(info.param.theta);
      std::replace(theta.begin(), theta.end(), '.', '_');
      std::replace(theta.begin(), theta.end(), '-', 'n');
      return nameOf(info.param.gate) + "_" + theta;
    });

} // namespace
