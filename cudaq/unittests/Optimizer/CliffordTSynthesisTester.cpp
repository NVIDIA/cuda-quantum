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

// Operator-norm proxy distance between two 2x2 unitaries, modulo a global
// phase. Consistent with cudaq::synth::rz_approximation_error (the metric
// behind rz_gate_sequence_error). Both compute sqrt(|det(A - B)|), which for
// 2x2 matrices equals sqrt(sigma_max * sigma_min) and tracks the operator
// norm in the small-error regime where the two singular values of (A - B)
// are nearly equal.
//
// Phase alignment is required here because CliffordTSynthesis drops the W
// (global phase) gates emitted by gridsynth, so the reconstructed unitary
// can differ from the ideal by a global phase even when synthesis succeeds.
double distance(const M2 &A, const M2 &B) {
  const std::complex<double> inner =
      std::conj(A.a) * B.a + std::conj(A.b) * B.b + std::conj(A.c) * B.c +
      std::conj(A.d) * B.d;
  const double mag = std::abs(inner);
  const std::complex<double> phase =
      mag > 0.0 ? inner / mag : std::complex<double>(1.0, 0.0);
  const M2 Aligned = {phase * A.a, phase * A.b, phase * A.c, phase * A.d};
  const std::complex<double> e00 = Aligned.a - B.a;
  const std::complex<double> e01 = Aligned.b - B.b;
  const std::complex<double> e10 = Aligned.c - B.c;
  const std::complex<double> e11 = Aligned.d - B.d;
  const std::complex<double> det = e00 * e11 - e01 * e10;
  return std::sqrt(std::abs(det));
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
  EXPECT_LT(distance(U, expected), opts.epsilon);
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

namespace cudaq::opt::detail {
uint64_t lastCliffordTSynthCacheHits();
uint64_t lastCliffordTSynthCacheUniqueAngles();
} // namespace cudaq::opt::detail

namespace {

// Module with one Rz op per (theta) entry.
// Each op gets its own qubit so the rewriter can lower them independently.
OwningOpRef<ModuleOp> buildRzListModule(MLIRContext *context,
                                        llvm::ArrayRef<double> thetas) {
  OpBuilder builder(context);
  auto loc = builder.getUnknownLoc();
  auto module = ModuleOp::create(builder, loc);
  builder.setInsertionPointToEnd(module.getBody());

  auto funcType = builder.getFunctionType({}, {});
  auto func = func::FuncOp::create(builder, loc, "rz_dedup_test", funcType);
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto refType = cudaq::quake::RefType::get(context);
  for (double theta : thetas) {
    Value q = cudaq::quake::AllocaOp::create(builder, loc, refType);
    Value angle = cudaq::opt::factory::createFloatConstant(
        loc, builder, theta, builder.getF64Type());
    cudaq::quake::RzOp::create(builder, loc, /*isAdj=*/false, ValueRange{angle},
                               ValueRange{}, q);
  }

  func::ReturnOp::create(builder, loc);
  return OwningOpRef<ModuleOp>(module);
}

class CliffordTSynthesisCacheTest : public ::testing::Test {
protected:
  void SetUp() override {
    context = std::make_unique<MLIRContext>();
    context->loadDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                         func::FuncDialect, cudaq::quake::QuakeDialect>();
  }

  std::unique_ptr<MLIRContext> context;
};

// 5 Rz(pi/4) + 3 Rz(pi/3) -> gridsynth runs twice. The remaining six
// rotations should be served from the cache.
TEST_F(CliffordTSynthesisCacheTest, RepeatedAnglesAreDeduplicated) {
  const double a = M_PI / 4.0;
  const double b = M_PI / 3.0;
  auto module = buildRzListModule(context.get(), {a, a, a, a, a, b, b, b});

  PassManager pm(context.get());
  cudaq::opt::CliffordTSynthesisOptions opts;
  opts.epsilon = 1e-3;
  pm.addPass(cudaq::opt::createCliffordTSynthesis(opts));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  EXPECT_EQ(cudaq::opt::detail::lastCliffordTSynthCacheUniqueAngles(), 2);
  EXPECT_EQ(cudaq::opt::detail::lastCliffordTSynthCacheHits(), 6);
}

// All distinct angles produce zero cache hits
TEST_F(CliffordTSynthesisCacheTest, DistinctAnglesProduceNoCacheHits) {
  auto module = buildRzListModule(
      context.get(), {M_PI / 4.0, M_PI / 5.0, M_PI / 6.0, M_PI / 7.0});

  PassManager pm(context.get());
  cudaq::opt::CliffordTSynthesisOptions opts;
  opts.epsilon = 1e-3;
  pm.addPass(cudaq::opt::createCliffordTSynthesis(opts));
  ASSERT_TRUE(succeeded(pm.run(*module)));

  EXPECT_EQ(cudaq::opt::detail::lastCliffordTSynthCacheUniqueAngles(), 4);
  EXPECT_EQ(cudaq::opt::detail::lastCliffordTSynthCacheHits(), 0);
}

} // namespace
