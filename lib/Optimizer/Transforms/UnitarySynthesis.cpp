/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_UNITARYSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "unitary-synthesis"

using namespace mlir;

namespace {

/// TODO: Refactor common code from 'StatePreparation' pass
std::vector<std::complex<double>>
readGlobalConstantArray(cudaq::cc::GlobalOp &global) {
  std::vector<std::complex<double>> result{};

  auto attr = global.getValue();
  auto elementsAttr = cast<mlir::ElementsAttr>(attr.value());
  auto eleTy = elementsAttr.getElementType();
  auto values = elementsAttr.getValues<mlir::Attribute>();

  for (auto it = values.begin(); it != values.end(); ++it) {
    auto valAttr = *it;

    auto v = [&]() -> std::complex<double> {
      assert(isa<ComplexType>(eleTy));
      auto arrayAttr = cast<mlir::ArrayAttr>(valAttr);
      auto real = cast<FloatAttr>(arrayAttr[0]).getValue().convertToDouble();
      auto imag = cast<FloatAttr>(arrayAttr[1]).getValue().convertToDouble();
      return {real, imag};
    }();

    result.push_back(v);
  }
  return result;
}

struct EulerAngles {
  double alpha;
  double beta;
  double gamma;
};

struct BasisZYZ {

  std::array<std::complex<double>, 4> matrix;
  EulerAngles angles;
  double phase;

  /// This logic is based on https://arxiv.org/pdf/quant-ph/9503016 and its
  /// corresponding explanation in https://threeplusone.com/pubs/on_gates.pdf,
  /// Section 4.
  void decompose() {
    auto det = (matrix[0] * matrix[3]) - (matrix[1] * matrix[2]);
    phase = 0.5 * std::atan2(det.imag(), det.real());

    auto abs_00 = std::abs(matrix[0]);
    auto abs_01 = std::abs(matrix[1]);

    if (abs_00 >= abs_01)
      angles.beta = 2 * std::acos(abs_00);
    else
      angles.beta = 2 * std::asin(abs_01);

    double half_sum_alpha_gamma = 0.;
    double divisor = std::cos(angles.beta / 2.);
    if (0. != divisor) {
      auto arg = matrix[3] / divisor;
      half_sum_alpha_gamma = std::atan2(arg.imag(), arg.real());
    }

    double half_diff_alpha_gamma = 0.;
    divisor = std::sin(angles.beta / 2.);
    if (0. != divisor) {
      auto arg = matrix[2] / divisor;
      half_diff_alpha_gamma = std::atan2(arg.imag(), arg.real());
    }

    angles.alpha = half_sum_alpha_gamma + half_diff_alpha_gamma;
    angles.gamma = half_sum_alpha_gamma - half_diff_alpha_gamma;

    /// ASKME: Normalize all angles?
  }

  BasisZYZ(std::vector<std::complex<double>> vec) {
    std::move(vec.begin(), vec.end(), matrix.begin());
  }
};

class CustomUnitaryPattern
    : public OpRewritePattern<quake::CustomUnitarySymbolOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::CustomUnitarySymbolOp customOp,
                                PatternRewriter &rewriter) const override {

    // Fetch the unitary matrix generator for this custom operation
    auto parentModule = customOp->getParentOfType<ModuleOp>();
    auto sref = customOp.getGenerator();
    StringRef generatorName = sref.getRootReference();
    auto globalOp =
        parentModule.lookupSymbol<cudaq::cc::GlobalOp>(generatorName);
    auto unitary = readGlobalConstantArray(globalOp);

    if (unitary.size() != 4)
      return customOp.emitError(
          "Decomposition of only single qubit custom operations supported.");

    /// TODO: Maintain a cache of decomposed custom operations

    // Use Euler angle decomposition for single qubit operation
    auto zyz = BasisZYZ(unitary);
    zyz.decompose();

    // op info
    Location loc = customOp->getLoc();
    auto targets = customOp.getTargets();
    auto controls = customOp.getControls();

    /// TODO: Handle adjoint case

    auto floatTy = cast<FloatType>(rewriter.getF64Type());

    if (0. != zyz.angles.alpha) {
      auto alpha = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, zyz.angles.alpha, floatTy);
      rewriter.create<quake::RzOp>(loc, alpha, controls, targets);
    }

    if (0. != zyz.angles.beta) {
      auto beta = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, zyz.angles.beta, floatTy);
      rewriter.create<quake::RyOp>(loc, beta, controls, targets);
    }

    if (0. != zyz.angles.gamma) {
      auto gamma = cudaq::opt::factory::createFloatConstant(
          loc, rewriter, zyz.angles.gamma, floatTy);
      rewriter.create<quake::RzOp>(loc, gamma, controls, targets);
    }

    /// ASKME: Apply global phase?
    // if (0. != zyz.phase) {
    //   auto phase = cudaq::opt::factory::createFloatConstant(loc, rewriter,
    //                                                         zyz.phase,
    //                                                         floatTy);
    //   Value negPhase = rewriter.create<arith::NegFOp>(loc, phase);
    //   rewriter.create<quake::R1Op>(loc, phase, controls, targets);
    //   rewriter.create<quake::RzOp>(loc, negPhase, controls, targets);
    // }

    rewriter.eraseOp(customOp);
    return success();
  }
};

class UnitarySynthesisPass
    : public cudaq::opt::impl::UnitarySynthesisBase<UnitarySynthesisPass> {
public:
  using UnitarySynthesisBase::UnitarySynthesisBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<CustomUnitaryPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
