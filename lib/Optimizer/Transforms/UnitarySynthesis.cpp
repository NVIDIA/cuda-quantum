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

namespace cudaq {
std::vector<std::complex<double>>
readGlobalConstantArray(cudaq::cc::GlobalOp &global);
}

namespace {

struct EulerAngles {
  double alpha;
  double beta;
  double gamma;
};

struct BasisZYZ {

  std::array<std::complex<double>, 4> matrix;
  EulerAngles angles;
  /// Updates to the global phase
  double phase;

  /// This logic is based on https://arxiv.org/pdf/quant-ph/9503016 and its
  /// corresponding explanation in https://threeplusone.com/pubs/on_gates.pdf,
  /// Section 4.
  void decompose() {
    using namespace std::complex_literals;

    /// Rescale the input unitary matrix, `u`, to be special unitary.
    /// Extract a phase factor, `phase`, so that `det(inv_phase * unitary) = 1`
    auto det = (matrix[0] * matrix[3]) - (matrix[1] * matrix[2]);
    phase = 0.5 * std::atan2(det.imag(), det.real());
    auto inv_phase = std::exp(-1i * phase);

    std::array<std::complex<double>, 4> special_unitary;
    std::transform(matrix.begin(), matrix.end(), special_unitary.begin(),
                   [&](auto element) { return element * inv_phase; });

    auto abs_00 = std::abs(special_unitary[0]);
    auto abs_01 = std::abs(special_unitary[1]);

    if (abs_00 >= abs_01)
      angles.beta = 2 * std::acos(abs_00);
    else
      angles.beta = 2 * std::asin(abs_01);

    auto half_sum_alpha_gamma =
        std::atan2(special_unitary[3].imag(), special_unitary[3].real());
    auto half_diff_alpha_gamma =
        std::atan2(special_unitary[2].imag(), special_unitary[2].real());

    angles.alpha = half_sum_alpha_gamma + half_diff_alpha_gamma;
    angles.gamma = half_sum_alpha_gamma - half_diff_alpha_gamma;
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

    auto parentModule = customOp->getParentOfType<ModuleOp>();

    Location loc = customOp->getLoc();
    auto targets = customOp.getTargets();
    auto controls = customOp.getControls();

    /// Get the global constant holding the concrete matrix corresponding to
    /// this custom operation invocation
    StringRef generatorName = customOp.getGenerator().getRootReference();
    auto globalOp =
        parentModule.lookupSymbol<cudaq::cc::GlobalOp>(generatorName);

    /// The decomposed sequence of quantum operations are in a function
    auto pair = generatorName.split(".rodata");
    std::string funcName = pair.first.str() + ".kernel" + pair.second.str();

    /// If the replacement function doesn't exist, create it here
    if (!parentModule.lookupSymbol<LLVM::LLVMFuncOp>(funcName)) {

      auto unitary = cudaq::readGlobalConstantArray(globalOp);

      /// TODO: Expand the logic to decompose upto 4-qubit operations
      if (unitary.size() != 4)
        return customOp.emitError(
            "Decomposition of only single qubit custom operations supported.");

      /// Controls are handled via apply specialization
      auto funcTy =
          FunctionType::get(parentModule.getContext(), targets.getTypes(), {});
      auto insPt = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointToStart(parentModule.getBody());
      auto func = rewriter.create<func::FuncOp>(parentModule->getLoc(),
                                                funcName, funcTy);
      func.setPrivate();
      rewriter.restoreInsertionPoint(insPt);

      insPt = rewriter.saveInsertionPoint();
      auto *block = func.addEntryBlock();
      rewriter.setInsertionPointToStart(block);

      /// Use Euler angle decomposition for single qubit operation
      auto zyz = BasisZYZ(unitary);
      zyz.decompose();

      /// For 1-qubit operation, apply on 'all' the targets
      auto arguments = func.getArguments();
      auto floatTy = cast<FloatType>(rewriter.getF64Type());

      /// Ignore angles less than some threshold
      double epsilon = 1e-9;
      auto isAboveThreshold = [&](auto value) {
        return std::abs(value) > epsilon;
      };

      if (isAboveThreshold(zyz.angles.alpha)) {
        auto alpha = cudaq::opt::factory::createFloatConstant(
            loc, rewriter, zyz.angles.alpha, floatTy);
        rewriter.create<quake::RzOp>(loc, alpha, ValueRange{}, arguments);
      }

      if (isAboveThreshold(zyz.angles.beta)) {
        auto beta = cudaq::opt::factory::createFloatConstant(
            loc, rewriter, zyz.angles.beta, floatTy);
        rewriter.create<quake::RyOp>(loc, beta, ValueRange{}, arguments);
      }

      if (isAboveThreshold(zyz.angles.gamma)) {
        auto gamma = cudaq::opt::factory::createFloatConstant(
            loc, rewriter, zyz.angles.gamma, floatTy);
        rewriter.create<quake::RzOp>(loc, gamma, ValueRange{}, arguments);
      }

      /// NOTE: Typically global phase can be ignored but, if this decomposition
      /// is applied in a kernel that is called with `cudaq::control`, the
      /// global phase will become a local phase and give a wrong result if we
      /// don't keep track of that.
      if (isAboveThreshold(zyz.phase)) {
        auto phase = cudaq::opt::factory::createFloatConstant(
            loc, rewriter, zyz.phase, floatTy);
        Value negPhase = rewriter.create<arith::NegFOp>(loc, phase);
        rewriter.create<quake::R1Op>(loc, phase, ValueRange{}, arguments[0]);
        rewriter.create<quake::RzOp>(loc, negPhase, ValueRange{}, arguments[0]);
      }

      rewriter.create<func::ReturnOp>(loc);
      rewriter.restoreInsertionPoint(insPt);
    }

    rewriter.create<quake::ApplyOp>(
        loc, TypeRange{}, SymbolRefAttr::get(rewriter.getContext(), funcName),
        customOp.isAdj(), controls, targets);

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
