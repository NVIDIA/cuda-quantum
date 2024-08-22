/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
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

class CustomUnitaryPattern
    : public OpRewritePattern<quake::CustomUnitarySymbolOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::CustomUnitarySymbolOp customOp,
                                PatternRewriter &rewriter) const override {

    auto parentModule = customOp->getParentOfType<ModuleOp>();
    // Fetch the unitary matrix generator for this custom operation
    auto sref = customOp.getGenerator();
    StringRef generatorName = sref.getRootReference();
    auto globalOp =
        parentModule.lookupSymbol<cudaq::cc::GlobalOp>(generatorName);
    auto unitary = readGlobalConstantArray(globalOp);

    if (unitary.size() != 4)
      return customOp.emitError(
          "Decomposition of only single qubit custom operations supported.");

    // auto targets = customOp.getTargets();
    // auto controls = customOp.getControls();
    // Location loc = customOp->getLoc();
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
