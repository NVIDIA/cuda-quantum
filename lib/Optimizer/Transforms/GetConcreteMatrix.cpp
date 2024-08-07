/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_GETCONCRETEMATRIX
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "get-concrete-matrix"

using namespace mlir;

namespace {

class CustomUnitaryPattern
    : public OpRewritePattern<quake::CustomUnitarySymbolOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::CustomUnitarySymbolOp customOp,
                                PatternRewriter &rewriter) const override {

    // Check if the generator associated with custom operation is a function. If
    // not, it may already have been replaced.
    auto generator = customOp.getGenerator();

    auto parentModule = customOp->getParentOfType<ModuleOp>();
    auto funcOp = parentModule.lookupSymbol<func::FuncOp>(generator);
    if (!funcOp)
      return failure();

    // The generator function returns a concrete matrix. If prior passes have
    // run to constant fold and lift array values, the generator function will
    // have address of the global variable which holds the concrete matrix.
    StringRef concreteMatrix;

    funcOp.walk([&](cudaq::cc::AddressOfOp addrOp) {
      concreteMatrix = addrOp.getGlobalName();
    });

    if (concreteMatrix.empty()) {
      return customOp.emitError(
          "Constant matrix corresponding to custom operation's generator "
          "function not found in the module.");
    }
    // Modify the custom operation to use the global variable instead of the
    // generator function.
    auto ccGlobalOp =
        parentModule.lookupSymbol<cudaq::cc::GlobalOp>(concreteMatrix);

    if (ccGlobalOp) {

      rewriter.replaceOpWithNewOp<quake::CustomUnitarySymbolOp>(
          customOp,
          FlatSymbolRefAttr::get(parentModule.getContext(), concreteMatrix),
          customOp.getIsAdj(), customOp.getParameters(), customOp.getControls(),
          customOp.getTargets(), customOp.getNegatedQubitControlsAttr());
      return success();
    }
    return failure();
  }
};

class GetConcreteMatrixPass
    : public cudaq::opt::impl::GetConcreteMatrixBase<GetConcreteMatrixPass> {
public:
  using GetConcreteMatrixBase::GetConcreteMatrixBase;

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
