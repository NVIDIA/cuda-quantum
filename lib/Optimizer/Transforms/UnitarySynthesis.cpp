/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "unitary-synthesis"

namespace cudaq::opt {
#define GEN_PASS_DEF_UNITARYSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

/// TBD
class ReplaceUnitaryOp : public OpRewritePattern<quake::UnitaryOp> {
public:
  using OpRewritePattern<quake::UnitaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::UnitaryOp op,
                                PatternRewriter &rewriter) const override {
    // auto targets = op.getTargets();
    // auto controls = op.getControls();
    auto targetUnitary = op.getUnitary();

    LLVM_DEBUG(llvm::dbgs() << "\nUnitary = " << targetUnitary);

    // TODO: Expand rewriter logic

    rewriter.eraseOp(op);
    return success();
  }
};

namespace {

struct UnitarySynthesisPass
    : public cudaq::opt::impl::UnitarySynthesisBase<UnitarySynthesisPass> {
  using UnitarySynthesisBase::UnitarySynthesisBase;

  void runOnOperation() override {
    auto op = getOperation();
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<ReplaceUnitaryOp>(context);
    ConversionTarget target(*context);

    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      op.emitOpError("could not replace unitary");
      signalPassFailure();
    }
  }
};
} // namespace