/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace cudaq::opt {

class ReplaceNegativeControl : public RewritePattern {
public:
  ReplaceNegativeControl(MLIRContext *context)
      : RewritePattern(MatchInterfaceOpTypeTag(),
                       quake::OperatorInterface::getInterfaceID(), 1, context) {
  }
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto quantumOp = dyn_cast<quake::OperatorInterface>(op);
    if (!quantumOp)
      return failure();

    quantumOp.dump();
    return failure();
  }
};

struct ApplyControlNegationsPass
    : public cudaq::opt::ApplyControlNegationsBase<ApplyControlNegationsPass> {
  using ApplyControlNegationsBase::ApplyControlNegationsBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceNegativeControl>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect, cudaq::cc::CCDialect,
                           arith::ArithDialect, LLVM::LLVMDialect>();

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp->emitOpError("could not expand measurements");
      signalPassFailure();
    }
  }
};
} // namespace cudaq::opt

std::unique_ptr<Pass> cudaq::opt::createApplyControlNegationsPass() {
  return std::make_unique<ApplyControlNegationsPass>();
}
