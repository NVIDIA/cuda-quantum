/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_DEADSTOREREMOVAL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "dsr"

using namespace mlir;

namespace {

class DSRPattern : public OpRewritePattern<cudaq::cc::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // If we have a cc.alloca and all of its uses are cc.store ops, then this is a
  // dead store.
  LogicalResult matchAndRewrite(cudaq::cc::StoreOp store,
                                PatternRewriter &rewriter) const override {
    auto alloca = store.getPtrvalue().getDefiningOp<cudaq::cc::AllocaOp>();
    if (!alloca)
      return failure();
    for (auto u : alloca->getUsers()) {
      if (auto s = dyn_cast<cudaq::cc::StoreOp>(u))
        if (s.getPtrvalue() == alloca.getResult())
          continue;
      return failure();
    }
    rewriter.eraseOp(store);
    return success();
  }
};

class DSRPass : public cudaq::opt::impl::DeadStoreRemovalBase<DSRPass> {
public:
  using DeadStoreRemovalBase::DeadStoreRemovalBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<DSRPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
