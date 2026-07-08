/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_PYFRONTENDCLEANUP
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "py-frontend-cleanup"

using namespace mlir;

// Cleanup rewrites for IR constructs produced only by the Python AST bridge.
// Runs in the Python AOT pipeline and must be followed by `canonicalize`, which
// completes the folds these rewrites expose. See the `PyFrontendCleanup`
// description in Passes.td for details.

namespace {
// %0 = cc.if(%cond) -> !quake.veq<?> {
//        %1 = quake.subveq ...
//        cc.continue %1 : !quake.veq<?>
//      } else {
//        %2 = cc.undef !quake.veq<?>
//        cc.continue %2 : !quake.veq<?>
//      }
// %3 = quake.veq_size %0 : (!quake.veq<?>) -> i64
// ───────────────────────────────────────────────────────────────────────
// %0:2 = cc.if(%cond) -> (!quake.veq<?>, i64) {
//        %1 = quake.subveq ...
//        %4 = quake.veq_size %1 : (!quake.veq<?>) -> i64
//        cc.continue %1, %4 : !quake.veq<?>, i64
//      } else {
//        %2 = cc.undef !quake.veq<?>
//        %5 = quake.veq_size %2 : (!quake.veq<?>) -> i64
//        cc.continue %2, %5 : !quake.veq<?>, i64
//      }
// The hoist only exposes the undef/poison (or constant-size) branch value; the
// subsequent `canonicalize` does the actual fold via ForwardEmptyVeqSizePattern
// / ForwardConstantVeqSizePattern.
struct HoistVeqSizeThroughIfPattern
    : public OpRewritePattern<cudaq::quake::VeqSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::VeqSizeOp veqSize,
                                PatternRewriter &rewriter) const override {
    auto veq = veqSize.getVeq();
    auto ifOp = veq.getDefiningOp<cudaq::cc::IfOp>();
    // Only handle the case the veq is yielded from both a if/else.
    if (!ifOp || !ifOp.hasThen() || !ifOp.hasElse())
      return failure();

    auto resultNumber = cast<OpResult>(veq).getResultNumber();
    if (!regionYieldsFoldableEmptyVeq(ifOp.getThenRegion(), resultNumber) &&
        !regionYieldsFoldableEmptyVeq(ifOp.getElseRegion(), resultNumber))
      return failure();

    const unsigned origNumResults = ifOp.getNumResults();
    SmallVector<Type> resultTypes(ifOp.getResultTypes());
    resultTypes.push_back(veqSize.getType());

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(ifOp);
    auto newIf = cudaq::cc::IfOp::create(
        rewriter, ifOp.getLoc(), resultTypes, ifOp.getCondition(),
        ifOp.getLinearArgs(),
        [&](OpBuilder &, Location, Region &region) {
          rewriter.inlineRegionBefore(ifOp.getThenRegion(), region,
                                      region.end());
        },
        [&](OpBuilder &, Location, Region &region) {
          rewriter.inlineRegionBefore(ifOp.getElseRegion(), region,
                                      region.end());
        });

    appendVeqSizeToContinues(newIf.getThenRegion(), resultNumber,
                             veqSize.getType(), rewriter);
    appendVeqSizeToContinues(newIf.getElseRegion(), resultNumber,
                             veqSize.getType(), rewriter);

    SmallVector<Value> replacements;
    replacements.reserve(origNumResults);
    for (unsigned i = 0; i < origNumResults; ++i)
      replacements.push_back(newIf.getResult(i));

    rewriter.replaceOp(ifOp, replacements);
    rewriter.replaceOp(veqSize, newIf.getResult(origNumResults));
    return success();
  }

private:
  static bool regionYieldsFoldableEmptyVeq(Region &region,
                                           unsigned resultNumber) {
    auto isFoldableEmptyVeq = [](Value yielded) {
      return yielded.getDefiningOp<cudaq::cc::UndefOp>() ||
             yielded.getDefiningOp<cudaq::cc::PoisonOp>();
    };

    for (auto &block : region) {
      auto cont = dyn_cast<cudaq::cc::ContinueOp>(block.getTerminator());
      if (!cont)
        continue;
      if (isFoldableEmptyVeq(cont.getOperand(resultNumber)))
        return true;
    }
    return false;
  }

  static void appendVeqSizeToContinues(Region &region, unsigned resultNumber,
                                       Type sizeType,
                                       PatternRewriter &rewriter) {
    for (auto &block : region) {
      auto cont = dyn_cast<cudaq::cc::ContinueOp>(block.getTerminator());
      if (!cont)
        continue;
      Value yielded = cont.getOperand(resultNumber);
      rewriter.setInsertionPoint(cont);
      auto size = cudaq::quake::VeqSizeOp::create(rewriter, cont.getLoc(),
                                                  sizeType, yielded);
      auto operands = llvm::to_vector(cont.getOperands());
      operands.push_back(size);
      rewriter.replaceOpWithNewOp<cudaq::cc::ContinueOp>(cont, operands);
    }
  }
};

class PyFrontendCleanupPass
    : public cudaq::opt::impl::PyFrontendCleanupBase<PyFrontendCleanupPass> {
public:
  using PyFrontendCleanupBase::PyFrontendCleanupBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<HoistVeqSizeThroughIfPattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
