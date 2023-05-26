/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "cc-loop-unroll"

using namespace mlir;

inline std::pair<Block *, Block *> findCloneRange(Block *first, Block *last) {
  return {first->getNextNode(), last->getPrevNode()};
}

namespace {
// We fully unroll a counted loop (so marked with the counted attribute) as long
// as the number of iterations is constant and that constant is less than the
// threshold value.
//
// Assumptions are made that the counted loop has a particular structural
// layout as is consistent with the factory producing the counted loop.
//
// After this pass, all loops marked "counted" will be unrolled or marked
// "invariant". An invariant loop means the loop must execute exactly some
// specific number of times, even if that number is only known at runtime.
class UnrollCountedLoop : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  explicit UnrollCountedLoop(MLIRContext *ctx, std::size_t max)
      : OpRewritePattern(ctx), threshold(max) {}

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    assert(loop->hasAttr("counted"));
    Value totalIterations = getTotalIterationsConstant(loop);
    auto markInvariant = [&]() {
      rewriter.updateRootInPlace(loop.getOperation(), [&]() {
        loop->removeAttr("counted");
        loop->setAttr("invariant", rewriter.getUnitAttr());
      });
    };
    if (!totalIterations || (threshold == 0)) {
      LLVM_DEBUG(llvm::dbgs() << "non-constant iterations\n");
      // Change the attribute on the loop from counted to invariant if the
      // number of iterations on the counted loop is not constant, exceeds the
      // threshold, etc.
      markInvariant();
      return success();
    }
    auto constOp = cast<arith::ConstantOp>(totalIterations.getDefiningOp());
    auto intAttr = constOp.getValue().cast<IntegerAttr>();
    std::size_t iterations = intAttr.getInt();
    if (iterations > threshold) {
      LLVM_DEBUG(llvm::dbgs() << "iterations exceed threshold value\n");
      markInvariant();
      return success();
    }

    // At this point, we're ready to unroll the loop and replace it with a
    // sequence of blocks. Each block will receive a block argument that is the
    // iteration number. The original cc.loop will be replaced by a constant,
    // the total number of iterations.
    const auto unrollBy = iterations;
    LLVM_DEBUG(llvm::dbgs()
               << "unrolling loop by " << unrollBy << " iterations\n");
    auto loc = loop.getLoc();
    // Split the basic block in which this cc.loop appears.
    auto *insBlock = rewriter.getInsertionBlock();
    auto insPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(insBlock, insPos);
    rewriter.setInsertionPointToEnd(insBlock);
    Value iterCount = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Location> locsRange(loop.getNumResults(), loc);
    auto &bodyRegion = loop.getBodyRegion();
    // Make a constant number of copies of the body.
    for (std::size_t i = 0u; i < unrollBy; ++i) {
      rewriter.cloneRegionBefore(bodyRegion, endBlock);
      auto [cloneFront, cloneBack] = findCloneRange(insBlock, endBlock);
      rewriter.eraseOp(cloneBack->getTerminator());
      rewriter.setInsertionPointToEnd(cloneBack);
      // Append the next iteration number.
      Value nextIterCount = rewriter.create<arith::ConstantIndexOp>(loc, i + 1);
      rewriter.setInsertionPointToEnd(insBlock);
      // Propagate the previous iteration number into the new block.
      rewriter.create<cf::BranchOp>(loc, cloneFront, ValueRange{iterCount});
      iterCount = nextIterCount;
      insBlock = cloneBack;
    }
    rewriter.setInsertionPointToEnd(insBlock);
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(loop, unrollBy);
    rewriter.create<cf::BranchOp>(loc, endBlock);
    return success();
  }

  // Return the Value that is the total number of iterations to execute the
  // loop.
  Value getTotalIterationsConstant(cudaq::cc::LoopOp loop) const {
    auto &block = loop.getWhileRegion().front();
    // CmpIOp is second from last instruction.
    for (auto &suspectInst : block) {
      if (auto compare = dyn_cast<arith::CmpIOp>(suspectInst)) {
        // assert(make sure compare is less than);
        Value totalIterations = compare.getOperand(1);
        if (isa<arith::ConstantOp>(totalIterations.getDefiningOp()))
          return totalIterations;
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "block is:\n");
    LLVM_DEBUG(block.dump());
    return {};
  }

  std::size_t threshold;
};

class LoopUnrollPass : public cudaq::opt::LoopUnrollBase<LoopUnrollPass> {
public:
  LoopUnrollPass() = default;
  LoopUnrollPass(std::size_t max) { threshold = max; }

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<UnrollCountedLoop>(ctx, threshold);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<cudaq::cc::LoopOp>(
        [](cudaq::cc::LoopOp loop) { return !loop->hasAttr("counted"); });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op->getLoc(), "error unrolling cc.loop\n");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createLoopUnrollPass() {
  return std::make_unique<LoopUnrollPass>();
}

std::unique_ptr<Pass>
cudaq::opt::createLoopUnrollPass(std::size_t maxIterations) {
  return std::make_unique<LoopUnrollPass>(maxIterations);
}
