/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPUNROLL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-unroll"

using namespace mlir;

inline std::pair<Block *, Block *> findCloneRange(Block *first, Block *last) {
  return {first->getNextNode(), last->getPrevNode()};
}

static std::optional<std::size_t>
unrollLoopByValue(cudaq::cc::LoopOp loop,
                  const cudaq::opt::LoopComponents &components) {
  auto v = components.compareValue;
  if (auto c = v.getDefiningOp<arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

static std::optional<std::size_t> unrollLoopByValue(cudaq::cc::LoopOp loop) {
  if (auto components = cudaq::opt::getLoopComponents(loop))
    return unrollLoopByValue(loop, *components);
  return std::nullopt;
}

namespace {

/// We fully unroll a counted loop (so marked with the counted attribute) as
/// long as the number of iterations is constant and that constant is less than
/// the threshold value.
///
/// Assumptions are made that the counted loop has a particular structural
/// layout as is consistent with the factory producing the counted loop.
///
/// After this pass, all loops marked counted will be unrolled or marked
/// invariant. An invariant loop means the loop must execute exactly some
/// specific number of times, even if that number is only known at runtime.
class UnrollCountedLoop : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  using Base = OpRewritePattern<cudaq::cc::LoopOp>;
  using Base::Base;

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    // At this point, we're ready to unroll the loop and replace it with a
    // sequence of blocks. Each block will receive a block argument that is the
    // iteration number. The original cc.loop will be replaced by a constant,
    // the total number of iterations.
    // TODO: Allow the threading of other block arguments to the result.
    auto components = cudaq::opt::getLoopComponents(loop);
    if (!components)
      return loop.emitOpError("loop analysis unexpectedly failed");
    auto unrollByOpt = unrollLoopByValue(loop, *components);
    if (!unrollByOpt)
      return loop.emitOpError("expected a counted loop");
    std::size_t unrollBy = *unrollByOpt;
    if (components->isClosedIntervalForm())
      ++unrollBy;
    Type inductionTy = loop.getOperands()[components->induction].getType();
    if (!isa<IntegerType, IndexType>(inductionTy))
      return loop.emitOpError("induction must be integral type");
    LLVM_DEBUG(llvm::dbgs()
               << "unrolling loop by " << unrollBy << " iterations\n");
    auto loc = loop.getLoc();
    // Split the basic block in which this cc.loop appears.
    auto *insBlock = rewriter.getInsertionBlock();
    auto insPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(insBlock, insPos);
    rewriter.setInsertionPointToEnd(insBlock);
    Value iterCount = getIntegerConstant(loc, inductionTy, 0, rewriter);
    SmallVector<Location> locsRange(loop.getNumResults(), loc);
    auto &bodyRegion = loop.getBodyRegion();
    // Make a constant number of copies of the body.
    for (std::size_t i = 0u; i < unrollBy; ++i) {
      rewriter.cloneRegionBefore(bodyRegion, endBlock);
      auto [cloneFront, cloneBack] = findCloneRange(insBlock, endBlock);
      rewriter.eraseOp(cloneBack->getTerminator());
      rewriter.setInsertionPointToEnd(cloneBack);
      // Append the next iteration number.
      Value nextIterCount =
          getIntegerConstant(loc, inductionTy, i + 1, rewriter);
      rewriter.setInsertionPointToEnd(insBlock);
      // Propagate the previous iteration number into the new block.
      rewriter.create<cf::BranchOp>(loc, cloneFront, ValueRange{iterCount});
      iterCount = nextIterCount;
      insBlock = cloneBack;
    }
    rewriter.setInsertionPointToEnd(insBlock);
    auto total = getIntegerConstant(loc, inductionTy, unrollBy, rewriter);
    rewriter.replaceOp(loop, total);
    rewriter.create<cf::BranchOp>(loc, endBlock);
    return success();
  }

  static Value getIntegerConstant(Location loc, Type ty, std::int64_t val,
                                  PatternRewriter &rewriter) {
    auto attr = rewriter.getIntegerAttr(ty, val);
    return rewriter.create<arith::ConstantOp>(loc, ty, attr);
  }
};

/// The loop unrolling pass will fully unroll a `cc::LoopOp` when the loop is
/// known to always execute a constant number of iterations. That is, the loop
/// is a counted loop. (A threshold value can be used to bound the legal range
/// of iterations. The default is 50.)
class LoopUnrollPass : public cudaq::opt::impl::LoopUnrollBase<LoopUnrollPass> {
public:
  using LoopUnrollBase::LoopUnrollBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<UnrollCountedLoop>(ctx);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<cudaq::cc::LoopOp>(
        [&](cudaq::cc::LoopOp loop) {
          return !cudaq::opt::isaCountedLoop(loop) ||
                 exceedsThresholdValue(loop);
        });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op->getLoc(), "could not unroll cc.loop\n");
      signalPassFailure();
    }
  }

  bool exceedsThresholdValue(cudaq::cc::LoopOp loop) {
    if (auto valOpt = unrollLoopByValue(loop))
      return *valOpt >= threshold;
    return true;
  }
};
} // namespace
