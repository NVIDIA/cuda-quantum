/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPPEELING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-peeling"

using namespace mlir;

namespace {
class LoopPat : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    if (!loop.isPostConditional())
      return failure();
    // Peel do-while loop and convert it to a while loop.
    LLVM_DEBUG(llvm::dbgs() << "will peel do-while loop:\n"; loop.dump());

    // Split the block after loop, old-loop-block and after-block.
    Operation *nextOp = loop->getNextNode();
    auto *oldLoopBlock = rewriter.getInsertionBlock();
    auto loopPos = rewriter.getInsertionPoint();
    auto *afterBlock =
        rewriter.splitBlock(oldLoopBlock, Block::iterator{nextOp});

    // Add terminator to old-loop-block: thread results of loop to after-block's
    // arguments and replace the loop results uses.
    for (auto res : loop.getResults())
      afterBlock->addArgument(res.getType(), loop.getLoc());
    rewriter.setInsertionPointToEnd(oldLoopBlock);
    auto finalBranch = rewriter.create<cf::BranchOp>(loop.getLoc(), afterBlock,
                                                     loop.getResults());
    // NB: the results of the original loop are now split between the peeled
    // copy of body and the modified new loop. Introduce explicit block
    // arguments for the phi node functionality.
    for (auto iter : llvm::enumerate(loop.getResults())) {
      Value v = iter.value();
      v.replaceUsesWithIf(
          finalBranch.getDestOperands()[iter.index()],
          [branch = finalBranch.getOperation()](OpOperand &opnd) {
            auto *op = opnd.getOwner();
            return op != branch;
          });
    }

    // Split the block before loop, before-block and new-loop-block. Add branch
    // arguments to new-loop-block corresponding to loop's region arguments.
    auto *beforeBlock = oldLoopBlock;
    auto *newLoopBlock = rewriter.splitBlock(oldLoopBlock, loopPos);
    for (auto res : loop.getResults())
      newLoopBlock->addArgument(res.getType(), loop.getLoc());
    SmallVector<Value> loopArgs = loop.getOperands();
    loop.getInitialArgsMutable().assign(newLoopBlock->getArguments());

    // Clone the body region. Add branch from before-loop to the entry block of
    // the cloned CFG.
    rewriter.cloneRegionBefore(loop.getBodyRegion(), newLoopBlock);
    Block *firstBlock = beforeBlock->getNextNode();
    rewriter.setInsertionPointToEnd(beforeBlock);
    rewriter.create<cf::BranchOp>(loop.getLoc(), firstBlock, loopArgs);

    // Replace continue ops with branches to the new-loop-block. Replace break
    // ops with branches to the after-block.
    auto rewriteBranch = [&](auto op, Block *dest) {
      rewriter.setInsertionPointToEnd(op->getBlock());
      rewriter.create<cf::BranchOp>(op.getLoc(), dest, op.getOperands());
      rewriter.eraseOp(op);
    };
    for (Block *b = firstBlock; b != newLoopBlock; b = b->getNextNode())
      for (auto &op : *b) {
        if (auto contOp = dyn_cast<cudaq::cc::ContinueOp>(op)) {
          rewriteBranch(contOp, newLoopBlock);
          break;
        } else if (auto brkOp = dyn_cast<cudaq::cc::BreakOp>(op)) {
          rewriteBranch(brkOp, afterBlock);
          break;
        }
      }

    // Turn the do-while into a while loop.
    loop.setPostCondition(/*do-while=*/false);
    LLVM_DEBUG({
      llvm::dbgs() << "peeled loop:\n";
      for (Block *b = firstBlock; b != afterBlock; b = b->getNextNode())
        b->dump();
    });
    return success();
  }
};

class LoopPeelingPass
    : public cudaq::opt::impl::LoopPeelingBase<LoopPeelingPass> {
public:
  using LoopPeelingBase::LoopPeelingBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<LoopPat>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      op->emitOpError("could not peel loop");
      signalPassFailure();
    }
  }
};
} // namespace
