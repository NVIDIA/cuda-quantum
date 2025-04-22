/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// These patterns are used by the lower-to-cfg pass and cc-loop-unroll pass.

// This file must be included after a `using namespace mlir;` as it uses bare
// identifiers from that namespace.

namespace {
class RewriteIf : public OpRewritePattern<cudaq::cc::IfOp> {
public:
  explicit RewriteIf(MLIRContext *ctx)
      : OpRewritePattern(ctx), rewriteOnlyIfConst(false) {}

  explicit RewriteIf(MLIRContext *ctx, bool rewriteOnlyIfConst)
      : OpRewritePattern(ctx), rewriteOnlyIfConst(rewriteOnlyIfConst) {}

  /// Rewrites an if construct like
  /// ```mlir
  /// (0)
  /// quake.if %cond {
  ///   (1)
  /// } else {
  ///   (2)
  /// }
  /// (3)
  /// ```
  /// to a CFG like
  /// ```mlir
  ///   (0)
  ///   cf.cond_br %cond, ^bb1, ^bb2
  /// ^bb1:
  ///   (1)
  ///   cf.br ^bb3
  /// ^bb2:
  ///   (2)
  ///   cf.br ^bb3
  /// ^bb3:
  ///   (3)
  /// ```
  LogicalResult matchAndRewrite(cudaq::cc::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    // Bail out on non-constant conditions if we just need to rewrite if the
    // condition value is constant.
    if (rewriteOnlyIfConst) {
      auto cond = ifOp.getCondition().getDefiningOp<arith::ConstantIntOp>();
      if (!cond)
        return failure();
    }

    auto loc = ifOp.getLoc();
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPos);
    if (ifOp.getNumResults() != 0) {
      Block *continueBlock = rewriter.createBlock(
          endBlock, ifOp.getResultTypes(),
          SmallVector<Location>(ifOp.getNumResults(), loc));
      rewriter.create<cf::BranchOp>(loc, endBlock);
      endBlock = continueBlock;
    }
    auto *thenBlock = &ifOp.getThenRegion().front();
    bool hasElse = !ifOp.getElseRegion().empty();
    auto *elseBlock = hasElse ? &ifOp.getElseRegion().front() : endBlock;
    updateBodyBranches(&ifOp.getThenRegion(), rewriter, endBlock);
    updateBodyBranches(&ifOp.getElseRegion(), rewriter, endBlock);
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), endBlock);
    if (hasElse)
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), endBlock);
    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<cf::CondBranchOp>(loc, ifOp.getCondition(), thenBlock,
                                      ifOp.getLinearArgs(), elseBlock,
                                      ifOp.getLinearArgs());
    rewriter.replaceOp(ifOp, endBlock->getArguments());
    return success();
  }

  // Replace all the ContinueOp in the body region with branches to the correct
  // basic blocks.
  void updateBodyBranches(Region *bodyRegion, PatternRewriter &rewriter,
                          Block *continueBlock) const {
    // Walk body region and replace all continue and break ops.
    for (Block &block : *bodyRegion) {
      auto *terminator = block.getTerminator();
      if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(terminator)) {
        rewriter.setInsertionPointToEnd(&block);
        LLVM_DEBUG(llvm::dbgs() << "replacing " << *terminator << '\n');
        rewriter.replaceOpWithNewOp<cf::BranchOp>(cont, continueBlock,
                                                  cont.getOperands());
      }
      // Other ad-hoc control flow in the region need not be rewritten.
    }
  }

private:
  bool rewriteOnlyIfConst = false;
};
} // namespace
