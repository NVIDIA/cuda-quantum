/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

#define DEBUG_TYPE "lower-to-cfg"

using namespace mlir;

namespace {
class RewriteScope : public OpRewritePattern<cudaq::cc::ScopeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// Rewrites a scope construct like
  /// ```mlir
  /// (0)
  /// quake.scope {
  ///   (1)
  /// }
  /// (2)
  /// ```
  /// to a CFG like
  /// ```mlir
  ///   (0)
  ///   cf.br ^bb1
  /// ^bb1:
  ///   (1)
  ///   cf.br ^bb2
  /// ^bb2:
  ///   (2)
  /// ```
  LogicalResult matchAndRewrite(cudaq::cc::ScopeOp scopeOp,
                                PatternRewriter &rewriter) const override {
    auto loc = scopeOp.getLoc();
    auto *initBlock = rewriter.getInsertionBlock();
    Value stacksave;
    auto module = scopeOp.getOperation()->getParentOfType<ModuleOp>();
    auto ptrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    if (scopeOp.hasAllocation(/*quantumAllocs=*/false)) {
      auto fun = cudaq::opt::factory::createFunction(
          "llvm.stacksave", ArrayRef<Type>{ptrTy}, {}, module);
      fun.setPrivate();
      auto call = rewriter.create<func::CallOp>(
          loc, ptrTy, fun.getSymNameAttr(), ArrayRef<Value>{});
      stacksave = call.getResult(0);
    }
    auto initPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPos);
    ValueRange scopeResults;
    if (scopeOp.getNumResults() != 0) {
      Block *continueBlock = rewriter.createBlock(
          endBlock, scopeOp.getResultTypes(),
          SmallVector<Location>(scopeOp.getNumResults(), loc));
      scopeResults = continueBlock->getArguments();
      rewriter.create<cf::BranchOp>(loc, endBlock);
      endBlock = continueBlock;
    }

    for (auto &block : scopeOp.getInitRegion())
      if (auto contOp =
              dyn_cast<cudaq::cc::ContinueOp>(block.getTerminator())) {
        rewriter.setInsertionPointToEnd(&block);
        rewriter.replaceOpWithNewOp<cf::BranchOp>(contOp, endBlock,
                                                  contOp.getOperands());
      }

    auto *entryBlock = &scopeOp.getInitRegion().front();
    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<cf::BranchOp>(loc, entryBlock, ValueRange{});
    rewriter.inlineRegionBefore(scopeOp.getInitRegion(), endBlock);
    if (stacksave) {
      rewriter.setInsertionPointToStart(endBlock);
      auto fun = cudaq::opt::factory::createFunction(
          "llvm.stackrestore", {}, ArrayRef<Type>{ptrTy}, module);
      fun.setPrivate();
      rewriter.create<func::CallOp>(loc, ArrayRef<Type>{}, fun.getSymNameAttr(),
                                    ArrayRef<Value>{stacksave});
    }
    rewriter.replaceOp(scopeOp, scopeResults);
    return success();
  }
};

class RewriteLoop : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// There are two cases. A loop is either pre-conditional or post-conditional,
  /// meaning the iteration test is done before or after, resp., the body of the
  /// loop is executed.
  ///
  /// Pre-conditional case:
  /// ```mlir
  /// (0)
  /// quake.loop while { (1) } do { (2) } [ step { (3) } ]
  /// (4)
  /// ```
  /// becomes
  /// ```mlir
  ///   (0)
  ///   br ^bb0
  /// ^bb0:
  ///   (1)
  ///   cond_br %cond, ^bb1, ^bb3
  /// ^bb1:
  ///   (2) ; break -> ^bb3
  ///   br ^bb0 [or ^bb2]
  /// [ ^bb2:
  ///     (3)
  ///     br ^bb0 ]
  /// ^bb3:
  ///   (4)
  /// ```
  ///
  /// Post-conditional case:
  /// ```mlir
  /// (0)
  /// quake.loop do { (1) } while { (2) }
  /// (3)
  /// ```
  /// becomes
  /// ```mlir
  ///   (0)
  ///   br ^bb0
  /// ^bb0:
  ///   (1)
  ///   br ^bb1
  /// ^bb1:
  ///   (2)
  ///   cond_br %cond, ^bb0, ^bb3
  /// ^bb3:
  ///   (3)
  /// ```
  ///
  /// Pythonic case:
  /// ```mlir
  /// (0)
  /// quake.loop while { (1) } do { (2) } step { (3) } else { (4) }
  /// (5)
  /// ```
  /// becomes
  /// ```mlir
  ///   (0)
  ///   br ^bb0
  /// ^bb0:
  ///   (1)
  ///   cond_br %cond, ^bb1, ^bb3
  /// ^bb1:
  ///   (2)  ; break -> ^bb4
  ///   br ^bb2
  /// ^bb2:
  ///   (3)
  ///   br ^bb0
  /// ^bb3:
  ///   (4)
  ///   br ^bb4
  /// ^bb4:
  ///   (5)
  /// ```
  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    auto loc = loopOp.getLoc();

    // Split the basic block in which this CLoop appears.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPos);

    SmallVector<Value> loopOperands;
    loopOperands.append(loopOp.getOperands().begin(),
                        loopOp.getOperands().end());

    auto *whileBlock = loopOp.getWhileBlock();
    auto whileCond = cast<cudaq::cc::ConditionOp>(whileBlock->getTerminator());
    if (loopOp.getNumResults() != 0) {
      Block *continueBlock = rewriter.createBlock(
          endBlock, loopOp.getResultTypes(),
          SmallVector<Location>(loopOp.getNumResults(), loc));
      rewriter.create<cf::BranchOp>(loc, endBlock);
      endBlock = continueBlock;
    }
    auto comparison = whileCond.getCondition();
    auto *bodyBlock = loopOp.getDoEntryBlock();
    auto *condBlock = loopOp.hasStep() ? loopOp.getStepBlock() : whileBlock;

    if (failed(updateBodyBranches(&loopOp.getBodyRegion(), rewriter, condBlock,
                                  endBlock)))
      return failure();
    if (loopOp.isPostConditional()) {
      // Branch from `initBlock` to getBodyRegion().front().
      rewriter.setInsertionPointToEnd(initBlock);
      rewriter.create<cf::BranchOp>(loc, bodyBlock, loopOperands);
      // Move the body region blocks between initBlock and end block.
      rewriter.inlineRegionBefore(loopOp.getBodyRegion(), endBlock);
      // Replace the condition op with a `cf.cond_br`.
      rewriter.setInsertionPointToEnd(whileBlock);
      rewriter.create<cf::CondBranchOp>(loc, comparison, bodyBlock,
                                        whileCond.getResults(), endBlock,
                                        whileCond.getResults());
      rewriter.eraseOp(whileCond);
      // Move the while region between the body and end block.
      rewriter.inlineRegionBefore(loopOp.getWhileRegion(), endBlock);
    } else {
      auto *elseBlock =
          loopOp.hasPythonElse() ? loopOp.getElseEntryBlock() : endBlock;
      // Branch from `initBlock` to whileRegion().front().
      rewriter.setInsertionPointToEnd(initBlock);
      rewriter.create<cf::BranchOp>(loc, whileBlock, loopOperands);
      // Replace the condition op with a `cf.cond_br` op.
      rewriter.setInsertionPointToEnd(whileBlock);
      rewriter.create<cf::CondBranchOp>(loc, comparison, bodyBlock,
                                        whileCond.getResults(), elseBlock,
                                        whileCond.getResults());
      rewriter.eraseOp(whileCond);
      // Move the while and body region blocks between initBlock and endBlock.
      rewriter.inlineRegionBefore(loopOp.getWhileRegion(), endBlock);
      rewriter.inlineRegionBefore(loopOp.getBodyRegion(), endBlock);
      // If there is a step region, replace the continue op with a branch and
      // move the region between the body region and end block.
      if (loopOp.hasStep()) {
        auto *stepBlock = loopOp.getStepBlock();
        auto *terminator = stepBlock->getTerminator();
        rewriter.setInsertionPointToEnd(stepBlock);
        rewriter.create<cf::BranchOp>(loc, whileBlock,
                                      terminator->getOperands());
        rewriter.eraseOp(terminator);
        rewriter.inlineRegionBefore(loopOp.getStepRegion(), endBlock);
      }
      if (loopOp.hasPythonElse()) {
        if (failed(updateBodyBranches(&loopOp.getElseRegion(), rewriter,
                                      endBlock, static_cast<Block *>(nullptr))))
          return failure();
        rewriter.inlineRegionBefore(loopOp.getElseRegion(), endBlock);
      }
    }
    rewriter.replaceOp(loopOp, endBlock->getArguments());
    return success();
  }

  /// Replace all the ContinueOp and BreakOp in the body region with branches to
  /// the correct basic blocks. If there is a BreakOp and no break block target,
  /// return failure.
  LogicalResult updateBodyBranches(Region *bodyRegion,
                                   PatternRewriter &rewriter,
                                   Block *continueBlock,
                                   Block *breakBlock) const {
    assert(continueBlock && "continue block target must exist");
    // Walk body region and replace all continue and break ops.
    for (Block &block : *bodyRegion) {
      auto *terminator = block.getTerminator();
      rewriter.setInsertionPointToEnd(&block);
      if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(terminator)) {
        rewriter.replaceOpWithNewOp<cf::BranchOp>(cont, continueBlock,
                                                  cont.getOperands());
      } else if (auto brk = dyn_cast<cudaq::cc::BreakOp>(terminator)) {
        if (!breakBlock)
          return failure();
        rewriter.replaceOpWithNewOp<cf::BranchOp>(brk, breakBlock,
                                                  brk.getOperands());
      }
      // Other ad-hoc control flow within the register need not be rewritten.
    }
    return success();
  }
};

class RewriteIf : public OpRewritePattern<cudaq::cc::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

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
                                      ValueRange{}, elseBlock, ValueRange{});
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
};

class RewriteReturn : public OpRewritePattern<cudaq::cc::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ReturnOp retOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(retOp, retOp.getOperands());
    return success();
  }
};

class LowerToCFGPass : public cudaq::opt::LowerToCFGBase<LowerToCFGPass> {
public:
  LowerToCFGPass() = default;

  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RewriteLoop, RewriteScope, RewriteIf, RewriteReturn>(ctx);
    ConversionTarget target(*ctx);
    target.addIllegalOp<cudaq::cc::ScopeOp, cudaq::cc::LoopOp, cudaq::cc::IfOp,
                        cudaq::cc::ConditionOp, cudaq::cc::ContinueOp,
                        cudaq::cc::BreakOp, cudaq::cc::ReturnOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op.getLoc(), "error converting to CFG\n");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createLowerToCFGPass() {
  return std::make_unique<LowerToCFGPass>();
}
