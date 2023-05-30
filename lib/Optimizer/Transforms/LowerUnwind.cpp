/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "unwind-lowering"

using namespace mlir;

namespace {

// A cc.scope may have break, continue, and return landing pads. A func.func may
// have a return landing pad. This struture tracks the landing pads associated
// with these ops. Unwind ops may require lowering to primitive cf.br ops or
// retain some structure and conversion to cc.break or cc.continue. That
// conversion is also tracked.
struct ContainsUnwindGotoOf {
  Operation *parent = nullptr;    // the original parent op
  Block *continueBlock = nullptr; // unwind_continue landing pad
  Block *breakBlock = nullptr;    // unwind_break landing pad
  Block *returnBlock = nullptr;   // unwind_return landing pad
  bool asPrimitive = false;       // convert to cf.br?
};

using UnwindOpAnalysisInfo = llvm::DenseMap<Operation *, ContainsUnwindGotoOf>;

struct UnwindOpAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnwindOpAnalysis)

  UnwindOpAnalysis(func::FuncOp op) : func(op) {
    performAnalysis(op.getOperation());
  }

  UnwindOpAnalysisInfo getAnalysisInfo() const { return infoMap; }

private:
  template <typename A>
  void addToParents(A unwindOp) {
    bool asPrimitive = false;
    if constexpr (std::is_same_v<A, cudaq::cc::UnwindReturnOp>) {
      asPrimitive = true;
    } else {
      for (Operation *parent = unwindOp->getParentOp();
           !isa<cudaq::cc::LoopOp>(parent) && !asPrimitive;
           parent = parent->getParentOp())
        asPrimitive = isa<func::FuncOp, cudaq::cc::ScopeOp>(parent);
    }
    Operation *parent = nullptr;
    for (Operation *op = unwindOp.getOperation(); op; op = parent) {
      ContainsUnwindGotoOf unwindGoto;
      parent = op->getParentOp();
      auto iter = infoMap.find(op);
      auto &ref = iter == infoMap.end() ? unwindGoto : iter->second;
      ref.asPrimitive = asPrimitive;
      bool requiresBlock =
          !isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
               cudaq::cc::UnwindReturnOp>(op);
      if constexpr (std::is_same_v<A, cudaq::cc::UnwindBreakOp>) {
        if (requiresBlock && !ref.breakBlock)
          ref.breakBlock = new Block;
        if (isa<cudaq::cc::LoopOp>(op))
          parent = nullptr;
      } else if constexpr (std::is_same_v<A, cudaq::cc::UnwindContinueOp>) {
        if (requiresBlock && !ref.continueBlock)
          ref.continueBlock = new Block;
        if (isa<cudaq::cc::LoopOp>(op))
          parent = nullptr;
      } else {
        if (requiresBlock && !ref.returnBlock)
          ref.returnBlock = new Block;
        if (isa<func::FuncOp>(op)) {
          auto *ctx = op->getContext();
          op->setAttr("add_dealloc", UnitAttr::get(ctx));
          parent = nullptr;
        }
      }
      if (!ref.parent)
        ref.parent = parent;
      if (iter == infoMap.end()) {
        LLVM_DEBUG(llvm::dbgs() << "analysis adding: " << op << '\n');
        infoMap.insert(std::make_pair(op, unwindGoto));
      }
    }
  }

  void performAnalysis(Operation *func) {
    func->walk([this](Operation *o) {
      if (auto brkOp = dyn_cast<cudaq::cc::UnwindBreakOp>(o))
        addToParents(brkOp);
      else if (auto cntOp = dyn_cast<cudaq::cc::UnwindContinueOp>(o))
        addToParents(cntOp);
      else if (auto retOp = dyn_cast<cudaq::cc::UnwindReturnOp>(o))
        addToParents(retOp);
    });
  }

  func::FuncOp func;
  UnwindOpAnalysisInfo infoMap;
};
} // namespace

static Operation *originalParent(const UnwindOpAnalysisInfo &map,
                                 Operation *arg) {
  auto iter = map.find(arg);
  assert(iter != map.end());
  return iter->second.parent;
}

static const ContainsUnwindGotoOf &
getLandingPad(const UnwindOpAnalysisInfo &infoMap, Operation *arg) {
  auto *op = originalParent(infoMap, arg);
  auto iter = infoMap.find(op);
  if (iter != infoMap.end() &&
      (iter->second.breakBlock || iter->second.continueBlock ||
       iter->second.returnBlock)) {
    LLVM_DEBUG(llvm::dbgs()
               << "arg " << arg << " has {" << iter->second.parent << " ["
               << iter->second.continueBlock << ' ' << iter->second.breakBlock
               << ' ' << iter->second.returnBlock << "] "
               << iter->second.asPrimitive << "}\n");
    return iter->second;
  }
  cudaq::emitFatalError(arg->getLoc(), "landing pad not found");
}

static Block *getLandingPad(cudaq::cc::UnwindBreakOp op,
                            const UnwindOpAnalysisInfo &infoMap) {
  return getLandingPad(infoMap, op).breakBlock;
}

static Block *getLandingPad(cudaq::cc::UnwindContinueOp op,
                            const UnwindOpAnalysisInfo &infoMap) {
  return getLandingPad(infoMap, op).continueBlock;
}

static Block *getLandingPad(cudaq::cc::UnwindReturnOp op,
                            const UnwindOpAnalysisInfo &infoMap) {
  return getLandingPad(infoMap, op).returnBlock;
}

namespace {
/// A scope op that contains an unwind op and is contained by a loop (for break
/// or continue) or for return always, dictates that the unwind op must transfer
/// control to a landing pad for the continue, break, or return semantics of
/// that scope. The exact lowering of this control transfer is determined in the
/// analysis.
struct ScopeOpPattern : public OpRewritePattern<cudaq::cc::ScopeOp> {
  explicit ScopeOpPattern(MLIRContext *ctx, const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(cudaq::cc::ScopeOp scope,
                                PatternRewriter &rewriter) const override {
    auto iter = infoMap.find(scope.getOperation());
    assert(iter != infoMap.end() && iter->second.asPrimitive);
    LLVM_DEBUG(llvm::dbgs() << "replacing scope @" << scope.getLoc() << '\n');
    auto loc = scope.getLoc();
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *nextBlock = rewriter.splitBlock(initBlock, initPos);
    auto *scopeBlock = &scope.getInitRegion().front();
    auto *scopeEndBlock = &scope.getInitRegion().back();
    auto *contOp = scopeEndBlock->getTerminator();
    // Scan the scope for quake allocations.
    SmallVector<quake::AllocaOp> qallocas;
    for (Block &b : scope.getInitRegion())
      for (Operation &o : b)
        if (auto q = dyn_cast<quake::AllocaOp>(o))
          qallocas.push_back(q);
    // Setup a block with arguments that can be forwarded.
    if (scope.getNumResults() != 0) {
      SmallVector<Location> locs(scope.getNumResults(), loc);
      Block *continueBlock =
          rewriter.createBlock(nextBlock, scope.getResultTypes(), locs);
      rewriter.create<cf::BranchOp>(loc, nextBlock);
      nextBlock = continueBlock;
    }
    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<cf::BranchOp>(loc, scopeBlock, ValueRange{});
    // Normal scope exit with deallocations.
    rewriter.setInsertionPoint(contOp);
    for (auto a : llvm::reverse(qallocas))
      rewriter.create<quake::DeallocOp>(a.getLoc(), a.getResult());
    rewriter.replaceOpWithNewOp<cf::BranchOp>(contOp, nextBlock,
                                              contOp->getOperands());
    // Loop continue from within scope with deallocations.
    if (Block *blk = iter->second.continueBlock) {
      rewriter.setInsertionPointToEnd(blk);
      for (auto a : llvm::reverse(qallocas))
        rewriter.create<quake::DeallocOp>(a.getLoc(), a.getResult());
      Block *landingPad = getLandingPad(infoMap, scope).continueBlock;
      rewriter.create<cf::BranchOp>(loc, landingPad, blk->getArguments());
      scope.getInitRegion().push_back(blk);
    }
    // Loop break from within scope with deallocations.
    if (Block *blk = iter->second.breakBlock) {
      rewriter.setInsertionPointToEnd(blk);
      for (auto a : llvm::reverse(qallocas))
        rewriter.create<quake::DeallocOp>(a.getLoc(), a.getResult());
      rewriter.create<cf::BranchOp>(
          loc, getLandingPad(infoMap, scope).breakBlock, blk->getArguments());
      scope.getInitRegion().push_back(blk);
    }
    // Function return from within scope with deallocations.
    if (Block *blk = iter->second.returnBlock) {
      rewriter.setInsertionPointToEnd(blk);
      for (auto a : llvm::reverse(qallocas))
        rewriter.create<quake::DeallocOp>(a.getLoc(), a.getResult());
      rewriter.create<cf::BranchOp>(
          loc, getLandingPad(infoMap, scope).returnBlock, blk->getArguments());
      scope.getInitRegion().push_back(blk);
    }
    rewriter.inlineRegionBefore(scope.getInitRegion(), nextBlock);
    rewriter.replaceOp(scope, contOp->getOperands());
    return success();
  }

  const UnwindOpAnalysisInfo &infoMap;
};

/// A func.func op is updated in-place to rewrite all returns to branches to a
/// return block. The return block will deallocate all quake.alloca operations
/// before returning from the function.
struct FuncOpPattern : public OpRewritePattern<func::FuncOp> {
  explicit FuncOpPattern(MLIRContext *ctx, const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    auto iter = infoMap.find(func.getOperation());
    assert(iter != infoMap.end());
    if (!iter->second.asPrimitive) {
      LLVM_DEBUG(llvm::dbgs() << "func was not marked as primitive in map\n");
      return success();
    }
    if (!func->hasAttr("add_dealloc"))
      return success();
    LLVM_DEBUG(llvm::dbgs() << "updating func " << func.getName() << '\n');
    // Cannot have a break or continue block.
    assert(!iter->second.breakBlock && !iter->second.continueBlock &&
           iter->second.returnBlock);
    // Scan the function for quake allocations.
    SmallVector<quake::AllocaOp> qallocas;
    for (Block &b : func.getBody())
      for (Operation &o : b)
        if (auto q = dyn_cast<quake::AllocaOp>(o))
          qallocas.push_back(q);
    // Add the new exit block to the end of the function with all the quake
    // deallocations. Don't need to worry about stack allocations as they are
    // about to be reclaimed when the function returns.
    Block *exitBlock = iter->second.returnBlock;
    rewriter.setInsertionPointToEnd(exitBlock);
    for (auto a : llvm::reverse(qallocas))
      rewriter.create<quake::DeallocOp>(a.getLoc(), a.getResult());
    rewriter.create<func::ReturnOp>(func.getLoc(), exitBlock->getArguments());
    rewriter.updateRootInPlace(func, [&]() {
      for (Block &b : func.getBody())
        for (Operation &o : b)
          if (isa<func::ReturnOp, cudaq::cc::ReturnOp>(o)) {
            rewriter.setInsertionPointToEnd(&b);
            rewriter.replaceOpWithNewOp<cf::BranchOp>(&o, exitBlock,
                                                      o.getOperands());
          }
      func.getBody().push_back(exitBlock);
      func->removeAttr("add_dealloc");
    });
    return success();
  }

  const UnwindOpAnalysisInfo &infoMap;
};

/// An `if` statement that contains an unwind macro is always lowered to a
/// primitive CFG. The presence or absence of scopes between the unwind op and
/// the nearest loop or function dictates whether the branching must be to
/// landing pads or not, resp.
struct IfOpPattern : public OpRewritePattern<cudaq::cc::IfOp> {
  explicit IfOpPattern(MLIRContext *ctx, const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(cudaq::cc::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto iter = infoMap.find(ifOp.getOperation());
    assert(iter != infoMap.end());
    if (!iter->second.asPrimitive)
      return success();
    LLVM_DEBUG(llvm::dbgs() << "replacing if @" << ifOp.getLoc() << '\n');

    // Decompose the cc.loop to a CFG.
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
    // Append blocks to tailRegion
    auto &tailRegion = hasElse ? ifOp.getElseRegion() : ifOp.getThenRegion();
    if (auto *blk = iter->second.continueBlock) {
      rewriter.setInsertionPointToEnd(blk);
      auto *dest = getLandingPad(infoMap, ifOp).continueBlock;
      rewriter.create<cf::BranchOp>(loc, dest, blk->getArguments());
      tailRegion.push_back(blk);
    }
    if (auto *blk = iter->second.breakBlock) {
      rewriter.setInsertionPointToEnd(blk);
      auto *dest = getLandingPad(infoMap, ifOp).breakBlock;
      rewriter.create<cf::BranchOp>(loc, dest, blk->getArguments());
      tailRegion.push_back(blk);
    }
    if (auto *blk = iter->second.returnBlock) {
      rewriter.setInsertionPointToEnd(blk);
      auto *dest = getLandingPad(infoMap, ifOp).returnBlock;
      rewriter.create<cf::BranchOp>(loc, dest, blk->getArguments());
      tailRegion.push_back(blk);
    }
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), endBlock);
    if (hasElse)
      rewriter.inlineRegionBefore(ifOp.getElseRegion(), endBlock);
    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(ifOp, ifOp.getCondition(),
                                                  thenBlock, ValueRange{},
                                                  elseBlock, ValueRange{});
    return success();
  }

  // Replace all the ContinueOp in the body region with branches to the correct
  // basic blocks.
  void updateBodyBranches(Region *bodyRegion, PatternRewriter &rewriter,
                          Block *continueBlock) const {
    // Walk body region and replace all continue and break ops.
    for (Block &block : *bodyRegion) {
      auto *terminator = block.getTerminator();
      // Handle the normal fall-through case.
      if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(terminator)) {
        rewriter.setInsertionPointToEnd(&block);
        LLVM_DEBUG(llvm::dbgs() << "replacing " << *terminator << '\n');
        rewriter.replaceOpWithNewOp<cf::BranchOp>(cont, continueBlock,
                                                  cont.getOperands());
      }
      // Other ad-hoc control flow in the region need not be rewritten.
    }
  }

  const UnwindOpAnalysisInfo &infoMap;
};

/// There are two cases for a loop construct. If the loop body does not contain
/// a return, then the loop can remain a high-level construct. Only the body
/// region needs to be converted to a primitive CFG. Otherwise, the loop
/// contains a return and the entire loop must be decomposed into a primitive
/// CFG. In either case, the presence or absence of scopes dictates whether the
/// branching must be to landing pads or not, resp.
struct LoopOpPattern : public OpRewritePattern<cudaq::cc::LoopOp> {
  explicit LoopOpPattern(MLIRContext *ctx, const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loopOp,
                                PatternRewriter &rewriter) const override {
    auto iter = infoMap.find(loopOp.getOperation());
    assert(iter != infoMap.end());
    if (!iter->second.asPrimitive)
      return success();
    LLVM_DEBUG(llvm::dbgs() << "replacing loop @" << loopOp.getLoc() << '\n');

    // Decompose the cc.loop to a CFG.
    auto loc = loopOp.getLoc();

    // Split the basic block in which this CLoop appears.
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPos);

    SmallVector<Value> loopOperands;
    loopOperands.append(loopOp.getOperands().begin(),
                        loopOp.getOperands().end());

    auto *whileBlock = &loopOp.getWhileRegion().front();
    auto whileCond = cast<cudaq::cc::ConditionOp>(whileBlock->getTerminator());
    if (loopOp.getNumResults() != 0) {
      Block *continueBlock = rewriter.createBlock(
          endBlock, loopOp.getResultTypes(),
          SmallVector<Location>(loopOp.getNumResults(), loc));
      rewriter.create<cf::BranchOp>(loc, endBlock);
      endBlock = continueBlock;
    }
    auto comparison = whileCond.getCondition();
    auto *bodyBlock = &loopOp.getBodyRegion().front();
    auto *condBlock =
        loopOp.hasStep()
            ? &loopOp.getStepRegion().front()
            : (loopOp.isPostConditional() ? bodyBlock : whileBlock);

    // Append blocks to tailRegion
    auto &tailRegion = loopOp.getBodyRegion();
    if (auto *blk = iter->second.continueBlock) {
      rewriter.setInsertionPointToEnd(blk);
      rewriter.create<cf::BranchOp>(loc, condBlock, blk->getArguments());
      tailRegion.push_back(blk);
    }
    if (auto *blk = iter->second.breakBlock) {
      rewriter.setInsertionPointToEnd(blk);
      rewriter.create<cf::BranchOp>(loc, endBlock, blk->getArguments());
      tailRegion.push_back(blk);
    }
    if (auto *blk = iter->second.returnBlock) {
      rewriter.setInsertionPointToEnd(blk);
      auto *retBlk = getLandingPad(infoMap, loopOp).returnBlock;
      assert(retBlk);
      rewriter.create<cf::BranchOp>(loc, retBlk, blk->getArguments());
      tailRegion.push_back(blk);
    }

    // Update the normal local branches.
    updateBodyBranches(&loopOp.getBodyRegion(), rewriter, condBlock, endBlock);
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
      // Branch from `initBlock` to whileRegion().front().
      rewriter.setInsertionPointToEnd(initBlock);
      rewriter.create<cf::BranchOp>(loc, whileBlock, loopOperands);
      // Replace the condition op with a `cf.cond_br` op.
      rewriter.setInsertionPointToEnd(whileBlock);
      rewriter.create<cf::CondBranchOp>(loc, comparison, bodyBlock,
                                        whileCond.getResults(), endBlock,
                                        whileCond.getResults());
      rewriter.eraseOp(whileCond);
      // Move the while and body region blocks between initBlock and endBlock.
      rewriter.inlineRegionBefore(loopOp.getWhileRegion(), endBlock);
      rewriter.inlineRegionBefore(loopOp.getBodyRegion(), endBlock);
      // If there is a step region, replace the continue op with a branch and
      // move the region between the body region and end block.
      if (loopOp.hasStep()) {
        auto *stepBlock = &loopOp.getStepRegion().front();
        auto *terminator = stepBlock->getTerminator();
        rewriter.setInsertionPointToEnd(stepBlock);
        rewriter.create<cf::BranchOp>(loc, whileBlock,
                                      terminator->getOperands());
        rewriter.eraseOp(terminator);
        rewriter.inlineRegionBefore(loopOp.getStepRegion(), endBlock);
      }
    }
    rewriter.replaceOp(loopOp, whileCond.getResults());
    return success();
  }

  // Replace all the ContinueOp and BreakOp in the body region with branches to
  // the correct basic blocks.
  void updateBodyBranches(Region *bodyRegion, PatternRewriter &rewriter,
                          Block *continueBlock, Block *breakBlock) const {
    // Walk body region and replace all continue and break ops.
    for (Block &block : *bodyRegion) {
      auto *terminator = block.getTerminator();
      rewriter.setInsertionPointToEnd(&block);
      if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(terminator))
        rewriter.replaceOpWithNewOp<cf::BranchOp>(cont, continueBlock,
                                                  cont.getOperands());
      else if (auto brk = dyn_cast<cudaq::cc::BreakOp>(terminator))
        rewriter.replaceOpWithNewOp<cf::BranchOp>(brk, breakBlock,
                                                  brk.getOperands());
      // Other ad-hoc control flow within the register need not be rewritten.
    }
  }

  const UnwindOpAnalysisInfo &infoMap;
};

/// A `return` statement is a global transfer of control that unwinds all
/// current scope contexts up to an including the enclosing function, then
/// returns from the function.
struct UnwindReturnOpPattern
    : public OpRewritePattern<cudaq::cc::UnwindReturnOp> {
  explicit UnwindReturnOpPattern(MLIRContext *ctx,
                                 const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(cudaq::cc::UnwindReturnOp retOp,
                                PatternRewriter &rewriter) const override {
    auto iter = infoMap.find(retOp.getOperation());
    assert(iter != infoMap.end());
    auto *blk = rewriter.getInsertionBlock();
    auto pos = rewriter.getInsertionPoint();
    rewriter.splitBlock(blk, std::next(pos));
    LLVM_DEBUG(llvm::dbgs() << "replacing " << retOp);
    rewriter.replaceOpWithNewOp<cf::BranchOp>(
        retOp, getLandingPad(retOp, infoMap), retOp.getOperands());
    return success();
  }

  const UnwindOpAnalysisInfo &infoMap;
};

template <typename TO, typename FROM>
LogicalResult intraLoopJump(FROM op, PatternRewriter &rewriter,
                            const UnwindOpAnalysisInfo &infoMap) {
  auto iter = infoMap.find(op.getOperation());
  assert(iter != infoMap.end());
  auto *blk = rewriter.getInsertionBlock();
  auto pos = rewriter.getInsertionPoint();
  rewriter.splitBlock(blk, std::next(pos));
  if (iter->second.asPrimitive)
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, getLandingPad(op, infoMap),
                                              op.getOperands());
  else
    rewriter.replaceOpWithNewOp<TO>(op, op.getOperands());
  return success();
}

/// A `break` statement is a global transfer of control that unwinds all current
/// scope contexts up to an including the nearest loop construct. The loop
/// terminates (or returns).
struct UnwindBreakOpPattern
    : public OpRewritePattern<cudaq::cc::UnwindBreakOp> {
  explicit UnwindBreakOpPattern(MLIRContext *ctx,
                                const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(cudaq::cc::UnwindBreakOp brkOp,
                                PatternRewriter &rewriter) const override {
    return intraLoopJump<cudaq::cc::BreakOp>(brkOp, rewriter, infoMap);
  }

  const UnwindOpAnalysisInfo &infoMap;
};

/// A `continue` statement is a global transfer of control that unwinds all
/// current scope contexts up to the loop's body statement. The loop iteration
/// terminates and control transfers to the next iteration of the loop.
struct UnwindContinueOpPattern
    : public OpRewritePattern<cudaq::cc::UnwindContinueOp> {
  explicit UnwindContinueOpPattern(MLIRContext *ctx,
                                   const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(cudaq::cc::UnwindContinueOp cntOp,
                                PatternRewriter &rewriter) const override {
    return intraLoopJump<cudaq::cc::ContinueOp>(cntOp, rewriter, infoMap);
  }

  const UnwindOpAnalysisInfo &infoMap;
};

class UnwindLoweringPass
    : public cudaq::opt::UnwindLoweringBase<UnwindLoweringPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    UnwindOpAnalysis analysis(func);
    auto unwindInfo = analysis.getAnalysisInfo();
    // If there are no unwinding goto ops, then leave the function as-is.
    if (unwindInfo.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "analysis found no unwinds\n");
      return;
    }
    // Otherwise lower the structured constructs to expose the CFG structure.
    auto *ctx = func.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<UnwindBreakOpPattern, UnwindContinueOpPattern,
                    UnwindReturnOpPattern, IfOpPattern, LoopOpPattern,
                    FuncOpPattern, ScopeOpPattern>(ctx, unwindInfo);
    ConversionTarget target(*ctx);
    target.addIllegalOp<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
                        cudaq::cc::UnwindReturnOp>();
    target.addDynamicallyLegalOp<cudaq::cc::IfOp, cudaq::cc::LoopOp,
                                 cudaq::cc::ScopeOp>([&](Operation *op) {
      auto iter = unwindInfo.find(op);
      if (iter == unwindInfo.end())
        return true;
      return !iter->second.asPrimitive;
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
      auto iter = unwindInfo.find(op);
      if (iter == unwindInfo.end())
        return true;
      return !op->hasAttr("add_dealloc");
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      emitError(func.getLoc(), "error unwinding control flow\n");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createUnwindLoweringPass() {
  return std::make_unique<UnwindLoweringPass>();
}
