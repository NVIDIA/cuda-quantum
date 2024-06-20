/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

struct BlockInfo {
  Block *continueBlock = nullptr; // unwind_continue landing pads
  Block *breakBlock = nullptr;    // unwind_break landing pads
  Block *returnBlock = nullptr;   // unwind_return landing pads
};

struct UnwindGotoAsPrimitive {
  Operation *parent = nullptr; // the original parent op
  bool asPrimitive = false;    // convert to cf.br?
};

struct BlockDetails {
  using Key = unsigned;
  DenseMap<Operation *, Key> keyMap;
  DenseMap<Key, SmallVector<Operation *>> allocaDomMap;
  DenseMap<Key, BlockInfo> blockMap;
};

struct UnwindOpAnalysisInfo {
  DenseMap<Operation *, UnwindGotoAsPrimitive> opParentMap;
  DenseMap<Operation *, BlockDetails> blockDetails;

  bool empty() const { return opParentMap.empty(); }
};

struct UnwindOpAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnwindOpAnalysis)

  UnwindOpAnalysis(func::FuncOp op, DominanceInfo &di) : func(op), dom(di) {
    performAnalysis(op.getOperation());
  }

  UnwindOpAnalysisInfo getAnalysisInfo() const { return infoMap; }

private:
  Operation *getParent(Operation *p) {
    return (!p || isa<func::FuncOp, cudaq::cc::CreateLambdaOp>(p))
               ? nullptr
               : p->getParentOp();
  }

  template <typename A>
  void addParents(A unwindOp) {
    bool asPrimitive = false;
    Operation *parent = unwindOp->getParentOp();
    if constexpr (std::is_same_v<A, cudaq::cc::UnwindReturnOp>) {
      asPrimitive = true;
    } else {
      // The unwind can be demoted to a high-level break or continue if there is
      // no scope between the unwind and the nearest enclosing loop.
      for (Operation *parent = unwindOp->getParentOp();
           !isa<cudaq::cc::LoopOp>(parent) && !asPrimitive;
           parent = parent->getParentOp())
        asPrimitive = isa<func::FuncOp, cudaq::cc::CreateLambdaOp>(parent);
    }
    auto *op = unwindOp.getOperation();
    infoMap.opParentMap.insert({op, {parent, asPrimitive}});
    for (auto *p = parent; p; p = parent) {
      if constexpr (std::is_same_v<A, cudaq::cc::UnwindReturnOp>) {
        parent = getParent(p);
      } else {
        parent = isa<cudaq::cc::LoopOp>(p) ? nullptr : getParent(p);
      }
      if (infoMap.opParentMap.count(p)) {
        // p is already in the op-parent map, so merge new information.
        if (!infoMap.opParentMap[p].parent)
          infoMap.opParentMap[p].parent = parent;
        infoMap.opParentMap[p].asPrimitive |= asPrimitive;
      } else {
        // p is not in the map, so add it with the information.
        infoMap.opParentMap.insert({p, {parent, asPrimitive}});
      }
    }
  }

  template <typename A>
  Block *createNewBlock(A unwindOp) {
    auto *newBlock = new Block;
    // Add blocks arguments corresponding to the jump to thread results.
    auto argTys = unwindOp.getOperandTypes();
    SmallVector<Location> locations(argTys.size(), unwindOp.getLoc());
    newBlock->addArguments(argTys, locations);
    return newBlock;
  }

  void performAnalysis(Operation *func) {
    // 1) Find all the unwind jump operations and add them and their parents to
    // our analysis map. Record if high-level control flow must be decomposed to
    // primitive control flow.
    func->walk([this](Operation *o) {
      if (auto brkOp = dyn_cast<cudaq::cc::UnwindBreakOp>(o))
        addParents(brkOp);
      else if (auto cntOp = dyn_cast<cudaq::cc::UnwindContinueOp>(o))
        addParents(cntOp);
      else if (auto retOp = dyn_cast<cudaq::cc::UnwindReturnOp>(o))
        addParents(retOp);
    });
    if (infoMap.opParentMap.empty())
      return;

    // 2) Walk all the parent ops and add an empty block details entry for each.
    // (The unwind jumps do not have block details.)
    for (auto &pr : infoMap.opParentMap) {
      auto *key = pr.first;
      if (!isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
               cudaq::cc::UnwindReturnOp>(key)) {
        if (!infoMap.blockDetails.count(key)) {
          BlockDetails details;
          LLVM_DEBUG(llvm::dbgs() << "adding to details " << key << '\n');
          infoMap.blockDetails.insert({key, details});
        }
      }
    }

    // 3) Find scopes that contain quantum allocations.
    DenseMap<Operation *, SmallVector<Operation *>> scopeAllocMap;
    for (auto &pr : infoMap.opParentMap) {
      if (isa<func::FuncOp, cudaq::cc::ScopeOp, cudaq::cc::CreateLambdaOp>(
              pr.first)) {
        SmallVector<Operation *> allocas;
        for (auto &region : pr.first->getRegions())
          for (auto &block : region)
            for (auto &o : block)
              if (isa<quake::AllocaOp>(o))
                allocas.push_back(&o);
        if (!allocas.empty())
          scopeAllocMap[pr.first] = allocas;
      }
    }

    // 4) Walk the parent chains and record a reverse map from the unwind jump
    // to a set of allocations that dominate the unwind jump at this parent. The
    // set may be empty. For the set of allocations, create a block entry as
    // needed. The block will be shared for all incident unwind jumps in the
    // reverse map with equal allocation sets.
    std::map<SmallVector<Operation *>, BlockDetails::Key> uniqAllocas;
    BlockDetails::Key uniqValue = 0;
    for (auto &pr : infoMap.opParentMap) {
      if (isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
              cudaq::cc::UnwindReturnOp>(pr.first)) {
        auto *currentOp = pr.first;
        for (auto *p = pr.second.parent; p; p = getParent(p)) {
          assert(infoMap.blockDetails.count(p));

          // Compute the subset of allocas that dominate the current op.
          SmallVector<Operation *> domAllocas;
          if (scopeAllocMap.count(p)) {
            for (auto *a : scopeAllocMap[p])
              if (dom.dominates(a, currentOp))
                domAllocas.push_back(a);
          }

          // Map the list of allocas to a unique key value.
          auto ui = uniqAllocas.find(domAllocas);
          BlockDetails::Key key;
          if (ui == uniqAllocas.end()) {
            key = uniqValue++;
            uniqAllocas.insert({domAllocas, key});
          } else {
            key = ui->second;
          }

          // Add a relation from <parent x unwind> -> [dominating allocas]
          auto &details = infoMap.blockDetails[p];
          details.keyMap[currentOp] = key;
          if (!details.allocaDomMap.count(key))
            details.allocaDomMap[key] = domAllocas;

          // Depending on the type of unwind add a relation from <parent x
          // [dominating allocas]> -> {target blocks}
          auto &blockInfo = details.blockMap[key];
          if (auto unwindOp = dyn_cast<cudaq::cc::UnwindBreakOp>(pr.first)) {
            if (!blockInfo.breakBlock)
              blockInfo.breakBlock = createNewBlock(unwindOp);
          } else if (auto unwindOp =
                         dyn_cast<cudaq::cc::UnwindContinueOp>(pr.first)) {
            if (!blockInfo.continueBlock)
              blockInfo.continueBlock = createNewBlock(unwindOp);
          } else if (auto unwindOp =
                         dyn_cast<cudaq::cc::UnwindReturnOp>(pr.first)) {
            if (!blockInfo.returnBlock)
              blockInfo.returnBlock = createNewBlock(unwindOp);
          }
          if (isa<cudaq::cc::LoopOp>(p) &&
              isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp>(
                  pr.first)) {
            p = nullptr;
          }
          currentOp = p;
        }
      }
    }
    if (infoMap.blockDetails.count(func))
      func->setAttr("add_dealloc", UnitAttr::get(func->getContext()));
  }

  func::FuncOp func;
  UnwindOpAnalysisInfo infoMap;
  DominanceInfo &dom;
};
} // namespace

static Operation *originalParent(const UnwindOpAnalysisInfo &infoMap,
                                 Operation *arg) {
  auto iter = infoMap.opParentMap.find(arg);
  assert(iter != infoMap.opParentMap.end());
  return iter->second.parent;
}

static const BlockInfo &getLandingPad(const UnwindOpAnalysisInfo &infoMap,
                                      Operation *arg) {
  auto *parent = originalParent(infoMap, arg);
  LLVM_DEBUG(llvm::dbgs() << "op " << arg << " has parent " << parent << '\n');
  auto iter = infoMap.blockDetails.find(parent);
  assert(iter != infoMap.blockDetails.end() && "parent not added");
  auto &details = iter->second;
  auto jter = details.keyMap.find(arg);
  assert(jter != details.keyMap.end() && "no block details for enclosed op");
  auto key = jter->second;
  auto kter = details.blockMap.find(key);
  assert(kter != details.blockMap.end() && "map of deallocations not added");
  LLVM_DEBUG(llvm::dbgs() << "arg " << arg << " has {" << parent << " ["
                          << kter->second.continueBlock << ' '
                          << kter->second.breakBlock << ' '
                          << kter->second.returnBlock << "] "
                          << infoMap.opParentMap.find(arg)->second.asPrimitive
                          << "}\n");
  return kter->second;
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

static SmallVector<Operation *> populateExitTerminators(Region &reg) {
  SmallVector<Operation *> results;
  for (Block &b : reg)
    if (b.getSuccessors().empty())
      results.push_back(b.getTerminator());
  return results;
}

static SmallVector<quake::AllocaOp> populateQuakeAllocas(Region &reg) {
  SmallVector<quake::AllocaOp> results;
  for (Block &b : reg)
    for (Operation &o : b)
      if (auto q = dyn_cast<quake::AllocaOp>(o))
        results.push_back(q);
  return results;
}

static DenseMap<Operation *, SmallVector<quake::AllocaOp>>
populateTerminatorAllocaMap(const SmallVector<Operation *> &terminators,
                            const SmallVector<quake::AllocaOp> &qallocas,
                            DominanceInfo &dom) {
  DenseMap<Operation *, SmallVector<quake::AllocaOp>> results;
  for (auto *t : terminators) {
    SmallVector<quake::AllocaOp> domList;
    for (auto a : qallocas)
      if (dom.dominates(a.getOperation(), t))
        domList.push_back(a);
    results.insert({t, domList});
  }
  return results;
}

static bool anyPrimitiveAncestor(
    const DenseMap<Operation *, UnwindGotoAsPrimitive> &opParentMap,
    Operation *op) {
  for (auto iter = opParentMap.find(op); iter != opParentMap.end();) {
    auto *parent = iter->second.parent;
    if (iter->second.asPrimitive)
      return true;
    if (!parent)
      break;
    iter = opParentMap.find(parent);
  }
  return false;
}

static Value adjustedDeallocArg(quake::AllocaOp alloc) {
  if (auto init = alloc.getInitializedState())
    return init.getResult();
  return alloc.getResult();
}

static Value adjustedDeallocArg(Operation *op) {
  return adjustedDeallocArg(cast<quake::AllocaOp>(op));
}

namespace {
/// A scope op that contains an unwind op and is contained by a loop (for break
/// or continue) or for return always, dictates that the unwind op must transfer
/// control to a landing pad for the continue, break, or return semantics of
/// that scope. The exact lowering of this control transfer is determined in the
/// analysis.
struct ScopeOpPattern : public OpRewritePattern<cudaq::cc::ScopeOp> {
  explicit ScopeOpPattern(MLIRContext *ctx, const UnwindOpAnalysisInfo &info,
                          DominanceInfo &di)
      : OpRewritePattern(ctx), infoMap(info), dom(di) {}

  LogicalResult matchAndRewrite(cudaq::cc::ScopeOp scope,
                                PatternRewriter &rewriter) const override {
    [[maybe_unused]] auto iter = infoMap.opParentMap.find(scope.getOperation());
    assert(iter != infoMap.opParentMap.end());
    bool asPrimitive =
        anyPrimitiveAncestor(infoMap.opParentMap, scope.getOperation());
    LLVM_DEBUG(llvm::dbgs() << "replacing scope @" << scope.getLoc() << '\n');
    auto loc = scope.getLoc();
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *nextBlock = rewriter.splitBlock(initBlock, initPos);
    auto *scopeBlock = &scope.getInitRegion().front();

    // Find all terminators that leave the ScopeOp.
    auto terminators = populateExitTerminators(scope.getInitRegion());
    // Scan the scope for quantum allocations.
    auto qallocas = populateQuakeAllocas(scope.getInitRegion());
    auto termAllocMap = populateTerminatorAllocaMap(terminators, qallocas, dom);

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
    // Normal scope exit with inline deallocations.
    for (auto &pr : termAllocMap) {
      auto *contOp = pr.first;
      rewriter.setInsertionPoint(contOp);
      for (auto a : llvm::reverse(pr.second))
        rewriter.create<quake::DeallocOp>(a.getLoc(), adjustedDeallocArg(a));
      rewriter.replaceOpWithNewOp<cf::BranchOp>(contOp, nextBlock,
                                                contOp->getOperands());
    }
    auto blockMapIter = infoMap.blockDetails.find(scope.getOperation());
    assert(blockMapIter != infoMap.blockDetails.end());
    auto &details = blockMapIter->second;
    for (auto &pr : details.blockMap) {
      auto &blockInfo = pr.second;
      auto &qallocas = details.allocaDomMap.find(pr.first)->second;
      // Loop continue from within scope with deallocations.
      if (Block *blk = blockInfo.continueBlock) {
        rewriter.setInsertionPointToEnd(blk);
        for (auto a : llvm::reverse(qallocas))
          rewriter.create<quake::DeallocOp>(a->getLoc(), adjustedDeallocArg(a));
        if (asPrimitive) {
          Block *landingPad = getLandingPad(infoMap, scope).continueBlock;
          rewriter.create<cf::BranchOp>(loc, landingPad, blk->getArguments());
        } else {
          rewriter.create<cudaq::cc::ContinueOp>(loc, blk->getArguments());
        }
        scope.getInitRegion().push_back(blk);
      }
      // Loop break from within scope with deallocations.
      if (Block *blk = blockInfo.breakBlock) {
        rewriter.setInsertionPointToEnd(blk);
        for (auto a : llvm::reverse(qallocas))
          rewriter.create<quake::DeallocOp>(a->getLoc(), adjustedDeallocArg(a));
        if (asPrimitive) {
          Block *landingPad = getLandingPad(infoMap, scope).breakBlock;
          rewriter.create<cf::BranchOp>(loc, landingPad, blk->getArguments());
        } else {
          rewriter.create<cudaq::cc::BreakOp>(loc, blk->getArguments());
        }
        scope.getInitRegion().push_back(blk);
      }
      // Function return from within scope with deallocations.
      if (Block *blk = blockInfo.returnBlock) {
        rewriter.setInsertionPointToEnd(blk);
        for (auto a : llvm::reverse(qallocas))
          rewriter.create<quake::DeallocOp>(a->getLoc(), adjustedDeallocArg(a));
        assert(asPrimitive);
        Block *landingPad = getLandingPad(infoMap, scope).returnBlock;
        rewriter.create<cf::BranchOp>(loc, landingPad, blk->getArguments());
        scope.getInitRegion().push_back(blk);
      }
    }
    rewriter.inlineRegionBefore(scope.getInitRegion(), nextBlock);
    rewriter.replaceOp(scope, nextBlock->getArguments());
    return success();
  }

  const UnwindOpAnalysisInfo &infoMap;
  DominanceInfo &dom;
};

/// A func.func op is updated in-place to rewrite all returns to branches to a
/// return block. The return block will deallocate all quake.alloca operations
/// before returning from the function.
template <typename OP, typename TERM>
struct FuncLikeOpPattern : public OpRewritePattern<OP> {
  using Base = OpRewritePattern<OP>;

  explicit FuncLikeOpPattern(MLIRContext *ctx, const UnwindOpAnalysisInfo &info,
                             DominanceInfo &di)
      : Base(ctx), infoMap(info), dom(di) {}

  LogicalResult matchAndRewrite(OP func,
                                PatternRewriter &rewriter) const override {
    auto iter = infoMap.opParentMap.find(func.getOperation());
    assert(iter != infoMap.opParentMap.end());
    if (!func->hasAttr("add_dealloc"))
      return success();
    rewriter.updateRootInPlace(func,
                               [&]() { func->removeAttr("add_dealloc"); });
    if (!iter->second.asPrimitive) {
      LLVM_DEBUG(llvm::dbgs() << "func was not marked as primitive in map\n");
      return success();
    }
    LLVM_DEBUG(llvm::dbgs() << "updating func " << func.getName() << '\n');

    // Find all terminators that leave the ScopeOp.
    auto terminators = populateExitTerminators(func.getBody());
    // Scan the scope for quantum allocations.
    auto qallocas = populateQuakeAllocas(func.getBody());
    auto termAllocMap = populateTerminatorAllocaMap(terminators, qallocas, dom);

    // Normal func return with inline deallocations.
    for (auto &pr : termAllocMap) {
      auto *exitOp = pr.first;
      rewriter.setInsertionPoint(exitOp);
      for (auto a : llvm::reverse(pr.second))
        rewriter.create<quake::DeallocOp>(a.getLoc(), adjustedDeallocArg(a));
    }

    // Here, we handle the unwind return jumps.
    auto blockMapIter = infoMap.blockDetails.find(func.getOperation());
    assert(blockMapIter != infoMap.blockDetails.end());
    auto &details = blockMapIter->second;
    for (auto &pr : details.blockMap) {
      auto &blockInfo = pr.second;
      auto &qallocas = details.allocaDomMap.find(pr.first)->second;
      assert(!blockInfo.continueBlock && !blockInfo.breakBlock &&
             "FuncOp is not a loop");

      // Add the new exit block to the end of the function with all the quake
      // deallocations. Don't need to worry about stack allocations as they are
      // about to be reclaimed when the function returns.
      if (Block *exitBlock = blockInfo.returnBlock) {
        rewriter.setInsertionPointToEnd(exitBlock);
        for (auto a : llvm::reverse(qallocas))
          rewriter.create<quake::DeallocOp>(a->getLoc(), adjustedDeallocArg(a));
        rewriter.create<TERM>(func.getLoc(), exitBlock->getArguments());
        func.getBody().push_back(exitBlock);
      }
    }
    return success();
  }

  const UnwindOpAnalysisInfo &infoMap;
  DominanceInfo &dom;
};

using FuncOpPattern = FuncLikeOpPattern<func::FuncOp, func::ReturnOp>;
using CreateLambdaOpPattern =
    FuncLikeOpPattern<cudaq::cc::CreateLambdaOp, cudaq::cc::ReturnOp>;

/// An `if` statement that contains an unwind macro is always lowered to a
/// primitive CFG. The presence or absence of scopes between the unwind op and
/// the nearest loop or function dictates whether the branching must be to
/// landing pads or not, resp.
struct IfOpPattern : public OpRewritePattern<cudaq::cc::IfOp> {
  explicit IfOpPattern(MLIRContext *ctx, const UnwindOpAnalysisInfo &info)
      : OpRewritePattern(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(cudaq::cc::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto iter = infoMap.opParentMap.find(ifOp.getOperation());
    assert(iter != infoMap.opParentMap.end());
    LLVM_DEBUG(llvm::dbgs() << "replacing if @" << ifOp.getLoc() << '\n');

    // Decompose the cc.if to a CFG.
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
    // If the if statement is marked as primitive, add the jumps to the
    // continue, break, and/or return blocks of the parent. Otherwise, the
    // control-flow will be within the region of the parent and the branches
    // aren't needed.
    if (anyPrimitiveAncestor(infoMap.opParentMap, ifOp.getOperation())) {
      // Append blocks to tailRegion.
      auto &tailRegion = hasElse ? ifOp.getElseRegion() : ifOp.getThenRegion();
      auto blockMapIter = infoMap.blockDetails.find(ifOp.getOperation());
      assert(blockMapIter != infoMap.blockDetails.end());
      auto &details = blockMapIter->second;
      for (auto &pr : details.blockMap) {
        auto &blockInfo = pr.second;
        assert(details.allocaDomMap.find(pr.first)->second.empty());
        if (auto *blk = blockInfo.continueBlock) {
          rewriter.setInsertionPointToEnd(blk);
          auto *dest = getLandingPad(infoMap, ifOp).continueBlock;
          rewriter.create<cf::BranchOp>(loc, dest, blk->getArguments());
          tailRegion.push_back(blk);
        }
        if (auto *blk = blockInfo.breakBlock) {
          rewriter.setInsertionPointToEnd(blk);
          auto *dest = getLandingPad(infoMap, ifOp).breakBlock;
          rewriter.create<cf::BranchOp>(loc, dest, blk->getArguments());
          tailRegion.push_back(blk);
        }
        if (auto *blk = blockInfo.returnBlock) {
          rewriter.setInsertionPointToEnd(blk);
          auto *dest = getLandingPad(infoMap, ifOp).returnBlock;
          rewriter.create<cf::BranchOp>(loc, dest, blk->getArguments());
          tailRegion.push_back(blk);
        }
      }
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
    auto iter = infoMap.opParentMap.find(loopOp.getOperation());
    assert(iter != infoMap.opParentMap.end());
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
    auto blockMapIter = infoMap.blockDetails.find(loopOp.getOperation());
    assert(blockMapIter != infoMap.blockDetails.end());
    auto &details = blockMapIter->second;
    for (auto &pr : details.blockMap) {
      auto &blockInfo = pr.second;
      assert(details.allocaDomMap.find(pr.first)->second.empty());
      if (auto *blk = blockInfo.continueBlock) {
        rewriter.setInsertionPointToEnd(blk);
        rewriter.create<cf::BranchOp>(loc, condBlock, blk->getArguments());
        tailRegion.push_back(blk);
      }
      if (auto *blk = blockInfo.breakBlock) {
        rewriter.setInsertionPointToEnd(blk);
        rewriter.create<cf::BranchOp>(loc, endBlock, blk->getArguments());
        tailRegion.push_back(blk);
      }
      if (auto *blk = blockInfo.returnBlock) {
        rewriter.setInsertionPointToEnd(blk);
        auto *retBlk = getLandingPad(infoMap, loopOp).returnBlock;
        assert(retBlk);
        rewriter.create<cf::BranchOp>(loc, retBlk, blk->getArguments());
        tailRegion.push_back(blk);
      }
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
    [[maybe_unused]] auto iter = infoMap.opParentMap.find(retOp.getOperation());
    assert(iter != infoMap.opParentMap.end());
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
  auto iter = infoMap.opParentMap.find(op.getOperation());
  assert(iter != infoMap.opParentMap.end());
  auto *blk = rewriter.getInsertionBlock();
  auto pos = rewriter.getInsertionPoint();
  rewriter.splitBlock(blk, std::next(pos));
  if (isa<cudaq::cc::ScopeOp>(iter->second.parent) ||
      anyPrimitiveAncestor(infoMap.opParentMap, op.getOperation()))
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, getLandingPad(op, infoMap),
                                              op.getOperands());
  else
    rewriter.replaceOpWithNewOp<TO>(op, op.getOperands());
  return success();
}

template <typename OP, typename TERM>
struct UnwindLoopJumpOpPattern : public OpRewritePattern<OP> {
  using Base = OpRewritePattern<OP>;

  explicit UnwindLoopJumpOpPattern(MLIRContext *ctx,
                                   const UnwindOpAnalysisInfo &info)
      : Base(ctx), infoMap(info) {}

  LogicalResult matchAndRewrite(OP brkOp,
                                PatternRewriter &rewriter) const override {
    return intraLoopJump<TERM>(brkOp, rewriter, infoMap);
  }

  const UnwindOpAnalysisInfo &infoMap;
};

/// A `break` statement is a global transfer of control that unwinds all current
/// scope contexts up to an including the nearest loop construct. The loop
/// terminates (or returns).
using UnwindBreakOpPattern =
    UnwindLoopJumpOpPattern<cudaq::cc::UnwindBreakOp, cudaq::cc::BreakOp>;

/// A `continue` statement is a global transfer of control that unwinds all
/// current scope contexts up to the loop's body statement. The loop iteration
/// terminates and control transfers to the next iteration of the loop.
using UnwindContinueOpPattern =
    UnwindLoopJumpOpPattern<cudaq::cc::UnwindContinueOp, cudaq::cc::ContinueOp>;

class UnwindLoweringPass
    : public cudaq::opt::UnwindLoweringBase<UnwindLoweringPass> {
public:
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    DominanceInfo domInfo(func);
    UnwindOpAnalysis analysis(func, domInfo);
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
                    UnwindReturnOpPattern, IfOpPattern, LoopOpPattern>(
        ctx, unwindInfo);
    patterns.insert<FuncOpPattern, CreateLambdaOpPattern, ScopeOpPattern>(
        ctx, unwindInfo, domInfo);
    ConversionTarget target(*ctx);
    target.addIllegalOp<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
                        cudaq::cc::UnwindReturnOp>();
    target.addDynamicallyLegalOp<cudaq::cc::LoopOp>([&](Operation *op) {
      auto iter = unwindInfo.opParentMap.find(op);
      if (iter == unwindInfo.opParentMap.end())
        return true;
      return !iter->second.asPrimitive;
    });
    target.addDynamicallyLegalOp<cudaq::cc::IfOp, cudaq::cc::ScopeOp>(
        [&](Operation *op) {
          auto iter = unwindInfo.opParentMap.find(op);
          return iter == unwindInfo.opParentMap.end();
        });
    target.addDynamicallyLegalOp<func::FuncOp, cudaq::cc::CreateLambdaOp>(
        [&](Operation *op) {
          if (!unwindInfo.opParentMap.count(op))
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
