/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#define GEN_PASS_DEF_VARIABLECOALESCE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "variable-coalesce"

using namespace mlir;

namespace {
struct AllocationAnalysis {
  explicit AllocationAnalysis(Operation *op, bool hoistOnly)
      : hoistOnly(hoistOnly) {
    initialize(op);
    if (varsToMove.empty())
      return;
    coalesceVariablesByType();
    parentScopes.clear();
    scopeMap.clear();
  }

  Operation * /*AllocaOp*/
  getCoalescedTo(Operation * /*AllocaOp*/ mergee) const {
    auto iter = coalesceMap.find(mergee);
    if (iter != coalesceMap.end())
      return iter->second;
    return mergee;
  }

  void addBinding(Operation * /*AllocaOp*/ merged,
                  Operation * /*AllocaOp*/ newVar) {
    assert(newVar && "new variable must be defined");
    bindingMap[merged] = newVar;
  }

  Operation * /*AllocaOp*/ getBinding(Operation *merged) {
    auto iter = bindingMap.find(merged);
    if (iter == bindingMap.end())
      return {};
    return iter->second;
  }

  bool toBeMoved(cudaq::cc::AllocaOp alloc) const {
    return varsToMove.contains(alloc);
  }

  const SetVector<Operation * /*AllocaOp*/> &getVarsToMove() const {
    return varsToMove;
  }

  bool nothingToDo() const { return varsToMove.empty(); }

private:
  void initialize(Operation *op) {
    op->walk([&](cudaq::cc::AllocaOp alloc) {
      if (alloc.getSeqSize())
        return WalkResult::advance();
      auto *parent = alloc->getParentOp();
      if (isa<func::FuncOp, cudaq::cc::CreateLambdaOp>(parent))
        return WalkResult::advance();
      if (auto scope = dyn_cast<cudaq::cc::ScopeOp>(parent)) {
        varsToMove.insert(alloc);
        scopeMap[alloc] = scope;
        updateScopeTree(scope);
        return WalkResult::advance();
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Variables should be allocated in a cc.scope.\n");
      varsToMove.clear();
      return WalkResult::interrupt();
    });
  }

  void coalesceVariablesByType() {
    if (hoistOnly)
      return;

    DenseMap<Type::ImplType *, SmallVector<Operation * /*AllocaOp*/>> buckets;

    // Sort all the allocs into buckets by type.
    for (auto *o : varsToMove) {
      auto a = cast<cudaq::cc::AllocaOp>(o);
      buckets[a.getElementType().getImpl()].push_back(a);
    }

    // Prioritize buckets by scope depth for better coalescing.
    for (auto &iter : buckets) {
      std::sort(iter.second.begin(), iter.second.end(),
                [&](Operation *o1, Operation *o2) {
                  auto *sc1 = scopeMap[o1];
                  auto *sc2 = scopeMap[o2];
                  assert(sc1 && sc2);
                  return parentScopes[sc1].size() > parentScopes[sc2].size();
                });
    }

    // Go through each bucket and coalesce variables if and only if their scopes
    // are disjoint.
    for (auto &iter : buckets) {
      auto *v = iter.second.front();
      SmallVector<Operation * /*ScopeOp*/> liveScopes;
      updateScopes(v, liveScopes);
      ArrayRef<Operation * /*AllocaOp*/> others{iter.second.begin() + 1,
                                                iter.second.end()};
      for (auto *w : others) {
        if (disjointScopes(liveScopes, w)) {
          coalesceWith(v, w);
          updateScopes(w, liveScopes);
        }
      }
    }
  }

  void updateScopes(Operation * /*AllocaOp*/ alloc,
                    SmallVectorImpl<Operation * /*ScopeOp*/> &liveScopes) {
    auto *s = scopeMap[alloc];
    liveScopes.append(parentScopes[s].begin(), parentScopes[s].end());
  }

  // The scope of alloca, `a`, is disjoint from the current set if it cannot be
  // found in the set of scopes. Correctness here relies on the scopes being
  // prioritized such that a variable in a parent scope is always processed
  // *after* any child scopes. That processing order guarantees that the parent
  // scope is already in `scopes`.
  bool disjointScopes(const SmallVectorImpl<Operation * /*ScopeOp*/> &scopes,
                      Operation * /*AllocaOp*/ a) {
    auto *s = scopeMap[a];
    return std::find(scopes.begin(), scopes.end(), s) == scopes.end();
  }

  void coalesceWith(Operation * /*AllocaOp*/ merged,
                    Operation * /*AllocaOp*/ mergee) {
    LLVM_DEBUG(llvm::dbgs()
               << "adding coalesce of " << mergee << " -> " << merged << '\n');
    coalesceMap[mergee] = merged;
  }

  void updateScopeTree(cudaq::cc::ScopeOp scope) {
    parentScopes[scope].insert(scope);
    for (auto *p = scope->getParentOp(); p; p = p->getParentOp()) {
      if (isa<func::FuncOp, cudaq::cc::CreateLambdaOp, ModuleOp>(p))
        break;
      if (auto s = dyn_cast<cudaq::cc::ScopeOp>(p))
        parentScopes[scope].insert(s);
    }
  }

  DenseMap<Operation * /*ScopeOp*/, DenseSet<Operation * /*ScopeOp*/>>
      parentScopes;
  DenseMap<Operation * /*AllocaOp*/, Operation * /*ScopeOp*/> scopeMap;

  /// Set of variables to move to function scope.
  SetVector<Operation * /*AllocaOp*/> varsToMove;

  /// Maps a variable to another variable it should coalesce with.
  DenseMap<Operation * /*AllocaOp*/, Operation * /*AllocaOp*/> coalesceMap;

  /// Maps a coalesce leader to the new variable at function scope. This map
  /// will be updated by the rewriter.
  DenseMap<Operation * /*AllocaOp*/, Operation * /*AllocaOp*/> bindingMap;

  bool hoistOnly;
};

class PackingPattern : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  explicit PackingPattern(MLIRContext *ctx, AllocationAnalysis &analysis)
      : OpRewritePattern(ctx), analysis(analysis) {}

  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {
    if (!analysis.toBeMoved(alloca))
      return failure();
    auto *coalesceTo = analysis.getCoalescedTo(alloca);
    auto *b = analysis.getBinding(coalesceTo);
    if (!b)
      return failure();
    auto binding = cast<cudaq::cc::AllocaOp>(b);
    if (coalesceTo == alloca.getOperation()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "allocation " << coalesceTo << " moved to " << b << '\n');
    } else {
      LLVM_DEBUG(llvm::dbgs() << "allocation " << alloca.getOperation()
                              << " was coalesced with " << coalesceTo
                              << " which already has " << b << '\n');
    }
    rewriter.replaceOp(alloca, binding.getResult());
    return success();
  }

private:
  AllocationAnalysis &analysis;
};

class VariableCoalescePass
    : public cudaq::opt::impl::VariableCoalesceBase<VariableCoalescePass> {
public:
  using VariableCoalesceBase::VariableCoalesceBase;

  void runOnOperation() override {
    auto func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before variable coalescing:\n"
                            << func << "\n\n");
    auto *ctx = &getContext();
    AllocationAnalysis analysis(func.getOperation(), hoistOnly);

    if (analysis.nothingToDo())
      return;

    // Step 1: Introduce new variables.
    OpBuilder rewriter(ctx);
    for (auto *o : analysis.getVarsToMove()) {
      auto *coalesceTo = analysis.getCoalescedTo(o);
      if (coalesceTo == o) {
        // This `o` is a leader. Go ahead and create the new alloca and record
        // the binding.

        // Try lambda first since it may be contained by a FuncOp.
        if (auto lamb = o->getParentOfType<cudaq::cc::CreateLambdaOp>()) {
          rewriter.setInsertionPointToStart(&lamb.getBody().front());
        } else {
          // Typical path where this is a kernel function.
          auto func = o->getParentOfType<func::FuncOp>();
          if (!func)
            return;
          rewriter.setInsertionPointToStart(&func.front());
        }
        auto loc = o->getLoc();
        auto ty = cast<cudaq::cc::AllocaOp>(o).getElementType();
        auto newVar = rewriter.create<cudaq::cc::AllocaOp>(loc, ty);
        analysis.addBinding(o, newVar);
      }
    }

    // Step 2: Replace old variables with new ones.
    RewritePatternSet patterns(ctx);
    patterns.insert<PackingPattern>(ctx, analysis);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After variable coalescing:\n"
                            << func << "\n\n");
  }
};
} // namespace
