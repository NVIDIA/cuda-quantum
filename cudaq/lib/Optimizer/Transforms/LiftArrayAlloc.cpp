/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LIFTARRAYALLOC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lift-array-alloc"

using namespace mlir;

#include "LiftArrayAllocPatterns.inc"

namespace {
static void collectDefiningOps(Value value,
                               llvm::SmallPtrSetImpl<Operation *> &seen,
                               SmallVectorImpl<Operation *> &candidates) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || seen.contains(definingOp))
    return;

  SmallVector<std::pair<Operation *, bool>, 16> worklist;
  worklist.emplace_back(definingOp, false);
  while (!worklist.empty()) {
    auto [op, expanded] = worklist.pop_back_val();
    if (expanded) {
      candidates.push_back(op);
      continue;
    }
    if (!seen.insert(op).second)
      continue;
    worklist.emplace_back(op, true);
    for (Value operand : llvm::reverse(op->getOperands()))
      if (Operation *producer = operand.getDefiningOp())
        if (!seen.contains(producer))
          worklist.emplace_back(producer, false);
  }
}

static SmallVector<Operation *>
collectAllocationCandidates(func::FuncOp func, bool includeMemoryOps) {
  SmallVector<Operation *> candidates;
  llvm::SmallPtrSet<Operation *, 16> seen;
  auto addCandidate = [&](Operation *op) {
    if (seen.insert(op).second)
      candidates.push_back(op);
  };

  func.walk([&](cudaq::cc::AllocaOp alloc) {
    for (Operation *pointerOp : alloc->getUsers()) {
      if (!isa<cudaq::cc::CastOp, cudaq::cc::ComputePtrOp>(pointerOp))
        continue;
      if (includeMemoryOps)
        addCandidate(pointerOp);
      for (Operation *user : pointerOp->getUsers()) {
        auto store = dyn_cast<cudaq::cc::StoreOp>(user);
        if (!store)
          continue;
        collectDefiningOps(store.getValue(), seen, candidates);
        if (includeMemoryOps)
          addCandidate(store);
      }
    }
    if (includeMemoryOps)
      addCandidate(alloc);
  });
  return candidates;
}

class LiftArrayAllocPass
    : public cudaq::opt::impl::LiftArrayAllocBase<LiftArrayAllocPass> {
public:
  using LiftArrayAllocBase::LiftArrayAllocBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    SmallVector<Operation *> initializerOps =
        collectAllocationCandidates(func, false);
    SmallVector<Operation *> candidates =
        collectAllocationCandidates(func, true);
    if (candidates.empty())
      return;

    GreedyRewriteConfig config;
    config.setScope(&func.getBody())
        .setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);

    // The region driver folded initializer expressions before retrying the
    // allocation pattern. Preserve that phase ordering without scanning ops
    // unrelated to an allocation.
    if (!initializerOps.empty()) {
      RewritePatternSet foldingPatterns(ctx);
      if (failed(applyOpPatternsGreedily(initializerOps,
                                         std::move(foldingPatterns), config))) {
        signalPassFailure();
        return;
      }
      candidates = collectAllocationCandidates(func, true);
    }

    DominanceInfo domInfo(func);
    StringRef funcName = func.getName();
    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaPattern>(ctx, domInfo, funcName);

    LLVM_DEBUG(llvm::dbgs()
               << "Before lifting constant array: " << func << '\n');

    if (failed(
            applyOpPatternsGreedily(candidates, std::move(patterns), config)))
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "After lifting constant array: " << func << '\n');
  }
};
} // namespace
