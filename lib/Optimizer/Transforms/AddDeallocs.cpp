/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "add-deallocs"

using namespace mlir;

namespace {
// Map from quake.alloca -> bool. `true` means there is a deallocation of the
// alloca already present in the function.
using DeallocationMap = llvm::DenseMap<Operation *, bool>;
using RegionOpSet = llvm::DenseSet<Operation *>;

struct DeallocationAnalysisInfo {
  DeallocationAnalysisInfo() = default;
  DeallocationAnalysisInfo(DeallocationMap m, RegionOpSet p)
      : allocMap(m), parents(p) {}

  bool empty() const { return parents.empty(); }

  // \p op has a Region with a quake.alloca. The alloca may or may not already
  // be deallocated.
  bool containsAlloca(Operation *op) const { return parents.count(op); }

  bool needsDeallocations(Operation *op) const {
    if (!containsAlloca(op))
      return false;
    DeallocationMap deallocMap;
    // Do not use walk() here. We do not want to descend into other regions.
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &op : block) {
          if (auto alloca = dyn_cast<quake::AllocaOp>(op)) {
            if (!deallocMap.count(&op))
              deallocMap.insert(std::make_pair(&op, false));
          } else if (auto dealloc = dyn_cast<quake::DeallocOp>(op)) {
            auto val = dealloc.getReference();
            Operation *alloc = cast<quake::AllocaOp>(val.getDefiningOp());
            if (deallocMap.count(alloc))
              deallocMap[alloc] = true;
            else
              deallocMap.insert(std::make_pair(alloc, true));
          }
        }
    for (auto &[_, dealloced] : deallocMap)
      if (!dealloced)
        return true;
    // All alloca/dealloc pairs have been added.
    return false;
  }

  DeallocationMap allocMap;
  RegionOpSet parents;
};

class DeallocationAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeallocationAnalysis)

  DeallocationAnalysis(func::FuncOp op) { performAnalysis(op.getOperation()); }

  bool hasFailed() const { return hasErrors; }

  DeallocationAnalysisInfo getAnalysisInfo() const {
    if (hasFailed())
      return {};
    return {allocMap, parents};
  }

private:
  // Perform the analysis on \p func.
  void performAnalysis(Operation *func) {
    func->walk([this](Operation *o) {
      if (isa<cudaq::cc::UnwindBreakOp, cudaq::cc::UnwindContinueOp,
              cudaq::cc::UnwindReturnOp>(o)) {
        o->emitError("must run unwind-lowering before quake-add-deallocs.");
        hasErrors = true;
      } else if (auto alloc = dyn_cast<quake::AllocaOp>(o)) {
        auto *op = alloc.getOperation();
        if (!allocMap.count(op)) {
          allocMap.insert(std::make_pair(op, /*deallocated=*/false));
          parents.insert(op->getParentOp());
          LLVM_DEBUG(llvm::dbgs() << "adding alloca: " << op << " from "
                                  << op->getParentOp() << '\n');
        }
      } else if (auto dealloc = dyn_cast<quake::DeallocOp>(o)) {
        auto val = dealloc.getReference();
        if (auto alloc = val.getDefiningOp<quake::AllocaOp>()) {
          auto *op = alloc.getOperation();
          if (allocMap.count(op))
            allocMap[op] = true;
          else
            allocMap.insert(std::make_pair(op, /*deallocated=*/true));
          LLVM_DEBUG(llvm::dbgs() << "found dealloc of alloca: " << op << '\n');
        } else {
          dealloc->emitError("unable to determine associated allocation.");
          hasErrors = true;
        }
      }
    });
  }

  DeallocationMap allocMap;
  RegionOpSet parents;
  bool hasErrors = false;
};

inline void generateDeallocsForSet(PatternRewriter &rewriter,
                                   llvm::DenseSet<Operation *> &allocSet) {
  for (Operation *a : allocSet)
    rewriter.create<quake::DeallocOp>(a->getLoc(), cast<quake::AllocaOp>(a));
}

// The different rewrite cases involve the same work, but use different types.
template <typename RET, typename OP>
LogicalResult addDeallocations(OP wrapper, PatternRewriter &rewriter,
                               const DeallocationAnalysisInfo &infoMap,
                               const DominanceInfo &domInfo) {
  rewriter.startRootUpdate(wrapper);
  llvm::DenseSet<Operation *> allocs;
  for (auto &[op, done] : infoMap.allocMap)
    if ((op->getParentOp() == wrapper.getOperation()) && !done)
      allocs.insert(op);
  if (allocs.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "no deallocations to add\n");
    return success();
  }

  // Allocs contains alloca operations to deallocate.
  LLVM_DEBUG(llvm::dbgs() << "adding deallocations to "
                          << wrapper.getOperation() << '\n');

  // 1) Create an exit block to stick dealloc operations in.
  auto *exitBlock = new Block;
  exitBlock->addArguments(
      wrapper.getResultTypes(),
      SmallVector<Location>{wrapper.getNumResults(), wrapper.getLoc()});
  wrapper.getRegion().push_back(exitBlock);

  // 2) Update all the RET ops (at top level) to branches to the exit block
  // when it is correct to do so. Otherwise, add the subset of deallocations
  // inline before each RET op.
  auto entireSetDominates = [&](RET ret) {
    for (auto *alloc : allocs)
      if (!domInfo.dominates(alloc, ret))
        return false;
    return true;
  };
  for (Block &block : wrapper.getRegion())
    for (Operation &op : block)
      if (auto ret = dyn_cast<RET>(op)) {
        if (entireSetDominates(ret)) {
          // Replace the RET op with a branch to the shared deallocation block.
          rewriter.setInsertionPoint(ret);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(ret, exitBlock,
                                                    ret.getOperands());
        } else {
          // Collect only the subset that dominates this RET op. Insert the
          // deallocations directly in front of the RET op.
          llvm::DenseSet<Operation *> subset;
          for (auto *alloc : allocs)
            if (domInfo.dominates(alloc, ret))
              subset.insert(alloc);
          rewriter.setInsertionPoint(ret);
          generateDeallocsForSet(rewriter, subset);
        }
      }

  // 3) Create the deallocations.
  rewriter.setInsertionPointToEnd(exitBlock);
  generateDeallocsForSet(rewriter, allocs);
  rewriter.create<RET>(wrapper.getLoc(), exitBlock->getArguments());

  rewriter.finalizeRootUpdate(wrapper);
  LLVM_DEBUG(llvm::dbgs() << "updated " << wrapper.getOperation() << '\n');
  return success();
}

template <typename A, typename B>
struct DeallocPattern : public OpRewritePattern<A> {
  using Base = OpRewritePattern<A>;

  explicit DeallocPattern(MLIRContext *ctx,
                          const DeallocationAnalysisInfo &info,
                          const DominanceInfo &dom)
      : Base(ctx), infoMap(info), domInfo(dom) {}

  LogicalResult matchAndRewrite(A op,
                                PatternRewriter &rewriter) const override {
    return addDeallocations<B>(op, rewriter, infoMap, domInfo);
  }

  const DeallocationAnalysisInfo &infoMap;
  const DominanceInfo &domInfo;
};

using FuncDeallocPattern = DeallocPattern<func::FuncOp, func::ReturnOp>;
using LambdaDeallocPattern =
    DeallocPattern<cudaq::cc::CreateLambdaOp, cudaq::cc::ReturnOp>;
using ScopeDeallocPattern =
    DeallocPattern<cudaq::cc::ScopeOp, cudaq::cc::ContinueOp>;

/// This pass adds quake.dealloc operations to functions and λ expressions to
/// deallocate any quantum objects as allocated with quake.alloca operations.
/// Unlike classical objects that are stack allocated, quantum objects must be
/// explicitly deallocated in final code generation, etc.
///
/// It is the responsibility of this pass to add the deallocations in cases
/// where there is only high-level structured control-flow present.
/// Specifically, a operations of type func.func, cc.scope, and cc.create_lambda
/// may have quake.alloca operations which are not paired with quake.dealloc
/// operations.
///
/// This pass should be run <em>after</em> the UnwindLowering pass, which adds
/// dealloc ops along non-trivial control paths in the presence of global jumps.
/// DeallocationAnalysis will flag any unwinding jumps as errors.
class QuakeAddDeallocsPass
    : public cudaq::opt::QuakeAddDeallocsBase<QuakeAddDeallocsPass> {
public:
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!funcOp || funcOp.empty())
      return;

    DeallocationAnalysis analysis(funcOp);
    if (analysis.hasFailed()) {
      funcOp.emitError("error adding deallocations\n");
      signalPassFailure();
      return;
    }

    // Analysis was successful, so add the deallocations as needed.
    auto allocInfo = analysis.getAnalysisInfo();
    if (allocInfo.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "no deallocs to add.\n");
      return;
    }
    auto *ctx = funcOp.getContext();
    DominanceInfo dom(funcOp);
    RewritePatternSet patterns(ctx);
    patterns
        .insert<FuncDeallocPattern, ScopeDeallocPattern, LambdaDeallocPattern>(
            ctx, allocInfo, dom);
    ConversionTarget target(*ctx);
    target.addDynamicallyLegalOp<func::FuncOp, cudaq::cc::ScopeOp,
                                 cudaq::cc::CreateLambdaOp>(
        [&](Operation *op) { return !allocInfo.needsDeallocations(op); });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      funcOp.emitError("error adding deallocations\n");
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQuakeAddDeallocs() {
  return std::make_unique<QuakeAddDeallocsPass>();
}
