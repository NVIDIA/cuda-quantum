/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_COMBINEQUANTUMALLOCATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "combine-quantum-alloc"

using namespace mlir;

namespace {
struct Analysis {
  Analysis() = default;
  Analysis(const Analysis &) = delete;
  Analysis(Analysis &&) = delete;
  Analysis &operator=(const Analysis &) = delete;

  SmallVector<cudaq::quake::AllocaOp> allocations;
  SmallVector<std::pair<std::size_t, std::size_t>> offsetSizes;
  SmallVector<cudaq::quake::DeallocOp> deallocs;
  cudaq::quake::AllocaOp newAlloc;

  bool empty() const { return allocations.empty(); }
};

class AllocaPat : public OpRewritePattern<cudaq::quake::AllocaOp> {
public:
  explicit AllocaPat(MLIRContext *ctx, Analysis &a)
      : OpRewritePattern(ctx), analysis(a) {}

  LogicalResult matchAndRewrite(cudaq::quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    for (auto p : llvm::enumerate(analysis.allocations)) {
      if (alloc == p.value()) {
        auto i = p.index();
        auto &os = analysis.offsetSizes[i];
        if (isa<cudaq::quake::RefType>(alloc.getType())) {
          [[maybe_unused]] Value ext =
              rewriter.replaceOpWithNewOp<cudaq::quake::ExtractRefOp>(
                  alloc, analysis.newAlloc, os.first);
          LLVM_DEBUG(llvm::dbgs()
                     << "replace " << alloc << " with " << ext << '\n');
          return success();
        }
        if (isa<cudaq::quake::VeqType>(alloc.getType())) {
          Value lo = arith::ConstantIntOp::create(
              rewriter, alloc.getLoc(), rewriter.getI64Type(), os.first);
          Value hi = arith::ConstantIntOp::create(rewriter, alloc.getLoc(),
                                                  rewriter.getI64Type(),
                                                  os.first + os.second - 1);
          // trying to print alloc after the replace gives a segfault
          LLVM_DEBUG(llvm::dbgs() << "replace " << alloc);
          [[maybe_unused]] Value subveq =
              rewriter.replaceOpWithNewOp<cudaq::quake::SubVeqOp>(
                  alloc, alloc.getType(), analysis.newAlloc, lo, hi);
          LLVM_DEBUG(llvm::dbgs() << " with " << subveq << '\n');
          return success();
        }
        if (auto sty = dyn_cast<cudaq::quake::StruqType>(alloc.getType())) {
          SmallVector<Value> parts;
          std::size_t inner = os.first;
          auto loc = alloc.getLoc();
          for (auto m : sty.getMembers()) {
            auto v = [&]() -> Value {
              if (isa<cudaq::quake::RefType>(m)) {
                auto result = cudaq::quake::ExtractRefOp::create(
                    rewriter, loc, analysis.newAlloc, inner);
                inner++;
                return result;
              }
              assert(cast<cudaq::quake::VeqType>(m).hasSpecifiedSize());
              std::size_t dist =
                  inner + cast<cudaq::quake::VeqType>(m).getSize() - 1;
              auto result = cudaq::quake::SubVeqOp::create(
                  rewriter, loc, m, analysis.newAlloc, inner, dist);
              inner = dist + 1;
              return result;
            }();
            parts.push_back(v);
          }
          rewriter.replaceOpWithNewOp<cudaq::quake::MakeStruqOp>(alloc, sty,
                                                                 parts);
          return success();
        }
        return alloc.emitOpError("has unexpected type");
      }
    }
    return failure();
  }

  Analysis &analysis;
};

class CombineQuantumAllocationsPass
    : public cudaq::opt::impl::CombineQuantumAllocationsBase<
          CombineQuantumAllocationsPass> {
public:
  using CombineQuantumAllocationsBase::CombineQuantumAllocationsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Function before combining quake alloca:\n"
                            << func << "\n\n");

    if (func->hasAttr(cudaq::opt::disableQubitCombineAttrName)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Combining quake alloca is disabled for " << func.getName()
                 << " as it contains scoped qubits.\n");
      return;
    }

    [[maybe_unused]] bool unused = combineAllocationsInRegion(func.getRegion());

    LLVM_DEBUG(llvm::dbgs() << "Function after combining quake alloca:\n"
                            << func << "\n\n");
  }

  /// Combine the `quake.alloca` ops that appear directly in the top level of
  /// \p region (i.e. not nested inside a further `cc.scope` /
  /// `cc.create_lambda`, which is handled independently below) into a single
  /// `quake.alloca` at the top of \p region. \p region may be a func.func's own
  /// body, or the body of a `cc.scope` / `cc.create_lambda`.
  ///
  /// A `cc.scope`'s own body is combined independently of its enclosing
  /// region, with the merged `alloca` placed at the top of \e that body, rather
  /// than hoisted out. If the scope executes more than once (i.e. nested in
  /// a loop), each execution still produces its own fresh merged `alloca`,
  /// exactly matching the un-combined semantics — this is a \e CORRECTNESS
  /// constraint that cannot be whimsically ignored and why lowering `cc.scope`
  /// to CFG form sets `disableQubitCombineAttrName` when it contains quantum
  /// allocations. A `cc.create_lambda`'s body is likewise combined
  /// independently: it is an entirely separate callable activation that may be
  /// invoked any number of times, independent of the enclosing function.
  bool combineAllocationsInRegion(Region &region) {
    if (region.empty())
      return true;

    // Recurse first (innermost first) into every cc.scope / cc.create_lambda
    // found directly in this region's own top level. Recursion handled
    // naturally here, so go no further.
    for (auto &block : region)
      for (auto &op : block)
        if (isa<cudaq::cc::ScopeOp, cudaq::cc::CreateLambdaOp>(op))
          for (auto &nested : op.getRegions()) {
            bool ok = combineAllocationsInRegion(nested);
            if (!ok)
              return ok;
          }

    // 1. Scan the top level of \p region for all `alloca` operations.
    // FIXME: other passes rely on this pass to cleanup broken IR. Preserve the
    // bugs here for now.
    // Eventually, skip those that are parametric or married to an initialize
    // with state: they are left exactly as they are, so their own dealloc must
    // be left alone too -- only a dealloc of an alloca actually being combined
    // belongs in analysis.deallocs, or step 4 below would erase the dealloc of
    // a skipped (untouched) alloca along with the ones actually being merged,
    // silently leaving it never deallocated.
    Analysis analysis;
    std::size_t currentOffset = 0;
    for (auto &block : region)
      for (auto &op : block) {
        if (auto alloc = dyn_cast_or_null<cudaq::quake::AllocaOp>(&op)) {
          if (alloc.getSize() || alloc.hasInitializedState())
            return false;
          auto size = cudaq::quake::getAllocationSize(alloc.getType());
          if (size == 0) {
            // Skip zero-size allocas. Merging them would produce
            // subveq(lo, lo-1) which is invalid.
            continue;
          }
          analysis.allocations.push_back(alloc);
          analysis.offsetSizes.emplace_back(currentOffset, size);
          currentOffset += size;
        } else if (auto dealloc =
                       dyn_cast_or_null<cudaq::quake::DeallocOp>(&op)) {
          analysis.deallocs.push_back(dealloc);
        }
      }

    if (analysis.empty())
      return true;

    // 2. Combine all the allocas into a single alloca at the top of the region.
    auto *entryBlock = &region.front();
    auto *ctx = &getContext();
    auto loc = analysis.allocations.front().getLoc();
    OpBuilder rewriter(ctx);
    rewriter.setInsertionPointToStart(entryBlock);
    auto veqTy = cudaq::quake::VeqType::get(ctx, currentOffset);
    analysis.newAlloc = cudaq::quake::AllocaOp::create(rewriter, loc, veqTy);

    // 3. Greedily replace the uses of the original alloca ops with uses of
    // partitions of the new alloca op. Replace subveq of subveq with a single
    // new subveq. Replace extract from subveq with extract from original
    // veq. AllocaPat only matches ops literally in analysis.allocations (this
    // call's own, region-scoped list), so it is harmless to always root the
    // driver at the pass's own top-level function rather than region's
    // owning op: a cc.scope/cc.create_lambda is not IsolatedFromAbove (it may
    // reference values from its enclosing region), so applyPatternsGreedily
    // cannot be rooted there directly, but the func::FuncOp always is and the
    // driver already walks every nested region regardless of where it's
    // rooted.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<AllocaPat>(ctx, analysis);
      cudaq::quake::ExtractRefOp::getCanonicalizationPatterns(patterns, ctx);
      cudaq::quake::GetMemberOp::getCanonicalizationPatterns(patterns, ctx);
      cudaq::quake::SubVeqOp::getCanonicalizationPatterns(patterns, ctx);
      cudaq::quake::ConcatOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
        region.getParentOp()->emitOpError(
            "combining alloca, subveq, and extract ops failed");
        signalPassFailure();
      }
    }

    // 4. Remove the deallocations, if any. Add new dealloc to exits.
    // FIXME: This unintentionally "fixes" broken IR by potentially removing
    // any number of bogus deallocs on the same alloca. There are other passes
    // that rely on this behavior to repair the IR behind themselves.
    if (!analysis.deallocs.empty()) {
      for (auto d : analysis.deallocs)
        d.erase();
      for (auto &block : region) {
        if (block.hasNoSuccessors()) {
          rewriter.setInsertionPoint(block.getTerminator());
          cudaq::quake::DeallocOp::create(rewriter, analysis.newAlloc.getLoc(),
                                          analysis.newAlloc);
        }
      }
    }

    return true;
  }
};
} // namespace
