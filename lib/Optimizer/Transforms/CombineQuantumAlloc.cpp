/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/Canonical.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
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

  SmallVector<quake::AllocaOp> allocations;
  SmallVector<std::pair<std::size_t, std::size_t>> offsetSizes;
  SmallVector<quake::DeallocOp> deallocs;
  quake::AllocaOp newAlloc;

  bool empty() const { return allocations.empty(); }
};

class AllocaPat : public OpRewritePattern<quake::AllocaOp> {
public:
  explicit AllocaPat(MLIRContext *ctx, Analysis &a)
      : OpRewritePattern(ctx), analysis(a) {}

  LogicalResult matchAndRewrite(quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    for (auto p : llvm::enumerate(analysis.allocations)) {
      if (alloc == p.value()) {
        auto i = p.index();
        auto &os = analysis.offsetSizes[i];
        if (isa<quake::RefType>(alloc.getType())) {
          [[maybe_unused]] Value ext =
              rewriter.replaceOpWithNewOp<quake::ExtractRefOp>(
                  alloc, analysis.newAlloc, os.first);
          LLVM_DEBUG(llvm::dbgs()
                     << "replace " << alloc << " with " << ext << '\n');
          return success();
        }
        if (isa<quake::VeqType>(alloc.getType())) {
          Value lo = rewriter.create<arith::ConstantIntOp>(
              alloc.getLoc(), os.first, rewriter.getI64Type());
          Value hi = rewriter.create<arith::ConstantIntOp>(
              alloc.getLoc(), os.first + os.second - 1, rewriter.getI64Type());
          // trying to print alloc after the replace gives a segfault
          LLVM_DEBUG(llvm::dbgs() << "replace " << alloc);
          [[maybe_unused]] Value subveq =
              rewriter.replaceOpWithNewOp<quake::SubVeqOp>(
                  alloc, alloc.getType(), analysis.newAlloc, lo, hi);
          LLVM_DEBUG(llvm::dbgs() << " with " << subveq << '\n');
          return success();
        }
        if (auto sty = dyn_cast<quake::StruqType>(alloc.getType())) {
          SmallVector<Value> parts;
          std::size_t inner = os.first;
          auto loc = alloc.getLoc();
          for (auto m : sty.getMembers()) {
            auto v = [&]() -> Value {
              if (isa<quake::RefType>(m)) {
                auto result = rewriter.create<quake::ExtractRefOp>(
                    loc, analysis.newAlloc, inner);
                inner++;
                return result;
              }
              assert(cast<quake::VeqType>(m).hasSpecifiedSize());
              std::size_t dist = inner + cast<quake::VeqType>(m).getSize() - 1;
              auto result = rewriter.create<quake::SubVeqOp>(
                  loc, m, analysis.newAlloc, inner, dist);
              inner = dist + 1;
              return result;
            }();
            parts.push_back(v);
          }
          rewriter.replaceOpWithNewOp<quake::MakeStruqOp>(alloc, sty, parts);
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

    // 1. Scan the top-level of the function for all alloca operations. Exit if
    // any of them are parametric.
    Analysis analysis;
    std::size_t currentOffset = 0;
    for (auto &block : func.getRegion())
      for (auto &op : block) {
        if (auto alloc = dyn_cast_or_null<quake::AllocaOp>(&op)) {
          if (alloc.getSize() || alloc.hasInitializedState())
            return;
          analysis.allocations.push_back(alloc);
          auto size = allocationSize(alloc);
          analysis.offsetSizes.emplace_back(currentOffset, size);
          currentOffset += size;
        } else if (auto dealloc = dyn_cast_or_null<quake::DeallocOp>(&op)) {
          analysis.deallocs.push_back(dealloc);
        }
      }
    if (analysis.empty())
      return;

    // 2. Combine all the allocas into a single alloca at the top of the
    // function.
    auto *entryBlock = &func.getRegion().front();
    auto *ctx = &getContext();
    auto loc = analysis.allocations.front().getLoc();
    OpBuilder rewriter(ctx);
    rewriter.setInsertionPointToStart(entryBlock);
    auto veqTy = quake::VeqType::get(ctx, currentOffset);
    analysis.newAlloc = rewriter.create<quake::AllocaOp>(loc, veqTy);

    // 3. Greedily replace the uses of the original alloca ops with uses of
    // partitions of the new alloca op. Replace subveq of subveq with a single
    // new subveq. Replace extract from subveq with extract from original
    // veq.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<AllocaPat>(ctx, analysis);
      patterns.insert<quake::canonical::ExtractRefFromSubVeqPattern,
                      quake::canonical::CombineSubVeqsPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns)))) {
        func.emitOpError("combining alloca, subveq, and extract ops failed");
        signalPassFailure();
      }
    }

    // 4. Remove the deallocations, if any. Add new dealloc to exits.
    if (!analysis.deallocs.empty()) {
      for (auto d : analysis.deallocs)
        d.erase();
      for (auto &block : func.getRegion()) {
        if (block.hasNoSuccessors()) {
          rewriter.setInsertionPoint(block.getTerminator());
          rewriter.create<quake::DeallocOp>(analysis.newAlloc.getLoc(),
                                            analysis.newAlloc);
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Function after combining quake alloca:\n"
                            << func << "\n\n");
  }

  // TODO: move this to a place where it can be shared.
  static std::size_t allocationSize(quake::AllocaOp alloc) {
    return quake::getAllocationSize(alloc.getType());
  }
};
} // namespace
