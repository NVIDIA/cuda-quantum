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
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_COMBINEQUANTUMALLOCATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "combine-quantum-alloc"

using namespace mlir;

static Value createCast(PatternRewriter &rewriter, Location loc, Value inVal) {
  auto i64Ty = rewriter.getI64Type();
  if (inVal.getType() == rewriter.getIndexType())
    return rewriter.create<arith::IndexCastOp>(loc, i64Ty, inVal);
  return rewriter.create<cudaq::cc::CastOp>(loc, i64Ty, inVal,
                                            cudaq::cc::CastOpMode::Unsigned);
}

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
    Type refTy = quake::RefType::get(rewriter.getContext());
    for (auto p : llvm::enumerate(analysis.allocations)) {
      if (alloc == p.value()) {
        auto i = p.index();
        auto &os = analysis.offsetSizes[i];
        if (alloc.getType() == refTy) {
          [[maybe_unused]] Value ext =
              rewriter.replaceOpWithNewOp<quake::ExtractRefOp>(
                  alloc, analysis.newAlloc, os.first);
          LLVM_DEBUG(llvm::dbgs()
                     << "replace " << alloc << " with " << ext << '\n');
          return success();
        }
        Value lo = rewriter.create<arith::ConstantIntOp>(
            alloc.getLoc(), os.first, rewriter.getI64Type());
        Value hi = rewriter.create<arith::ConstantIntOp>(
            alloc.getLoc(), os.first + os.second - 1, rewriter.getI64Type());
        [[maybe_unused]] Value subveq =
            rewriter.replaceOpWithNewOp<quake::SubVeqOp>(
                alloc, alloc.getType(), analysis.newAlloc, lo, hi);
        LLVM_DEBUG(llvm::dbgs()
                   << "replace " << alloc << " with " << subveq << '\n');
        return success();
      }
    }
    return failure();
  }

  Analysis &analysis;
};

class ExtractPat : public OpRewritePattern<quake::ExtractRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Replace a pattern such as:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %2 = quake.subveq %1, %c2, %c3 : (!quake.veq<4>, i32, i32) ->
  //        !quake.veq<2>
  //   %3 = quake.extract_ref %2[0] : (!quake.veq<2>) -> !quake.ref
  // ```
  // with:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %3 = quake.extract_ref %1[2] : (!uwake.veq<4>) -> !quake.ref
  // ```
  LogicalResult matchAndRewrite(quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    auto subveq = extract.getVeq().getDefiningOp<quake::SubVeqOp>();
    if (!subveq || isa<quake::SubVeqOp>(subveq.getVeq().getDefiningOp()))
      return failure();

    Value offset;
    auto loc = extract.getLoc();
    Value low = subveq.getLow();
    if (extract.hasConstantIndex()) {
      Value cv = rewriter.create<arith::ConstantIntOp>(
          loc, extract.getConstantIndex(), low.getType());
      offset = rewriter.create<arith::AddIOp>(loc, cv, low);
    } else {
      Value cast1 = createCast(rewriter, loc, extract.getIndex());
      Value cast2 = createCast(rewriter, loc, low);
      offset = rewriter.create<arith::AddIOp>(loc, cast1, cast2);
    }
    rewriter.replaceOpWithNewOp<quake::ExtractRefOp>(extract, subveq.getVeq(),
                                                     offset);
    return success();
  }
};

class SubVeqPat : public OpRewritePattern<quake::SubVeqOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::SubVeqOp subveq,
                                PatternRewriter &rewriter) const override {
    auto prior = subveq.getVeq().getDefiningOp<quake::SubVeqOp>();
    if (!prior)
      return failure();

    auto loc = subveq.getLoc();
    Value cast1 = createCast(rewriter, loc, prior.getLow());
    Value cast2 = createCast(rewriter, loc, subveq.getLow());
    Value cast3 = createCast(rewriter, loc, subveq.getHigh());
    Value sum1 = rewriter.create<arith::AddIOp>(loc, cast1, cast2);
    Value sum2 = rewriter.create<arith::AddIOp>(loc, cast1, cast3);
    auto veqTy = subveq.getType();
    rewriter.replaceOpWithNewOp<quake::SubVeqOp>(subveq, veqTy, prior.getVeq(),
                                                 sum1, sum2);
    return success();
  }
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
          if (alloc.getSize())
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
      patterns.insert<ExtractPat, SubVeqPat>(ctx);
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
    if (isa<quake::RefType>(alloc.getType()))
      return 1;
    auto veq = cast<quake::VeqType>(alloc.getType());
    assert(veq.hasSpecifiedSize() && "veq type must have constant size");
    return veq.getSize();
  }
};
} // namespace
