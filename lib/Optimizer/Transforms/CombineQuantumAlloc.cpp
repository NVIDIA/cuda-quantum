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
        if (os.second == 1) {
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
        [[maybe_unused]] Value subvec =
            rewriter.replaceOpWithNewOp<quake::SubVecOp>(
                alloc, alloc.getType(), analysis.newAlloc, lo, hi);
        LLVM_DEBUG(llvm::dbgs()
                   << "replace " << alloc << " with " << subvec << '\n');
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
  //   %2 = quake.subvec %1, %c2, %c3 : (!quake.veq<4>, i32, i32) ->
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
    auto subvec = extract.getVeq().getDefiningOp<quake::SubVecOp>();
    if (!subvec || isa<quake::SubVecOp>(subvec.getVeq().getDefiningOp()))
      return failure();

    Value offset;
    auto loc = extract.getLoc();
    Value low = subvec.getLow();
    if (extract.hasConstantIndex()) {
      Value cv = rewriter.create<arith::ConstantIntOp>(
          loc, extract.getConstantIndex(), low.getType());
      offset = rewriter.create<arith::AddIOp>(loc, cv, low);
    } else {
      Value cast1 = rewriter.create<cudaq::cc::CastOp>(
          loc, rewriter.getI64Type(), extract.getIndex(),
          cudaq::cc::CastOpMode::Unsigned);
      Value cast2 = rewriter.create<cudaq::cc::CastOp>(
          loc, rewriter.getI64Type(), low, cudaq::cc::CastOpMode::Unsigned);
      offset = rewriter.create<arith::AddIOp>(loc, cast1, cast2);
    }
    rewriter.replaceOpWithNewOp<quake::ExtractRefOp>(extract, subvec.getVeq(),
                                                     offset);
    return success();
  }
};

class SubVecPat : public OpRewritePattern<quake::SubVecOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::SubVecOp subvec,
                                PatternRewriter &rewriter) const override {
    auto prior = subvec.getVeq().getDefiningOp<quake::SubVecOp>();
    if (!prior)
      return failure();

    auto loc = subvec.getLoc();
    Value cast1 = rewriter.create<cudaq::cc::CastOp>(
        loc, rewriter.getI64Type(), prior.getLow(),
        cudaq::cc::CastOpMode::Unsigned);
    Value cast2 = rewriter.create<cudaq::cc::CastOp>(
        loc, rewriter.getI64Type(), subvec.getLow(),
        cudaq::cc::CastOpMode::Unsigned);
    Value cast3 = rewriter.create<cudaq::cc::CastOp>(
        loc, rewriter.getI64Type(), subvec.getHigh(),
        cudaq::cc::CastOpMode::Unsigned);
    Value sum1 = rewriter.create<arith::AddIOp>(loc, cast1, cast2);
    Value sum2 = rewriter.create<arith::AddIOp>(loc, cast1, cast3);
    auto veqTy = subvec.getType();
    rewriter.replaceOpWithNewOp<quake::SubVecOp>(subvec, veqTy, prior.getVeq(),
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
    // partitions of the new alloca op. Replace subvec of subvec with a single
    // new subvec. Replace extract from subvec with extract from original
    // veq.
    {
      RewritePatternSet patterns(ctx);
      patterns.insert<AllocaPat>(ctx, analysis);
      patterns.insert<ExtractPat, SubVecPat>(ctx);
      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns)))) {
        func.emitOpError("combining alloca, subvec, and extract ops failed");
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
