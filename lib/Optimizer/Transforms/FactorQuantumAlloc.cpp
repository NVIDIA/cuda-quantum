/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_FACTORQUANTUMALLOCATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "factor-quantum-alloc"

using namespace mlir;

namespace {
class AllocaPat : public OpRewritePattern<quake::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// If we are here, then all uses of \p allocOp are either an ExtractRefOp
  /// with a constant index or a DeallocOp. Any other user is assumed to block
  /// the factoring of the allocation.
  LogicalResult matchAndRewrite(quake::AllocaOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto veqTy = cast<quake::VeqType>(allocOp.getType());
    auto loc = allocOp.getLoc();
    std::size_t size = veqTy.getSize();
    SmallVector<quake::AllocaOp> newAllocs;
    auto *ctx = rewriter.getContext();
    auto refTy = quake::RefType::get(ctx);

    // 1. Split the aggregate veq into a sequence of distinct alloca of ref.
    for (std::size_t i = 0; i < size; ++i)
      newAllocs.emplace_back(rewriter.create<quake::AllocaOp>(loc, refTy));

    // 2. Visit all users and replace them accordingly.
    for (auto *user : allocOp->getUsers()) {
      if (auto dealloc = dyn_cast<quake::DeallocOp>(user)) {
        rewriter.setInsertionPoint(dealloc);
        auto deloc = dealloc.getLoc();
        for (std::size_t i = 0; i < size - 1; ++i)
          rewriter.create<quake::DeallocOp>(deloc, newAllocs[i]);
        rewriter.replaceOpWithNewOp<quake::DeallocOp>(dealloc,
                                                      newAllocs[size - 1]);
	continue;
      }
      auto ext = cast<quake::ExtractRefOp>(user);
      auto index = ext.getConstantIndex();
      rewriter.replaceOp(ext, newAllocs[index].getResult());
    }

    // 3. Remove the original alloca operation.
    rewriter.eraseOp(allocOp);

    return success();
  }
};

class FactorQuantumAllocationsPass
    : public cudaq::opt::impl::FactorQuantumAllocationsBase<
          FactorQuantumAllocationsPass> {
public:
  using FactorQuantumAllocationsBase::FactorQuantumAllocationsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Function before factoring quake alloca:\n"
                            << func << "\n\n");
    SmallVector<quake::AllocaOp> allocations;
    func.walk([&](quake::AllocaOp alloc) {
      if (!allocaOfVeq(alloc) || allocaOfUnspecifiedSize(alloc))
        return;
      bool usesAreConvertible = [&]() {
        for (auto *users : alloc->getUsers()) {
          if (isa<quake::DeallocOp>(users))
            continue;
          if (auto ext = dyn_cast<quake::ExtractRefOp>(users))
            if (ext.hasConstantIndex())
              continue;
          return false;
        }
        return true;
      }();
      if (usesAreConvertible)
        allocations.push_back(alloc);
    });

    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaPat>(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<quake::QuakeDialect>();
    target.addDynamicallyLegalOp<quake::AllocaOp>([&](quake::AllocaOp alloc) {
      return std::find(allocations.begin(), allocations.end(), alloc) ==
             allocations.end();
    });
    if (failed(applyPartialConversion(func.getOperation(), target,
                                      std::move(patterns)))) {
      func.emitOpError("factoring quantum allocations failed");
      signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Function after factoring quake alloca:\n"
                            << func << "\n\n");
  }

  static bool allocaOfVeq(quake::AllocaOp alloc) {
    return isa<quake::VeqType>(alloc.getType());
  }

  static bool allocaOfUnspecifiedSize(quake::AllocaOp alloc) {
    if (auto veqTy = dyn_cast<quake::VeqType>(alloc.getType()))
      return !veqTy.hasSpecifiedSize();
    return false;
  }
};
} // namespace
