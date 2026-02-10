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
#define GEN_PASS_DEF_DEADSTOREREMOVAL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "dsr"

using namespace mlir;

namespace {

class DSRPattern : public OpRewritePattern<cudaq::cc::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // If we have a cc.alloca and all of its uses are cc.store ops, then this is a
  // dead store.
  LogicalResult matchAndRewrite(cudaq::cc::StoreOp store,
                                PatternRewriter &rewriter) const override {
    cudaq::cc::AllocaOp alloca;
    if (auto getPtr =
            store.getPtrvalue().getDefiningOp<cudaq::cc::ComputePtrOp>()) {
      if (getPtr.getNumOperands() != 1)
        return failure();
      alloca = getPtr.getBase().getDefiningOp<cudaq::cc::AllocaOp>();
    } else if (auto getPtr =
                   store.getPtrvalue().getDefiningOp<cudaq::cc::CastOp>()) {
      alloca = getPtr.getValue().getDefiningOp<cudaq::cc::AllocaOp>();
    } else {
      alloca = store.getPtrvalue().getDefiningOp<cudaq::cc::AllocaOp>();
    }
    if (!alloca) {
      LLVM_DEBUG(llvm::dbgs() << "store not to alloca.\n");
      return failure();
    }

    auto testAllStoreUsers = [&](Operation *c) {
      for (auto v : c->getUsers()) {
        if (auto s = dyn_cast<cudaq::cc::StoreOp>(v)) {
          // Make sure this stores *to* the address rather stores the address.
          if (s.getPtrvalue() == c->getResult(0))
            continue;
        }
        return false;
      }
      return true;
    };

    for (auto u : alloca->getUsers()) {
      if (auto c = dyn_cast<cudaq::cc::CastOp>(u)) {
        if (!testAllStoreUsers(c)) {
          LLVM_DEBUG(llvm::dbgs() << "store not from cast of alloca.\n");
          return failure();
        }
        continue;
      }
      if (auto c = dyn_cast<cudaq::cc::ComputePtrOp>(u)) {
        if (!testAllStoreUsers(c)) {
          LLVM_DEBUG(llvm::dbgs() << "store not from compute_ptr of alloca.\n");
          return failure();
        }
        continue;
      }

      if (auto s = dyn_cast<cudaq::cc::StoreOp>(u))
        if (s.getPtrvalue() == alloca.getResult())
          continue;
      LLVM_DEBUG(llvm::dbgs() << "alloca use is not store/cast/compute_ptr.\n");
      return failure();
    }
    rewriter.eraseOp(store);
    return success();
  }
};

class DSRPass : public cudaq::opt::impl::DeadStoreRemovalBase<DSRPass> {
public:
  using DeadStoreRemovalBase::DeadStoreRemovalBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<DSRPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
