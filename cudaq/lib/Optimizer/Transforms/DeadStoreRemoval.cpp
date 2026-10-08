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

#define DEBUG_TYPE "dead-store-removal"

using namespace mlir;

namespace {

class DSRPattern : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // If we have a cc.alloca and all of its uses are cc.store ops, then these are
  // dead stores. Anchoring on the alloca rather than on each store means the
  // alloca's users are scanned once, instead of once per store.
  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloca,
                                PatternRewriter &rewriter) const override {
    SmallVector<cudaq::cc::StoreOp> stores;
    // At least one store must be directly to the alloca, through a cast, or
    // through a compute_ptr with constant offsets.
    bool hasCandidate = false;
    auto testAllStoreUsers = [&](Operation *c, bool isCandidate) {
      for (auto v : c->getUsers()) {
        if (auto s = dyn_cast<cudaq::cc::StoreOp>(v)) {
          // Make sure this stores *to* the address rather stores the address.
          if (s.getPtrvalue() == c->getResult(0)) {
            stores.push_back(s);
            hasCandidate |= isCandidate;
            continue;
          }
        }
        return false;
      }
      return true;
    };

    for (auto u : alloca->getUsers()) {
      if (auto c = dyn_cast<cudaq::cc::CastOp>(u)) {
        if (!testAllStoreUsers(c, /*isCandidate=*/true)) {
          LLVM_DEBUG(llvm::dbgs() << "store not from cast of alloca.\n");
          return failure();
        }
        continue;
      }
      if (auto c = dyn_cast<cudaq::cc::ComputePtrOp>(u)) {
        if (!testAllStoreUsers(c, c.getNumOperands() == 1)) {
          LLVM_DEBUG(llvm::dbgs() << "store not from compute_ptr of alloca.\n");
          return failure();
        }
        continue;
      }

      if (auto s = dyn_cast<cudaq::cc::StoreOp>(u))
        if (s.getPtrvalue() == alloca.getResult()) {
          stores.push_back(s);
          hasCandidate = true;
          continue;
        }
      LLVM_DEBUG(llvm::dbgs() << "alloca use is not store/cast/compute_ptr.\n");
      return failure();
    }
    if (!hasCandidate)
      return failure();
    for (auto s : stores)
      rewriter.eraseOp(s);
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
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
