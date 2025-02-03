/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_WRITEAFTERWRITEELIMINATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "write-after-write-elimination"

using namespace mlir;

namespace {
/// Remove stores followed by a store to the same pointer
/// if the pointer is not used in between.
/// ```
/// cc.store %c0_i64, %1 : !cc.ptr<i64>
/// // no use of %1 until next line
/// cc.store %0, %1 : !cc.ptr<i64>
/// ───────────────────────────────────────────
/// cc.store %0, %1 : !cc.ptr<i64>
/// ```
class RemoveOverriddenStorePattern
    : public OpRewritePattern<cudaq::cc::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  explicit RemoveOverriddenStorePattern(MLIRContext *ctx, DominanceInfo &di)
      : OpRewritePattern(ctx), dom(di) {}

  LogicalResult matchAndRewrite(cudaq::cc::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (isOverridenStore(store)) {
      rewriter.eraseOp(store);
      return success();
    }
    return failure();
  }

private:
  /// Detect if the current store is overriden by another store in the same
  /// block.
  bool isOverridenStore(cudaq::cc::StoreOp store) const {
    Value currentPtr;

    if (!isStoreToStack(store))
      return false;

    auto block = store.getOperation()->getBlock();
    for (auto &op : *block) {
      if (auto currentStore = dyn_cast<cudaq::cc::StoreOp>(&op)) {
        auto nextPtr = currentStore.getPtrvalue();
        if (store == currentStore) {
          // Start searching from the current store
          currentPtr = nextPtr;
        } else {
          // Found an overriding store, the current store is useless
          if (currentPtr == nextPtr)
            return isReplacement(currentPtr, store, currentStore);
        }
      }
    }
    // No multiple stores to the same location found
    return false;
  }

  /// Detect stores to stack locations, for example:
  /// ```
  /// %1 = cc.alloca !cc.array<i64 x 2>
  ///
  /// %2 = cc.cast %1 : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<i64>
  /// cc.store %c0_i64, %2 : !cc.ptr<i64>
  ///
  /// %3 = cc.compute_ptr %1[1] : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<i64>
  /// cc.store %c0_i64, %3 : !cc.ptr<i64>
  /// ```
  static bool isStoreToStack(cudaq::cc::StoreOp store) {
    auto ptrOp = store.getPtrvalue();
    if (auto cast = ptrOp.getDefiningOp<cudaq::cc::CastOp>())
      ptrOp = cast.getOperand();

    if (auto computePtr = ptrOp.getDefiningOp<cudaq::cc::ComputePtrOp>())
      ptrOp = computePtr.getBase();

    return isa_and_present<cudaq::cc::AllocaOp>(ptrOp.getDefiningOp());
  }

  /// Detect if value is used in the op or its nested blocks.
  bool isReplacement(Value ptr, cudaq::cc::StoreOp store,
                     cudaq::cc::StoreOp replacement) const {
    // Check that there are no stores dominated by the store and not dominated
    // by the replacement (i.e. used in between the store and the replacement)
    for (auto *user : ptr.getUsers()) {
      if (user != store && user != replacement) {
        if (dom.dominates(store, user) && !dom.dominates(replacement, user)) {
          LLVM_DEBUG(llvm::dbgs() << "store " << replacement
                                  << " is used before: " << store << '\n');
          return false;
        }
      }
    }
    return true;
  }

  DominanceInfo &dom;
};

class WriteAfterWriteEliminationPass
    : public cudaq::opt::impl::WriteAfterWriteEliminationBase<
          WriteAfterWriteEliminationPass> {
public:
  using WriteAfterWriteEliminationBase::WriteAfterWriteEliminationBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    DominanceInfo domInfo(func);
    RewritePatternSet patterns(ctx);
    patterns.insert<RemoveOverriddenStorePattern>(ctx, domInfo);

    LLVM_DEBUG(llvm::dbgs()
               << "Before write after write elimination: " << func << '\n');

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "After write after write elimination: " << func << '\n');
  }
};
} // namespace
