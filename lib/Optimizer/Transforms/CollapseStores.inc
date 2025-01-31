/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// These patterns are used by the collapse-stores and cc-loop-unroll passes.

// This file must be included after a `using namespace mlir;` as it uses bare
// identifiers from that namespace.

namespace {
/// Remove stores followed by a store to the same pointer
/// if the pointer is not used in between.
/// ```
/// cc.store %c0_i64, %5 : !cc.ptr<i64>
///  // no use of %5 until next line
/// cc.store %3, %5 : !cc.ptr<i64>
/// ───────────────────────────────────────────
/// cc.store %3, %5 : !cc.ptr<i64>
/// ```
class RemoveUselessStorePattern : public OpRewritePattern<cudaq::cc::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  explicit RemoveUselessStorePattern(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

  /// Remove stores followed by a store to the same pointer
  /// if the pointer is not used in between.
  /// ```
  /// cc.store %c0_i64, %5 : !cc.ptr<i64>
  ///  // no use of %5 until next line
  /// cc.store %3, %5 : !cc.ptr<i64>
  /// ───────────────────────────────────────────
  /// cc.store %3, %5 : !cc.ptr<i64>
  /// ```
  LogicalResult matchAndRewrite(cudaq::cc::StoreOp store,
                                PatternRewriter &rewriter) const override {
    if (isUselessStore(store)) {
      rewriter.eraseOp(store);
      return success();
    }
    return failure();
  }

private:
  /// Detect if the current store can be removed.
  static bool isUselessStore(cudaq::cc::StoreOp store) {
    Value currentPtr;

    if (!isStoreToStack(store))
      return false;

    auto block = store.getOperation()->getBlock();
    for (auto &op : *block) {
      if (auto s = dyn_cast<cudaq::cc::StoreOp>(&op)) {
        auto nextPtr = s.getPtrvalue();
        if (store == s) {
          // Start searching from the current store
          currentPtr = nextPtr;
        } else {
          // Found an overriding store, the current store is useless
          if (currentPtr == nextPtr)
            return true;

          // Found a use for a current ptr before the overriding store
          if (currentPtr && isUsed(currentPtr, &op))
            return false;
        }
      } else {
        // Found a use for a current ptr before the overriding store
        if (currentPtr && isUsed(currentPtr, &op))
          return false;
      }
    }
    // No multiple stores to the same location found
    return false;
  }

  /// Detect stores to stack locations
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

    if (auto alloca = ptrOp.getDefiningOp<cudaq::cc::AllocaOp>())
      return true;

    return false;
  }

  /// Detect if value is used in the op or its nested blocks.
  static bool isUsed(Value v, Operation *op) {
    for (auto opnd : op->getOperands())
      if (opnd == v)
        return true;

    for (auto &region : op->getRegions())
      for (auto &b : region)
        for (auto &innerOp : b)
          if (isUsed(v, &innerOp))
            return true;

    return false;
  }
};

} // namespace
