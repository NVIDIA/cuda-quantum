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

/// Remove stores followed by a store to the same pointer
/// if the pointer is not used in between.
/// ```
/// cc.store %c0_i64, %5 : !cc.ptr<i64>
///  // no use of %5 until next line
/// cc.store %3, %5 : !cc.ptr<i64>
/// ───────────────────────────────────────────
/// cc.store %3, %5 : !cc.ptr<i64>
/// ```
class RemoveUselessStorePattern : public mlir::OpRewritePattern<cudaq::cc::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  
  explicit RemoveUselessStorePattern(mlir::MLIRContext *ctx): OpRewritePattern(ctx) {}

  mlir::LogicalResult matchAndRewrite(cudaq::cc::StoreOp store,
                                mlir::PatternRewriter &rewriter) const override;

private:
  /// Detect if the current store can be removed.
  static bool isUselessStore(cudaq::cc::StoreOp store);

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
  static bool isStoreToStack(cudaq::cc::StoreOp store);

  /// Detect if value is used in the op or its nested blocks.
  static bool isUsed(mlir::Value v, mlir::Operation *op);
};
