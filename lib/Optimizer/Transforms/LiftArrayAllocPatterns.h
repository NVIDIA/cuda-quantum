/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"

class AllocaPattern : public mlir::OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  explicit AllocaPattern(mlir::MLIRContext *ctx, mlir::DominanceInfo &di,
                         llvm::StringRef fn)
      : OpRewritePattern(ctx), dom(di), funcName(fn) {}

  mlir::LogicalResult
  matchAndRewrite(cudaq::cc::AllocaOp alloc,
                  mlir::PatternRewriter &rewriter) const override;

  // Determine if \p alloc is a legit candidate for promotion to a constant
  // array value. \p scoreboard is a vector of store mlir::Operations. Each
  // element of the allocated array must be written to exactly 1 time, and the
  // scoreboard is used to track these stores. \p dom is the dominance info for
  // this function (to ensure the stores happen before uses).
  static bool
  isGoodCandidate(cudaq::cc::AllocaOp alloc,
                  llvm::SmallVectorImpl<mlir::Operation *> &scoreboard,
                  mlir::DominanceInfo &dom);
  mlir::DominanceInfo &dom;
  llvm::StringRef funcName;
};
