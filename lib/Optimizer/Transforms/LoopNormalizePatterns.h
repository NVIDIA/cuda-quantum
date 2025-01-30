/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

class LoopPat : public mlir::OpRewritePattern<cudaq::cc::LoopOp> {
public:
  explicit LoopPat(mlir::MLIRContext *ctx, bool aci, bool ab)
      : OpRewritePattern(ctx), allowClosedInterval(aci), allowEarlyExit(ab) {}

  mlir::LogicalResult
  matchAndRewrite(cudaq::cc::LoopOp loop,
                  mlir::PatternRewriter &rewriter) const override;
  bool allowClosedInterval;
  bool allowEarlyExit;
};
