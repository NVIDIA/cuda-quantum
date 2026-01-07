/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#define GEN_PASS_DEF_LIFTARRAYALLOC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lift-array-alloc"

using namespace mlir;

#include "LiftArrayAllocPatterns.inc"

namespace {
class LiftArrayAllocPass
    : public cudaq::opt::impl::LiftArrayAllocBase<LiftArrayAllocPass> {
public:
  using LiftArrayAllocBase::LiftArrayAllocBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    DominanceInfo domInfo(func);
    StringRef funcName = func.getName();
    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaPattern>(ctx, domInfo, funcName);

    LLVM_DEBUG(llvm::dbgs()
               << "Before lifting constant array: " << func << '\n');

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "After lifting constant array: " << func << '\n');
  }
};
} // namespace
