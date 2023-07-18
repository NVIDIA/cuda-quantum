/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_PROMOTEREFTOVEQALLOC
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
struct AllocaPat : public OpRewritePattern<quake::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  // Replace:
  //   %1 = quake.alloca !quake.ref
  // with:
  //   %0 = quake.alloca !quake.veq<1>
  //   %1 = quake.extract_ref %0[0] : (!quake.veq<1>) -> !quake.ref
  LogicalResult matchAndRewrite(quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    if (isa<quake::VeqType>(alloc.getType()))
      return failure();
    Value newAlloc = rewriter.create<quake::AllocaOp>(alloc.getLoc(), 1u);
    rewriter.replaceOpWithNewOp<quake::ExtractRefOp>(alloc, newAlloc, 0u);
    return success();
  }
};

class PromoteRefToVeqAllocPass
    : public cudaq::opt::impl::PromoteRefToVeqAllocBase<
          PromoteRefToVeqAllocPass> {
public:
  using PromoteRefToVeqAllocBase::PromoteRefToVeqAllocBase;

  void runOnOperation() override {
    auto *op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaPat>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns)))) {
      op->emitOpError("could not promote allocations");
      signalPassFailure();
    }
  }
};
} // namespace
