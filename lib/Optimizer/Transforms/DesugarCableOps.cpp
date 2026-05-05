/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_DESUGARCABLEOPS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "desugar-cable-ops"

using namespace mlir;

/**
   \file

   Desugar the `quake.attach_wire` and `quake.detach_wire` cable operations into
   more primitive operations.  See `Passes.td` for the description.
 */

namespace {
class AttachWirePattern : public OpRewritePattern<quake::AttachWireOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::AttachWireOp attach,
                                PatternRewriter &rewriter) const override {
    auto loc = attach.getLoc();
    auto *ctx = rewriter.getContext();
    auto size = cast<quake::CableType>(attach.getCable().getType()).getSize();
    LLVM_DEBUG(llvm::dbgs() << "attach: " << attach << '\n');
    SmallVector<Type> wiresTy(size, quake::WireType::get(ctx));
    auto split =
        quake::SplitCableOp::create(rewriter, loc, wiresTy, attach.getCable());
    SmallVector<Value> args{split.getResults().begin(),
                            split.getResults().begin() + attach.getIndex()};
    args.push_back(attach.getWire());
    args.append(split.getResults().begin() + attach.getIndex(),
                split.getResults().end());
    [[maybe_unused]] auto bundle =
        rewriter.replaceOpWithNewOp<quake::BundleCableOp>(
            attach, attach.getType(), ValueRange{args});
    LLVM_DEBUG(llvm::dbgs() << "split/bundle: " << split << '\n'
                            << bundle << '\n');
    return success();
  }
};

class DetachWirePattern : public OpRewritePattern<quake::DetachWireOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::DetachWireOp detach,
                                PatternRewriter &rewriter) const override {
    auto loc = detach.getLoc();
    auto *ctx = rewriter.getContext();
    auto size = cast<quake::CableType>(detach.getCable().getType()).getSize();
    LLVM_DEBUG(llvm::dbgs() << "detach: " << detach << '\n');
    SmallVector<Type> wiresTy(size, quake::WireType::get(ctx));
    auto split =
        quake::SplitCableOp::create(rewriter, loc, wiresTy, detach.getCable());
    auto pos = detach.getIndex();
    SmallVector<Value> args;
    args.reserve(split.getResults().size() - 1);
    for (auto [i, res] : llvm::enumerate(split.getResults()))
      if (i != pos)
        args.push_back(res);
    Value detachee = split.getResult(pos);
    Value bundle = quake::BundleCableOp::create(
        rewriter, loc, detach.getResult(1).getType(), ValueRange{args});
    rewriter.replaceOp(detach, ValueRange{detachee, bundle});
    LLVM_DEBUG(llvm::dbgs() << "split/bundle: " << split << '\n'
                            << bundle << '\n');
    return success();
  }
};

class DesugarCableOpsPass
    : public cudaq::opt::impl::DesugarCableOpsBase<DesugarCableOpsPass> {
public:
  using DesugarCableOpsBase::DesugarCableOpsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<AttachWirePattern, DetachWirePattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
