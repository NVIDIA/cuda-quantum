/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
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
#define GEN_PASS_DEF_DEADQUANTUMELIMINATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "dqe"

using namespace mlir;

// TODO: We can expand these patterns to cover veqs, struqs, and cables as well.
// In those cases, there are vacuous uses that must be accounted for in order to
// partially eliminate those quantum allocations. For example, a
// `quake.extract_ref` might be a use, but itself have no users.

namespace {
class RefPattern : public OpRewritePattern<cudaq::quake::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    if (std::distance(alloc->getUsers().begin(), alloc->getUsers().end()) > 1)
      return failure();
    if (alloc->use_empty()) {
      rewriter.eraseOp(alloc);
      return success();
    }
    // There is exactly 1 use.
    auto dealloc =
        dyn_cast<cudaq::quake::DeallocOp>(*alloc->getUsers().begin());
    if (!dealloc)
      return failure();
    rewriter.eraseOp(dealloc);
    rewriter.eraseOp(alloc);
    return success();
  }
};

class WirePattern : public OpRewritePattern<cudaq::quake::NullWireOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::NullWireOp nullWire,
                                PatternRewriter &rewriter) const override {
    // Wires are linear types. There must be exactly 1 use.
    auto sink = dyn_cast<cudaq::quake::SinkOp>(*nullWire->getUsers().begin());
    if (!sink)
      return failure();
    rewriter.eraseOp(sink);
    rewriter.eraseOp(nullWire);
    return success();
  }
};

class DQEPass : public cudaq::opt::impl::DeadQuantumEliminationBase<DQEPass> {
public:
  using DeadQuantumEliminationBase::DeadQuantumEliminationBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before DQE:\n" << *op << '\n');
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RefPattern, WirePattern>(ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After DQE:\n" << *op << '\n');
  }
};
} // namespace
