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

namespace cudaq::opt {
#define GEN_PASS_DEF_EXPANDVEQUSES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "expand-veq-uses"

using namespace mlir;

namespace {
/// Replace a single-qubit operator whose target is a constant sized veq of
/// size \e N with \e N copies of that operator, one per element of the veq.
/// The controls, parameters, and attributes of the original operator are
/// replicated verbatim on each copy.
template <typename OP>
class ExpandBroadcastPat : public OpRewritePattern<OP> {
public:
  using OpRewritePattern<OP>::OpRewritePattern;

  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    // Only an uncontrolled operator broadcasts
    if (op.getTargets().size() != 1 || !op.getControls().empty())
      return failure();
    Value target = op.getTargets()[0];
    if (!isa<cudaq::quake::VeqType>(target.getType()))
      return failure();
    auto size = cudaq::quake::getVeqSize(target);
    if (!size)
      return failure();

    auto loc = op.getLoc();
    // The sole target is the last operand (skip angles for rotations)
    unsigned targetPos = op->getNumOperands() - 1;
    for (std::size_t i = 0; i < *size; ++i) {
      Value ref = cudaq::quake::ExtractRefOp::create(rewriter, loc, target, i);
      Operation *clone = rewriter.clone(*op.getOperation());
      clone->setOperand(targetPos, ref);
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// quake.evince %veq, %r : (!quake.veq<n>, !quake.ref) -> ()
// ───────────────────────────────────────────────────────────────────
// %0 = quake.extract_ref %veq[0] : (!quake.veq<n>) -> !quake.ref
// ...
// %n = quake.extract_ref %veq[n-1] : (!quake.veq<n>) -> !quake.ref
// quake.evince %0, ..., %n, %r : (!quake.ref, ..., !quake.ref,
//     !quake.ref) -> ()
//
// A veq is never a linear type (only wires/cables are), so expanding it into
// its constituent refs never touches `outs` -- the rewrite only grows `args`.
class ExpandEvincePattern : public OpRewritePattern<cudaq::quake::EvinceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::EvinceOp evin,
                                PatternRewriter &rewriter) const override {
    auto loc = evin.getLoc();
    SmallVector<Value> newArgs;
    bool didExpand = false;
    for (Value arg : evin.getArgs()) {
      auto size = cudaq::quake::getVeqSize(arg);
      if (!size) {
        newArgs.push_back(arg);
        continue;
      }

      // extract_ref requires the sized source of a relaxed vector.
      Value vector = arg;
      if (auto relax = arg.getDefiningOp<cudaq::quake::RelaxSizeOp>())
        vector = relax.getInputVec();
      for (std::size_t i = 0; i < *size; ++i)
        newArgs.push_back(
            cudaq::quake::ExtractRefOp::create(rewriter, loc, vector, i));
      didExpand = true;
    }
    if (!didExpand)
      return failure();

    rewriter.replaceOpWithNewOp<cudaq::quake::EvinceOp>(
        evin, newArgs, evin.getCompilerGenerated());
    return success();
  }
};

struct ExpandVeqUsesPass
    : public cudaq::opt::impl::ExpandVeqUsesBase<ExpandVeqUsesPass> {
  using ExpandVeqUsesBase::ExpandVeqUsesBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ExpandBroadcastPat<cudaq::quake::HOp>,
                    ExpandBroadcastPat<cudaq::quake::PhasedRxOp>,
                    ExpandBroadcastPat<cudaq::quake::R1Op>,
                    ExpandBroadcastPat<cudaq::quake::RxOp>,
                    ExpandBroadcastPat<cudaq::quake::RyOp>,
                    ExpandBroadcastPat<cudaq::quake::RzOp>,
                    ExpandBroadcastPat<cudaq::quake::SOp>,
                    ExpandBroadcastPat<cudaq::quake::TOp>,
                    ExpandBroadcastPat<cudaq::quake::U2Op>,
                    ExpandBroadcastPat<cudaq::quake::U3Op>,
                    ExpandBroadcastPat<cudaq::quake::XOp>,
                    ExpandBroadcastPat<cudaq::quake::YOp>,
                    ExpandBroadcastPat<cudaq::quake::ZOp>, ExpandEvincePattern>(
        ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
