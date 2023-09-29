/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
#define GEN_PASS_DEF_PRUNECTRLRELATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "prune-ctrl-relations"

using namespace mlir;

namespace {
/// Pattern to convert any gate operator, which may have control operands that
/// are wires, to the same gate operator with `!quake.control` type operands.
template <typename OP>
class MakeControl : public OpRewritePattern<OP> {
public:
  using Base = OpRewritePattern<OP>;
  using Base::Base;

  LogicalResult matchAndRewrite(OP op,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    if (op.getControls().empty())
      return failure();
    auto wireTy = quake::WireType::get(ctx);
    bool nothingToDo = [&]() {
      for (auto cv : op.getControls())
        if (cv.getType() == wireTy)
          return false;
      return true;
    }();
    if (nothingToDo)
      return failure();

    // At this point `op` has (some) wires in control positions. Let's convert
    // this op to remove the artificial constraints.
    auto coarity = op.getWires().size();
    auto loc = op.getLoc();
    auto ctrlTy = quake::ControlType::get(ctx);
    SmallVector<Value> newCtrls;

    // For each wire control, convert the wire to a control with a ToControlOp.
    // this also reduces the coarity of the new op.
    for (auto cv : op.getControls()) {
      if (cv.getType() == wireTy) {
        Value input;
        if (auto fromCtrl = cv.template getDefiningOp<quake::FromControlOp>()) {
          input = fromCtrl.getCtrlbit();
        } else {
          input = rewriter.template create<quake::ToControlOp>(loc, ctrlTy, cv);
        }
        newCtrls.push_back(input);
        coarity--;
      } else {
        newCtrls.push_back(cv);
      }
    }

    // Create a copy of `op` with the correct coarity and with the control wires
    // each now passing through a ToControlOp.
    SmallVector<Type> wireTys{coarity, wireTy};
    auto newOp = rewriter.create<OP>(
        loc, wireTys, op.getIsAdjAttr(), op.getParameters(), newCtrls,
        op.getTargets(), op.getNegatedQubitControlsAttr());

    // Loop over the original controls again, this time adding a FromControlOp
    // so that the IR will type check when we replace the old op.
    std::size_t newIdx = 0;
    for (auto i : llvm::enumerate(op.getControls())) {
      auto cv = i.value();
      if (cv.getType() == wireTy) {
        Value fromCtrl = rewriter.template create<quake::FromControlOp>(
            loc, wireTy, newCtrls[i.index()]);
        op.getResult(i.index()).replaceAllUsesWith(fromCtrl);
      } else {
        op.getResult(i.index()).replaceAllUsesWith(newOp.getResult(newIdx++));
      }
    }
    // And finish by moving the target wires over to the new op.
    for (std::size_t i = op.getControls().size();
         i < op.getControls().size() + op.getTargets().size(); ++i) {
      op.getResult(i).replaceAllUsesWith(newOp.getResult(newIdx++));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

/// Simple forwarding of `!quake.control` values. If the argument to a
/// `quake.to_ctrl` operation is coming from a `quake.from_ctrl` operation, then
/// both operations can be bypassed and the input to the
/// `quake.from_control` can be forwarded directly to the users of the
/// `quake.to_ctrl`. There are no intervening ops on the wire by definition.
class ForwardControl : public OpRewritePattern<quake::ToControlOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ToControlOp toCtrl,
                                PatternRewriter &rewriter) const override {
    if (auto fromCtrl =
            toCtrl.getQubit().getDefiningOp<quake::FromControlOp>()) {
      rewriter.replaceOp(toCtrl, fromCtrl.getCtrlbit());
      return success();
    }
    return failure();
  }
};
} // namespace

#define WRAPPER(OpClass) MakeControl<quake::OpClass>
#define WRAPPER_GATE_OPS GATE_OPS(WRAPPER)

namespace {
class PruneCtrlRelationsPass
    : public cudaq::opt::impl::PruneCtrlRelationsBase<PruneCtrlRelationsPass> {
public:
  using PruneCtrlRelationsBase::PruneCtrlRelationsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<WRAPPER_GATE_OPS, ForwardControl>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
