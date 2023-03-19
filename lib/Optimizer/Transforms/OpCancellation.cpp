/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXInterfaces.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

class RewriteAdjPairs : public RewritePattern {
public:
  RewriteAdjPairs(MLIRContext *context)
      : RewritePattern(MatchInterfaceOpTypeTag(),
                       TypeID::get<qtx::OperatorInterface>(),
                       /*benefit*/ 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto thisOperator = dyn_cast<qtx::OperatorInterface>(op);
    bool isHermitian = op->hasTrait<cudaq::Hermitian>();

    auto opUsers = op->getUsers();
    // In QTX, an operator cannot have a empty list of users because at the
    // very least its resulting wires must be deallocated.
    assert(!opUsers.empty() && "A operator cannot have an empty list of users");

    auto other = *opUsers.begin();
    auto otherOperator = dyn_cast<qtx::OperatorInterface>(other);
    // In QTX, adjoint instructions always have the same operator name as
    // the non-adjoint one.  We use an attribute to indicate which kind we
    // are using. Hence, if the names are different, they cannot be cancel.
    if (op->getName() != other->getName())
      return failure();
    // To successfully cancel two instructions, the operator must be either
    // hermitian (i.e., self-adjoint) or only one has the `adj` attribute.
    bool maybeCancel =
        isHermitian | (thisOperator.isAdj() ^ otherOperator.isAdj());
    if (!maybeCancel)
      return failure();

    // We did some preliminary checks and things look good.  Maybe we can
    // cancel out these operations.  We the only thing left to check is
    // whether the operations have the same parameters and controls, and if
    // the target results of `this` is equal to the input targets of `other`
    if (thisOperator.getParameters() != otherOperator.getParameters())
      return failure();
    if (thisOperator.getControls() != otherOperator.getControls())
      return failure();
    if (thisOperator.getNewTargets() != otherOperator.getTargets())
      return failure();
    // At this point, we identified that `this` and `other` are adjoint
    // operations that can be removed.  We need to make sure that any
    // operation which relies on the results of `other` is correctly connected
    // to the target inputs of `this`
    //
    // Example:
    // ```mlir
    // %w_0 = qtx.alloc : !qtx.wire
    // %w_1 = qtx.x(%w_0) : (!qtx.wire) -> (!qtx.wire)  -> this
    // %w_2 = qtx.x(%w_1) : (!qtx.wire) -> (!qtx.wire)  -> other
    // %w_3 = qtx.t(%w_2) : (!qtx.wire) -> (!qtx.wire)
    // ```
    //
    // After removing both `qtx.x` operations, we need to make sure that
    // `qtx.t` will take `%w_0` as input.
    for (unsigned i = 0, end = other->getNumResults(); i < end; ++i)
      other->getResult(i).replaceAllUsesWith(thisOperator.getTarget(i));

    op->dropAllUses();
    rewriter.eraseOp(op);
    rewriter.eraseOp(other);
    return success();
  }
};

} // namespace

struct OpCancellation : public cudaq::opt::OpCancellationBase<OpCancellation> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<RewriteAdjPairs>(context);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    // TODO: investigate if using the result of this would be important for this
    // pass.  It tells us whether pattern rewriting converged or not.  For such
    // a simple pass, this might not be necessary
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                       config);
  }
};

std::unique_ptr<Pass> cudaq::opt::createOpCancellationPass() {
  return std::make_unique<OpCancellation>();
}
