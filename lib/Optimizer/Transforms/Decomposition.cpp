/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_DECOMPOSITIONPASS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct Decomposition
    : public cudaq::opt::impl::DecompositionPassBase<Decomposition> {
  using DecompositionPassBase::DecompositionPassBase;

  /// Initialize the decomposer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    cudaq::populateWithAllDecompositionPatterns(owningPatterns);
    patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                       disabledPatterns, enabledPatterns);
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();

    // First, we walk the whole module in search for controlled `quake.apply`
    // operations: If present, we conservatively don't do any decompostions. We
    // also collect quantum kernels.
    //
    // TODO: Evaluate if preventing decompostion when there is at least one
    // controlled `quake.apply` in the whole module is too convervative.
    SmallVector<Operation *, 16> kernels;
    auto walkResult = module.walk([&kernels](Operation *op) {
      // Check if it is a quantum kernel
      if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        if (funcOp->hasAttr(cudaq::entryPointAttrName)) {
          kernels.push_back(funcOp);
          return WalkResult::advance();
        }
        for (auto arg : funcOp.getArguments())
          if (isa<quake::RefType, quake::VeqType>(arg.getType())) {
            kernels.push_back(funcOp);
            return WalkResult::advance();
          }
        // Skip functions which are not quantum kernels
        return WalkResult::skip();
      }
      // Check if it is controlled quake.apply
      if (auto applyOp = dyn_cast<quake::ApplyOp>(op))
        if (!applyOp.getControls().empty())
          return WalkResult::interrupt();

      return WalkResult::advance();
    });

    // Nothing to do:
    if (walkResult.wasInterrupted() || kernels.empty())
      return;

    // Process kernels in parallel
    LogicalResult rewriteResult = failableParallelForEach(
        module.getContext(), kernels, [&](Operation *op) {
          LogicalResult converged = applyPatternsAndFoldGreedily(op, patterns);

          // Decomposition is best-effort. Non-convergence is only a pass
          // failure if the user asked for convergence.
          if (testConvergence && failed(converged))
            return failure();
          return success();
        });

    // It only fails when testing for convergence
    if (failed(rewriteResult))
      signalPassFailure();
  }

  FrozenRewritePatternSet patterns;
};

} // namespace
