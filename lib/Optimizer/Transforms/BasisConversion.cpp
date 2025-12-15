/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_BASISCONVERSIONPASS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct BasisConversion
    : public cudaq::opt::impl::BasisConversionPassBase<BasisConversion> {
  using BasisConversionPassBase::BasisConversionPassBase;

  void runOnOperation() override {
    auto module = getOperation();
    if (basis.empty()) {
      module.emitError("Basis conversion requires a target basis");
      signalPassFailure();
      return;
    }

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

    if (walkResult.wasInterrupted()) {
      module.emitError("Basis conversion doesn't work with `quake.apply`");
      signalPassFailure();
      return;
    }

    if (kernels.empty())
      return;

    // Setup target and patterns
    auto target = cudaq::createBasisTarget(getContext(), basis);
    RewritePatternSet owningPatterns(&getContext());
    FrozenRewritePatternSet patterns;
    if (enabledPatterns.empty()) {
      cudaq::selectDecompositionPatterns(owningPatterns, basis,
                                         disabledPatterns);
      patterns = FrozenRewritePatternSet(std::move(owningPatterns));
    } else {
      cudaq::populateWithAllDecompositionPatterns(owningPatterns);
      patterns = FrozenRewritePatternSet(std::move(owningPatterns),
                                         disabledPatterns, enabledPatterns);
    }

    // Process kernels in parallel
    LogicalResult rewriteResult = failableParallelForEach(
        module.getContext(), kernels, [&target, &patterns](Operation *op) {
          return applyFullConversion(op, *target, patterns);
        });

    if (failed(rewriteResult))
      signalPassFailure();
  }
};

} // namespace
