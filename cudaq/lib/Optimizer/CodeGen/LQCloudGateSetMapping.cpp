/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace cudaq::opt {
#define GEN_PASS_DEF_LQCLOUDGATEDECOMPOSITION
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

namespace {

// Preserve the provider's input contract before public decomposition rewrites
// controls, and retain native X/Y rather than adding physical gate depth.
struct LQCloudGateDecompositionPass
    : public cudaq::opt::impl::LQCloudGateDecompositionBase<
          LQCloudGateDecompositionPass> {
  using LQCloudGateDecompositionBase::LQCloudGateDecompositionBase;

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<Operation *> operators;
    module.walk([&](Operation *operation) {
      if (isa<cudaq::quake::OperatorInterface>(operation))
        operators.push_back(operation);
    });
    IRRewriter rewriter(&getContext());
    for (Operation *operation : operators) {
      auto op = cast<cudaq::quake::OperatorInterface>(operation);
      if (op.getNegatedControls() ||
          (isa<cudaq::quake::SwapOp>(operation) && !op.getControls().empty())) {
        operation->emitOpError(
            op.getNegatedControls()
                ? "LQCloud does not support negated controls"
                : "LQCloud does not support controlled SWAP");
        signalPassFailure();
        return;
      }
      // Main provides CX/CY, Rx/Ry, U3, S/T and R1 decompositions. Scalar Z
      // has no Rz public pattern; this is the only provider-local rewrite.
      if (auto z = dyn_cast<cudaq::quake::ZOp>(operation);
          z && z.getControls().empty() && z.getTargets().size() == 1) {
        rewriter.setInsertionPoint(z);
        auto angle = cudaq::opt::factory::createFloatConstant(
            z.getLoc(), rewriter, 3.14159265358979323846,
            rewriter.getF64Type());
        cudaq::quake::RzOp::create(rewriter, z.getLoc(), ValueRange{angle},
                                   ValueRange{}, z.getTargets());
        rewriter.eraseOp(z);
      }
    }
  }
};

} // namespace
