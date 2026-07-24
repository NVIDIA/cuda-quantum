/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cmath>

namespace cudaq::opt {
#define GEN_PASS_DEF_PRUNENEGLIGIBLEROTATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "prune-negligible-rotations"

using namespace mlir;

namespace {

// Erases an uncontrolled constant-angle rotation that is the identity up to a
// global phase, treating it as the identity. All of `rz`, `rx`, `ry`, and `r1`
// have period 2*pi up to a global phase (e.g. rz(2*pi) = -I), so the angle is
// folded into (-pi, pi] and pruned when the folded magnitude is below
// `threshold`. The discarded global phase is unobservable because only
// uncontrolled rotations are pruned. Works for both quake semantics:
//   - memory (`!quake.ref`): the op produces no result, so it is dropped.
//   - value (`!quake.wire`): the op threads its target wire to a result wire.
//     The identity forwards that input wire to the result's users.
// Controlled rotations are left in place. The wire threading of a controlled
// gate carries the control wires through as well, and materializing controls
// (ApplyOpSpecialization) is the intended path before pruning them.
template <typename RotOp>
struct PruneRotationPattern : OpRewritePattern<RotOp> {
  PruneRotationPattern(MLIRContext *ctx, double threshold)
      : OpRewritePattern<RotOp>(ctx), threshold(threshold) {}

  LogicalResult matchAndRewrite(RotOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();

    FloatAttr attr;
    if (!matchPattern(op.getParameter(), m_Constant<FloatAttr>(&attr)))
      return failure();

    double theta = attr.getValueAsDouble();
    if (!std::isfinite(theta))
      return failure();

    // Fold into (-pi, pi]: a full turn is identity up to a global phase, so
    // e.g. rz(2*pi) = -I is pruned just like rz(0). std::remainder yields the
    // representative in [-pi, pi] closest to zero.
    double residual = std::remainder(theta, 2.0 * M_PI);
    if (std::abs(residual) >= threshold)
      return failure();

    // Negligible angle: replace with the identity.
    if (op->getNumResults() == 0)
      rewriter.eraseOp(op); // memory semantics
    else
      rewriter.replaceOp(op, op.getTarget()); // value semantics: pass the wire
    return success();
  }

  double threshold;
};

class PruneNegligibleRotationsPass
    : public cudaq::opt::impl::PruneNegligibleRotationsBase<
          PruneNegligibleRotationsPass> {
public:
  using PruneNegligibleRotationsBase::PruneNegligibleRotationsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<PruneRotationPattern<cudaq::quake::RzOp>,
                    PruneRotationPattern<cudaq::quake::RxOp>,
                    PruneRotationPattern<cudaq::quake::RyOp>,
                    PruneRotationPattern<cudaq::quake::R1Op>>(ctx, threshold);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
