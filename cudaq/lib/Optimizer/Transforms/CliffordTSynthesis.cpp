/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#ifdef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS
#include "cudaq/Synthesis/Circuit/Circuit.h"
#include "cudaq/Synthesis/Circuit/Gate.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Support/Result.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#endif

namespace cudaq::opt {
#define GEN_PASS_DEF_CLIFFORDTSYNTHESIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "clifford-t-synthesis"

using namespace mlir;

#ifdef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS

namespace {

struct RzPattern : OpRewritePattern<cudaq::quake::RzOp> {
  RzPattern(MLIRContext *ctx, double epsilon, int32_t diophantineTimeoutMs,
            int32_t factoringTimeoutMs, int32_t retryCount,
            llvm::StringRef onDynamicAngle, double skipBelow)
      : OpRewritePattern(ctx), epsilon(epsilon),
        diophantineTimeoutMs(diophantineTimeoutMs),
        factoringTimeoutMs(factoringTimeoutMs), retryCount(retryCount),
        onDynamicAngle(onDynamicAngle.str()), skipBelow(skipBelow) {}

  LogicalResult matchAndRewrite(cudaq::quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty()) {
      op.emitRemark(
          "clifford-t-synthesis: skipping controlled rotation; run "
          "ApplyOpSpecialization to materialize controls before synthesis");
      return failure();
    }

    FloatAttr attr;
    if (!matchPattern(op.getParameter(), m_Constant<FloatAttr>(&attr))) {
      if (onDynamicAngle == "skip") {
        op.emitRemark("clifford-t-synthesis: skipping non-constant rotation "
                      "angle (--on-dynamic-angle=skip)");
        return failure();
      }
      op.emitError("clifford-t-synthesis: rotation angle is not a "
                   "compile-time constant. Fold the angle upstream or set "
                   "--on-dynamic-angle=skip to leave the op alone.");
      return failure();
    }

    double theta = attr.getValueAsDouble();
    if (!std::isfinite(theta)) {
      op.emitError("clifford-t-synthesis: rotation angle is NaN or Inf");
      return failure();
    }

    if (std::abs(theta) < skipBelow) {
      rewriter.eraseOp(op);
      return success();
    }

    cudaq::synth::Real thetaReal(theta);
    cudaq::synth::Real epsilonReal(epsilon);
    cudaq::synth::FailureOr<cudaq::synth::Circuit> circuit =
        cudaq::synth::failure();
    for (int32_t attempt = 0; attempt <= retryCount; ++attempt) {
      circuit = cudaq::synth::gridsynth(
          thetaReal, epsilonReal,
          static_cast<int32_t>(diophantineTimeoutMs << attempt),
          static_cast<int32_t>(factoringTimeoutMs << attempt));
      if (cudaq::synth::succeeded(circuit))
        break;
    }
    if (cudaq::synth::failed(circuit)) {
      op.emitError("clifford-t-synthesis: gridsynth failed for theta=")
          << theta << " after " << (retryCount + 1)
          << " attempts; raise --diophantine-timeout-ms or "
             "--factoring-timeout-ms";
      return failure();
    }

    Location loc = op.getLoc();
    Value target = op.getTarget();
    for (cudaq::synth::Gate g : *circuit) {
      switch (g) {
      case cudaq::synth::Gate::H:
        cudaq::quake::HOp::create(rewriter, loc, ValueRange{target});
        break;
      case cudaq::synth::Gate::S:
        cudaq::quake::SOp::create(rewriter, loc, ValueRange{target});
        break;
      case cudaq::synth::Gate::T:
        cudaq::quake::TOp::create(rewriter, loc, ValueRange{target});
        break;
      case cudaq::synth::Gate::X:
        cudaq::quake::XOp::create(rewriter, loc, ValueRange{target});
        break;
      case cudaq::synth::Gate::W:
        // ω = e^{iπ/4} global phase - dropped. Valid only when controls have
        // been materialized upstream (ApplyOpSpecialization). When controlled
        // rotations are reintroduced this must re-emit a phase pair on the
        // outermost control qubit.
        break;
      }
    }
    rewriter.eraseOp(op);
    return success();
  }

  double epsilon;
  int32_t diophantineTimeoutMs;
  int32_t factoringTimeoutMs;
  int32_t retryCount;
  std::string onDynamicAngle;
  double skipBelow;
};

} // namespace

#endif // CUDAQ_HAS_CLIFFORD_T_SYNTHESIS

namespace {

class CliffordTSynthesisPass
    : public cudaq::opt::impl::CliffordTSynthesisBase<CliffordTSynthesisPass> {
public:
  using CliffordTSynthesisBase::CliffordTSynthesisBase;

  void runOnOperation() override {
#ifndef CUDAQ_HAS_CLIFFORD_T_SYNTHESIS
    getOperation().emitError(
        "clifford-t-synthesis: CUDA-Q was built without the synthesis "
        "library (GMP/MPFR missing at configure time). Rebuild with "
        "libgmp-dev and libmpfr-dev installed.");
    signalPassFailure();
    return;
#else
    LLVM_DEBUG(llvm::dbgs()
               << "clifford-t-synthesis: epsilon=" << epsilon
               << " diophantine-timeout-ms=" << diophantineTimeoutMs
               << " factoring-timeout-ms=" << factoringTimeoutMs
               << " retry-count=" << retryCount << " on-dynamic-angle="
               << onDynamicAngle << " skip-below=" << skipBelow << '\n');

    auto prec = static_cast<mpfr_prec_t>(
        std::max<double>(64.0, std::ceil(-std::log2(epsilon) * 4.0 + 64.0)));
    cudaq::synth::Real::set_default_precision(prec);

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<RzPattern>(ctx, epsilon, diophantineTimeoutMs,
                            factoringTimeoutMs, retryCount, onDynamicAngle,
                            skipBelow);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
#endif
  }
};

} // namespace
