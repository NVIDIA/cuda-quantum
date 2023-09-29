/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

inline Value createConstant(Location loc, double value, Type type,
                            PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantFloatOp>(loc, llvm::APFloat(value),
                                                 cast<FloatType>(type));
}

inline Value createConstant(Location loc, size_t value,
                            PatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, value,
                                               rewriter.getI64Type());
}

inline Value createDivF(Location loc, Value numerator, double denominator,
                        PatternRewriter &rewriter) {
  auto denominatorValue =
      createConstant(loc, denominator, numerator.getType(), rewriter);
  return rewriter.create<arith::DivFOp>(loc, numerator, denominatorValue);
}

/// Check whether the operation has the correct number of controls.
///
/// Note: This function assumes that the operation has already been tested for
/// reference semantics.
LogicalResult checkNumControls(quake::OperatorInterface op,
                               size_t requiredNumControls) {
  auto opControls = op.getControls();
  if (opControls.size() > requiredNumControls)
    return failure();

  // Compute the number of controls
  size_t numControls = 0;
  for (auto control : opControls) {
    if (auto veq = dyn_cast<quake::VeqType>(control.getType())) {
      if (!veq.hasSpecifiedSize())
        return failure();
      numControls += veq.getSize();
      continue;
    }
    numControls += 1;
  }

  return numControls == requiredNumControls ? success() : failure();
}

/// Check whether the operation has the correct number of controls. This
/// function take as input a mutable array reference, `controls`, which must
/// have the size equal to the number of controls. If the operation has `veq`s
/// as controls, split those into single qubit references.
///
/// Note: This function assumes that the operation has already been tested for
/// reference semantics.
LogicalResult checkAndExtractControls(quake::OperatorInterface op,
                                      MutableArrayRef<Value> controls,
                                      PatternRewriter &rewriter) {
  if (failed(checkNumControls(op, controls.size())))
    return failure();

  size_t controlIndex = 0;
  for (Value control : op.getControls()) {
    if (auto veq = dyn_cast<quake::VeqType>(control.getType())) {
      for (size_t i = 0, end = veq.getSize(); i < end; ++i) {
        Value index = createConstant(op.getLoc(), i, rewriter);
        Value qref =
            rewriter.create<quake::ExtractRefOp>(op.getLoc(), control, index);
        controls[controlIndex] = qref;
        controlIndex += 1;
      }
    } else {
      controls[controlIndex] = control;
      controlIndex += 1;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// HOp decompositions
//===----------------------------------------------------------------------===//

// quake.h target
// ───────────────────────────────────
// quake.phased_rx(π/2, π/2) target
// quake.phased_rx(π, 0) target
struct HToPhasedRx : public OpRewritePattern<quake::HOp> {
  using OpRewritePattern<quake::HOp>::OpRewritePattern;

  void initialize() { setDebugName("HToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::HOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi = createConstant(loc, M_PI, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);

    std::array<Value, 2> parameters = {pi_2, pi_2};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = pi;
    parameters[1] = zero;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// Naive mapping of R1 to Rz, ignoring the global phase.
// This is only expected to work with full inlining and
// quake apply specialization.
struct R1ToRz : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(quake::R1Op r1Op,
                                PatternRewriter &rewriter) const override {
    if (!r1Op.getControls().empty())
      return failure();

    rewriter.replaceOpWithNewOp<quake::RzOp>(
        r1Op, r1Op.isAdj(), r1Op.getParameters(), r1Op.getControls(),
        r1Op.getTargets());
    return success();
  }
};

// quake.swap a, b
// ───────────────────────────────────
// quake.cnot b, a;
// quake.cnot a, b;
// quake.cnot b, a;
struct SwapToCX : public OpRewritePattern<quake::SwapOp> {
  using OpRewritePattern<quake::SwapOp>::OpRewritePattern;

  void initialize() { setDebugName("SwapToCX"); }

  LogicalResult matchAndRewrite(quake::SwapOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value a = op.getTarget(0);
    Value b = op.getTarget(1);

    rewriter.create<quake::XOp>(loc, b, a);
    rewriter.create<quake::XOp>(loc, a, b);
    rewriter.create<quake::XOp>(loc, b, a);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.h control, target
// ───────────────────────────────────
// quake.s target;
// quake.h target;
// quake.t target;
// quake.x control, target;
// quake.t<adj> target;
// quake.h target;
// quake.s<adj> target;
struct CHToCX : public OpRewritePattern<quake::HOp> {
  using OpRewritePattern<quake::HOp>::OpRewritePattern;

  void initialize() { setDebugName("CHToCX"); }

  LogicalResult matchAndRewrite(quake::HOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();
    if (failed(checkNumControls(op, 1)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value control = op.getControls()[0];
    Value target = op.getTarget();

    rewriter.create<quake::SOp>(loc, target);
    rewriter.create<quake::HOp>(loc, target);
    rewriter.create<quake::TOp>(loc, target);
    rewriter.create<quake::XOp>(loc, control, target);
    rewriter.create<quake::TOp>(loc, /*isAdj=*/true, target);
    rewriter.create<quake::HOp>(loc, target);
    rewriter.create<quake::SOp>(loc, /*isAdj=*/true, target);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SOp decompositions
//===----------------------------------------------------------------------===//

// quake.s target
// ──────────────────────────────
// phased_rx(π/2, 0) target
// phased_rx(-π/2, π/2) target
// phased_rx(-π/2, 0) target
struct SToPhasedRx : public OpRewritePattern<quake::SOp> {
  using OpRewritePattern<quake::SOp>::OpRewritePattern;

  void initialize() { setDebugName("SToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::SOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);

    Value angle = op.isAdj() ? pi_2 : negPi_2;

    std::array<Value, 2> parameters = {pi_2, zero};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = angle;
    parameters[1] = pi_2;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.s [control] target
// ────────────────────────────────────
// quake.r1(π/2) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct SToR1 : public OpRewritePattern<quake::SOp> {
  using OpRewritePattern<quake::SOp>::OpRewritePattern;

  void initialize() { setDebugName("SToR1"); }

  LogicalResult matchAndRewrite(quake::SOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    auto loc = op->getLoc();
    auto angle = createConstant(loc, op.isAdj() ? -M_PI_2 : M_PI_2,
                                rewriter.getF64Type(), rewriter);
    rewriter.create<quake::R1Op>(loc, angle, op.getControls(), op.getTarget());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TOp decompositions
//===----------------------------------------------------------------------===//

// quake.t target
// ────────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π/4, π/2) target
// quake.phased_rx(-π/2, 0) target
struct TToPhasedRx : public OpRewritePattern<quake::TOp> {
  using OpRewritePattern<quake::TOp>::OpRewritePattern;

  void initialize() { setDebugName("TToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::TOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = createConstant(loc, -M_PI_4, rewriter.getF64Type(), rewriter);
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);

    std::array<Value, 2> parameters = {pi_2, zero};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = angle;
    parameters[1] = pi_2;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.t [control] target
// ────────────────────────────────────
// quake.r1(π/4) [control] target
//
// Adding this gate equivalence will enable further decomposition via other
// patterns such as controlled-r1 to cnot.
struct TToR1 : public OpRewritePattern<quake::TOp> {
  using OpRewritePattern<quake::TOp>::OpRewritePattern;

  void initialize() { setDebugName("TToR1"); }

  LogicalResult matchAndRewrite(quake::TOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    auto loc = op->getLoc();
    auto angle = createConstant(loc, op.isAdj() ? -M_PI_4 : M_PI_4,
                                rewriter.getF64Type(), rewriter);
    rewriter.create<quake::R1Op>(loc, angle, op.getControls(), op.getTarget());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// XOp decompositions
//===----------------------------------------------------------------------===//

// quake.x [control] target
// ──────────────────────────────────
// quake.h target
// quake.z [control] target
// quake.h target
struct CXToCZ : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern<quake::XOp>::OpRewritePattern;

  void initialize() { setDebugName("CXToCZ"); }

  LogicalResult matchAndRewrite(quake::XOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();
    if (failed(checkNumControls(op, 1)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    rewriter.create<quake::HOp>(loc, target);
    if (negControl)
      rewriter.create<quake::XOp>(loc, op.getControls());
    rewriter.create<quake::ZOp>(loc, op.getControls(), target);
    if (negControl)
      rewriter.create<quake::XOp>(loc, op.getControls());
    rewriter.create<quake::HOp>(loc, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.x [controls] target
// ──────────────────────────────────
// quake.h target
// quake.z [controls] target
// quake.h target
struct CCXToCCZ : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern<quake::XOp>::OpRewritePattern;

  void initialize() { setDebugName("CCXToCCZ"); }

  LogicalResult matchAndRewrite(quake::XOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();
    if (failed(checkNumControls(op, 2)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    rewriter.create<quake::HOp>(loc, target);
    auto zOp = rewriter.create<quake::ZOp>(loc, op.getControls(), target);
    zOp.setNegatedQubitControls(op.getNegatedQubitControls());
    rewriter.create<quake::HOp>(loc, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.x target
// ───────────────────────────────
// quake.phased_rx(π, 0) target
struct XToPhasedRx : public OpRewritePattern<quake::XOp> {
  using OpRewritePattern<quake::XOp>::OpRewritePattern;

  void initialize() { setDebugName("XToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::XOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value pi = createConstant(loc, M_PI, rewriter.getF64Type(), rewriter);

    ValueRange parameters = {pi, zero};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// YOp decompositions
//===----------------------------------------------------------------------===//

// quake.y target
// ─────────────────────────────────
// quake.phased_rx(π, -π/2) target
struct YToPhasedRx : public OpRewritePattern<quake::YOp> {
  using OpRewritePattern<quake::YOp>::OpRewritePattern;

  void initialize() { setDebugName("YToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::YOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value pi = createConstant(loc, M_PI, rewriter.getF64Type(), rewriter);
    Value negPi_2 =
        createConstant(loc, -M_PI_2, rewriter.getF64Type(), rewriter);

    ValueRange parameters = {pi, negPi_2};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ZOp decompositions
//===----------------------------------------------------------------------===//

//                                                                  ┌───┐
//  ───●────  ──────────────●───────────────────●──────●─────────●──┤ T ├
//     │                    │                   │      │         │  └───┘
//     │                    │                   │    ┌─┴─┐┌───┐┌─┴─┐┌───┐
//  ───●─── = ────●─────────┼─────────●─────────┼────┤ X ├┤ ┴ ├┤ X ├┤ T ├
//     │          │         │         │         │    └───┘└───┘└───┘└───┘
//   ┌─┴─┐      ┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐┌───┐┌─┴─┐                 ┌───┐
//  ─┤ z ├─   ──┤ X ├┤ ┴ ├┤ X ├┤ T ├┤ X ├┤ ┴ ├┤ X ├─────────────────┤ T ├
//   └───┘      └───┘└───┘└───┘└───┘└───┘└───┘└───┘                 └───┘
//
// NOTE: `┴` denotes the adjoint of `T`.
struct CCZToCX : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern<quake::ZOp>::OpRewritePattern;

  void initialize() { setDebugName("CCZToCX"); }

  LogicalResult matchAndRewrite(quake::ZOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    Value controls[2];
    if (failed(checkAndExtractControls(op, controls, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negC0 = false;
    auto negC1 = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls) {
      negC0 = (*negatedControls)[0];
      negC1 = (*negatedControls)[1];
      // The order of conrols don't matter for the operation. However, this
      // pattern relies on a normalization: if only one control is complemented,
      // it must be the 0th one, which means that a negated 1th control implies
      // a negated 0th. This normalization allow us to decompose more
      // straifghtforwardly.
      if (!negC0 && negC1) {
        negC0 = true;
        negC1 = false;
        std::swap(controls[0], controls[1]);
      }
    }

    rewriter.create<quake::XOp>(loc, controls[1], target);
    rewriter.create<quake::TOp>(loc, /*isAdj=*/!negC0, target);
    rewriter.create<quake::XOp>(loc, controls[0], target);
    rewriter.create<quake::TOp>(loc, target);
    rewriter.create<quake::XOp>(loc, controls[1], target);
    rewriter.create<quake::TOp>(loc, /*isAdj=*/!negC1, target);
    rewriter.create<quake::XOp>(loc, controls[0], target);
    rewriter.create<quake::TOp>(loc, /*isAdj=*/negC0 && !negC1, target);

    rewriter.create<quake::XOp>(loc, controls[0], controls[1]);
    rewriter.create<quake::TOp>(loc, /*isAdj=*/true, controls[1]);
    rewriter.create<quake::XOp>(loc, controls[0], controls[1]);
    rewriter.create<quake::TOp>(loc, /*isAdj=*/negC0, controls[1]);

    rewriter.create<quake::TOp>(loc, /*isAdj=*/negC1, controls[0]);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.z [control] target
// ──────────────────────────────────
// quake.h target
// quake.x [control] target
// quake.h target
struct CZToCX : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern<quake::ZOp>::OpRewritePattern;

  void initialize() { setDebugName("CZToCX"); }

  LogicalResult matchAndRewrite(quake::ZOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();
    if (failed(checkNumControls(op, 1)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    rewriter.create<quake::HOp>(loc, target);
    if (negControl)
      rewriter.create<quake::XOp>(loc, op.getControls());
    rewriter.create<quake::XOp>(loc, op.getControls(), target);
    if (negControl)
      rewriter.create<quake::XOp>(loc, op.getControls());
    rewriter.create<quake::HOp>(loc, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.z target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-π, π/2) target
// quake.phased_rx(-π/2, 0) target
struct ZToPhasedRx : public OpRewritePattern<quake::ZOp> {
  using OpRewritePattern<quake::ZOp>::OpRewritePattern;

  void initialize() { setDebugName("ZToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::ZOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, rewriter.getF64Type(), rewriter);
    Value negPi = createConstant(loc, -M_PI, rewriter.getF64Type(), rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, rewriter.getF64Type(), rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);

    std::array<Value, 2> parameters = {pi_2, zero};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi;
    parameters[1] = pi_2;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// R1Op decompositions
//===----------------------------------------------------------------------===//

// quake.r1(λ) [control] target
// ───────────────────────────────
// quake.r1(λ/2) control
// quake.x [control] target
// quake.r1(-λ/2) target
// quake.x [control] target
// quake.r1(λ/2) target
struct CR1ToCX : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern<quake::R1Op>::OpRewritePattern;

  void initialize() { setDebugName("CR1ToCX"); }

  LogicalResult matchAndRewrite(quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    ValueRange noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);

    rewriter.create<quake::R1Op>(loc, /*isAdj*/ negControl, halfAngle,
                                 noControls, control);
    rewriter.create<quake::XOp>(loc, control, target);
    rewriter.create<quake::R1Op>(loc, /*isAdj*/ negControl, negHalfAngle,
                                 noControls, target);
    rewriter.create<quake::XOp>(loc, control, target);
    rewriter.create<quake::R1Op>(loc, halfAngle, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.r1(λ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-λ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct R1ToPhasedRx : public OpRewritePattern<quake::R1Op> {
  using OpRewritePattern<quake::R1Op>::OpRewritePattern;

  void initialize() { setDebugName("R1ToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::R1Op op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);
    Value negAngle = rewriter.create<arith::NegFOp>(loc, angle);

    std::array<Value, 2> parameters = {pi_2, zero};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negAngle;
    parameters[1] = pi_2;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RxOp decompositions
//===----------------------------------------------------------------------===//

// quake.rx(θ) [control] target
// ───────────────────────────────
// quake.s target
// quake.x [control] target
// quake.ry(-θ/2) target
// quake.x [control] target
// quake.ry(θ/2) target
// quake.rz(-π/2) target
struct CRxToCX : public OpRewritePattern<quake::RxOp> {
  using OpRewritePattern<quake::RxOp>::OpRewritePattern;

  void initialize() { setDebugName("CRxToCX"); }

  LogicalResult matchAndRewrite(quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();
    assert(control);

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);
    Value negPI_2 = createConstant(loc, -M_PI_2, angleType, rewriter);

    rewriter.create<quake::SOp>(loc, /*isAdj*/ negControl, target);
    rewriter.create<quake::XOp>(loc, control, target);
    rewriter.create<quake::RyOp>(loc, negHalfAngle, noControls, target);
    rewriter.create<quake::XOp>(loc, control, target);
    rewriter.create<quake::RyOp>(loc, /*isAdj*/ negControl, halfAngle,
                                 noControls, target);
    rewriter.create<quake::RzOp>(loc, /*isAdj*/ negControl, negPI_2, noControls,
                                 target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.rx(θ) target
// ───────────────────────────────
// quake.phased_rx(θ, 0) target
struct RxToPhasedRx : public OpRewritePattern<quake::RxOp> {
  using OpRewritePattern<quake::RxOp>::OpRewritePattern;

  void initialize() { setDebugName("RxToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::RxOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);

    ValueRange parameters = {angle, zero};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RyOp decompositions
//===----------------------------------------------------------------------===//

// quake.ry(θ) [control] target
// ───────────────────────────────
// quake.ry(θ/2) target
// quake.x [control] target
// quake.ry(-θ/2) target
// quake.x [control] target
struct CRyToCX : public OpRewritePattern<quake::RyOp> {
  using OpRewritePattern<quake::RyOp>::OpRewritePattern;

  void initialize() { setDebugName("CRyToCX"); }

  LogicalResult matchAndRewrite(quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    ValueRange noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);

    rewriter.create<quake::RyOp>(loc, halfAngle, noControls, target);
    rewriter.create<quake::XOp>(loc, control, target);
    rewriter.create<quake::RyOp>(loc, /*isAdj*/ negControl, negHalfAngle,
                                 noControls, target);
    rewriter.create<quake::XOp>(loc, control, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.ry(θ) target
// ─────────────────────────────────
// quake.phased_rx(θ, π/2) target
struct RyToPhasedRx : public OpRewritePattern<quake::RyOp> {
  using OpRewritePattern<quake::RyOp>::OpRewritePattern;

  void initialize() { setDebugName("RyToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::RyOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);

    ValueRange parameters = {angle, pi_2};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RzOp decompositions
//===----------------------------------------------------------------------===//

// quake.rz(λ) [control] target
// ───────────────────────────────
// quake.rz(λ/2) target
// quake.x [control] target
// quake.rz(-λ/2) target
// quake.x [control] target
struct CRzToCX : public OpRewritePattern<quake::RzOp> {
  using OpRewritePattern<quake::RzOp>::OpRewritePattern;

  void initialize() { setDebugName("CRzToCX"); }

  LogicalResult matchAndRewrite(quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    if (!quake::isAllReferences(op))
      return failure();

    Value control;
    if (failed(checkAndExtractControls(op, control, rewriter)))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    auto negControl = false;
    auto negatedControls = op.getNegatedQubitControls();
    if (negatedControls)
      negControl = (*negatedControls)[0];

    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);

    // Necessary/Helpful constants
    ValueRange noControls;
    Value halfAngle = createDivF(loc, angle, 2.0, rewriter);
    Value negHalfAngle = rewriter.create<arith::NegFOp>(loc, halfAngle);

    rewriter.create<quake::RzOp>(loc, halfAngle, noControls, target);
    rewriter.create<quake::XOp>(loc, control, target);
    rewriter.create<quake::RzOp>(loc, /*isAdj*/ negControl, negHalfAngle,
                                 noControls, target);
    rewriter.create<quake::XOp>(loc, control, target);

    rewriter.eraseOp(op);
    return success();
  }
};

// quake.rz(θ) target
// ──────────────────────────────────
// quake.phased_rx(π/2, 0) target
// quake.phased_rx(-θ, π/2) target
// quake.phased_rx(-π/2, 0) target
struct RzToPhasedRx : public OpRewritePattern<quake::RzOp> {
  using OpRewritePattern<quake::RzOp>::OpRewritePattern;

  void initialize() { setDebugName("RzToPhasedRx"); }

  LogicalResult matchAndRewrite(quake::RzOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getControls().empty())
      return failure();
    if (!quake::isAllReferences(op))
      return failure();

    // Op info
    Location loc = op->getLoc();
    Value target = op.getTarget();
    Value angle = op.getParameter();
    if (op.isAdj())
      angle = rewriter.create<arith::NegFOp>(loc, angle);
    Type angleType = op.getParameter().getType();

    // Necessary/Helpful constants
    ValueRange noControls;
    Value zero = createConstant(loc, 0.0, angleType, rewriter);
    Value pi_2 = createConstant(loc, M_PI_2, angleType, rewriter);
    Value negPi_2 = rewriter.create<arith::NegFOp>(loc, pi_2);
    Value negAngle = rewriter.create<arith::NegFOp>(loc, angle);

    std::array<Value, 2> parameters = {pi_2, zero};
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negAngle;
    parameters[1] = pi_2;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);
    parameters[0] = negPi_2;
    parameters[1] = zero;
    rewriter.create<quake::PhasedRxOp>(loc, parameters, noControls, target);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Populating pattern sets
//===----------------------------------------------------------------------===//

void cudaq::populateWithAllDecompositionPatterns(RewritePatternSet &patterns) {
  // clang-format off
  patterns.insert<
    // HOp patterns
    HToPhasedRx,
    CHToCX,
    // SOp patterns
    SToPhasedRx,
    SToR1,
    // TOp patterns
    TToPhasedRx,
    TToR1,
    // XOp patterns
    CXToCZ,
    CCXToCCZ,
    XToPhasedRx,
    // YOp patterns
    YToPhasedRx,
    // ZOp patterns
    CZToCX,
    CCZToCX,
    ZToPhasedRx,
    // R1Op patterns
    CR1ToCX,
    R1ToPhasedRx,
    R1ToRz,
    // RxOp patterns
    CRxToCX,
    RxToPhasedRx,
    // RyOp patterns
    CRyToCX,
    RyToPhasedRx,
    // RzOp patterns
    CRzToCX,
    RzToPhasedRx,
    // Swap
    SwapToCX
  >(patterns.getContext());
  // clang-format on
}
