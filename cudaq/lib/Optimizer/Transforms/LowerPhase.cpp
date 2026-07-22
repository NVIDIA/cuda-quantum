/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_LOWERPHASE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

static SmallVector<bool> getControlPolarities(cudaq::quake::PhaseOp phase) {
  SmallVector<bool> polarities(phase.getControls().size(), false);
  if (auto negated = phase.getNegatedQubitControls()) {
    for (auto [index, value] : llvm::enumerate(*negated)) {
      if (index == polarities.size())
        break;
      polarities[index] = value;
    }
  }
  return polarities;
}

static DenseBoolArrayAttr makeNegatedControlsAttr(OpBuilder &builder,
                                                  ArrayRef<bool> polarities) {
  if (llvm::none_of(polarities, [](bool value) { return value; }))
    return {};
  return builder.getDenseBoolArrayAttr(polarities);
}

static SmallVector<Type> getWireResultTypes(MLIRContext *context,
                                            ValueRange controls, Value target) {
  auto wireType = cudaq::quake::WireType::get(context);
  SmallVector<Type> resultTypes;
  for (Value control : controls)
    if (isa<cudaq::quake::WireType>(control.getType()))
      resultTypes.push_back(wireType);
  if (isa<cudaq::quake::WireType>(target.getType()))
    resultTypes.push_back(wireType);
  return resultTypes;
}

template <typename Op>
static Op createParameterizedGate(IRRewriter &rewriter, Location location,
                                  Value parameter, ValueRange controls,
                                  Value target,
                                  DenseBoolArrayAttr negatedControls = {}) {
  auto resultTypes =
      getWireResultTypes(rewriter.getContext(), controls, target);
  return Op::create(rewriter, location, resultTypes, /*is_adj=*/false,
                    ValueRange{parameter}, controls, ValueRange{target},
                    negatedControls);
}

static cudaq::quake::XOp createXGate(IRRewriter &rewriter, Location location,
                                     Value target) {
  auto resultTypes =
      getWireResultTypes(rewriter.getContext(), ValueRange{}, target);
  return cudaq::quake::XOp::create(rewriter, location, resultTypes,
                                   /*is_adj=*/false, ValueRange{}, ValueRange{},
                                   ValueRange{target}, DenseBoolArrayAttr{});
}

/// Update the control and target values to the wire results of a newly created
/// Quake operator. Quake operator results are ordered by wire controls followed
/// by wire targets.
template <typename Op>
static void threadGateResults(Op gate, MutableArrayRef<Value> controls,
                              Value &target) {
  unsigned result = 0;
  for (Value &control : controls)
    if (isa<cudaq::quake::WireType>(control.getType()))
      control = gate.getWires()[result++];
  if (isa<cudaq::quake::WireType>(target.getType()))
    target = gate.getWires()[result++];
  assert(result == gate.getWires().size() &&
         "gate result count does not match its wire operands");
}

static void threadXResult(cudaq::quake::XOp gate, Value &target) {
  if (isa<cudaq::quake::WireType>(target.getType()))
    target = gate.getWires().front();
}

static SmallVector<Value> getPhaseReplacements(cudaq::quake::PhaseOp phase,
                                               ArrayRef<Value> threadedControls,
                                               Value threadedAnchor) {
  SmallVector<Value> replacements;
  for (Value control : threadedControls)
    if (isa<cudaq::quake::WireType>(control.getType()))
      replacements.push_back(control);
  if (isa<cudaq::quake::WireType>(threadedAnchor.getType()))
    replacements.push_back(threadedAnchor);
  assert(replacements.size() == phase.getWires().size() &&
         "phase result count does not match its wire operands");
  return replacements;
}

static Value normalizeAdjointAngle(IRRewriter &rewriter,
                                   cudaq::quake::PhaseOp phase) {
  Value angle = phase.getParameter();
  if (phase.isAdj())
    angle = arith::NegFOp::create(rewriter, phase.getLoc(), angle);
  return angle;
}

static bool isScalarGateTarget(Value value) {
  return isa<cudaq::quake::RefType, cudaq::quake::WireType>(value.getType());
}

static Value getReferenceAnchor(Value anchor) {
  if (auto unwrap = anchor.getDefiningOp<cudaq::quake::UnwrapOp>())
    return unwrap.getRefValue();
  return anchor;
}

static bool aggregateContainsReference(Value aggregate, Value reference) {
  if (aggregate == reference)
    return true;
  if (auto relax = aggregate.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    return aggregateContainsReference(relax.getInputVec(), reference);
  if (auto concat = aggregate.getDefiningOp<cudaq::quake::ConcatOp>())
    return llvm::any_of(concat.getTargets(), [&](Value member) {
      return aggregateContainsReference(member, reference);
    });
  if (auto struq = aggregate.getDefiningOp<cudaq::quake::MakeStruqOp>())
    return llvm::any_of(struq.getVeqs(), [&](Value member) {
      return aggregateContainsReference(member, reference);
    });
  return false;
}

/// Return true for the direct alias forms that the anchored fallback can
/// identify locally. The phase producer is otherwise responsible for choosing
/// an anchor outside the control predicate.
static bool isKnownAnchorControlAlias(Value anchor, Value control) {
  Value referenceAnchor = getReferenceAnchor(anchor);
  if (aggregateContainsReference(control, referenceAnchor))
    return true;
  auto extract = referenceAnchor.getDefiningOp<cudaq::quake::ExtractRefOp>();
  if (extract && aggregateContainsReference(control, extract.getVeq()))
    return true;
  auto member = referenceAnchor.getDefiningOp<cudaq::quake::GetMemberOp>();
  return member && aggregateContainsReference(control, member.getStruq());
}

static void lowerWithScalarControl(IRRewriter &rewriter,
                                   cudaq::quake::PhaseOp phase, Value angle,
                                   ArrayRef<bool> polarities,
                                   unsigned selectedControl) {
  SmallVector<Value> controls(phase.getControls().begin(),
                              phase.getControls().end());
  Value anchor = phase.getTarget();
  Location location = phase.getLoc();

  // The selected control becomes the R1 target. If it is negative, flip that
  // target around the R1 so that its |0> state selects the phase.
  bool flipSelected = polarities[selectedControl];
  if (flipSelected) {
    auto x = createXGate(rewriter, location, controls[selectedControl]);
    threadXResult(x, controls[selectedControl]);
  }

  SmallVector<Value> remainingControls;
  SmallVector<bool> remainingPolarities;
  SmallVector<unsigned> remainingIndices;
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (index == selectedControl)
      continue;
    remainingControls.push_back(control);
    remainingPolarities.push_back(polarities[index]);
    remainingIndices.push_back(index);
  }

  auto r1 = createParameterizedGate<cudaq::quake::R1Op>(
      rewriter, location, angle, remainingControls, controls[selectedControl],
      makeNegatedControlsAttr(rewriter, remainingPolarities));
  threadGateResults(r1, remainingControls, controls[selectedControl]);
  for (auto [position, index] : llvm::enumerate(remainingIndices))
    controls[index] = remainingControls[position];

  if (flipSelected) {
    auto x = createXGate(rewriter, location, controls[selectedControl]);
    threadXResult(x, controls[selectedControl]);
  }

  rewriter.replaceOp(phase, getPhaseReplacements(phase, controls, anchor));
}

static void lowerWithAnchorFallback(IRRewriter &rewriter,
                                    cudaq::quake::PhaseOp phase, Value angle,
                                    ArrayRef<bool> polarities) {
  SmallVector<Value> controls(phase.getControls().begin(),
                              phase.getControls().end());
  Value anchor = phase.getTarget();
  Location location = phase.getLoc();
  auto negatedControls = makeNegatedControlsAttr(rewriter, polarities);

  // R1(phi)^2 Rz(-phi)^2 = exp(i phi) I on the active branch. This four-gate
  // form is equivalent to R1(2 phi) Rz(-2 phi), but it cannot overflow a
  // finite source angle while forming the doubled parameters.
  Value negatedAngle = arith::NegFOp::create(rewriter, location, angle);
  for (unsigned i = 0; i != 2; ++i) {
    auto r1 = createParameterizedGate<cudaq::quake::R1Op>(
        rewriter, location, angle, controls, anchor, negatedControls);
    threadGateResults(r1, controls, anchor);
  }
  for (unsigned i = 0; i != 2; ++i) {
    auto rz = createParameterizedGate<cudaq::quake::RzOp>(
        rewriter, location, negatedAngle, controls, anchor, negatedControls);
    threadGateResults(rz, controls, anchor);
  }

  rewriter.replaceOp(phase, getPhaseReplacements(phase, controls, anchor));
}

static LogicalResult lowerPhase(IRRewriter &rewriter,
                                cudaq::quake::PhaseOp phase) {
  rewriter.setInsertionPoint(phase);

  if (phase.getControls().empty()) {
    SmallVector<Value> controls;
    rewriter.replaceOp(
        phase, getPhaseReplacements(phase, controls, phase.getTarget()));
    return success();
  }

  Value angle = normalizeAdjointAngle(rewriter, phase);
  SmallVector<bool> polarities = getControlPolarities(phase);

  // Any scalar control can become the R1 target while vector controls remain
  // in the predicate. Prefer the last positive scalar, then the last negative
  // scalar, to make the lowering deterministic.
  std::optional<unsigned> positiveScalar;
  std::optional<unsigned> scalarControl;
  for (auto [index, control] : llvm::enumerate(phase.getControls())) {
    if (!isScalarGateTarget(control))
      continue;
    scalarControl = index;
    if (!polarities[index])
      positiveScalar = index;
  }
  std::optional<unsigned> selected =
      positiveScalar ? positiveScalar : scalarControl;
  if (selected) {
    lowerWithScalarControl(rewriter, phase, angle, polarities, *selected);
    return success();
  }

  // A vector (or another non-targetable control representation) cannot serve
  // as R1's scalar target. The anchored identity is exact on the full active
  // control branch and preserves the complete ordered predicate.
  for (Value control : phase.getControls())
    if (isKnownAnchorControlAlias(phase.getTarget(), control)) {
      phase.emitOpError(
          "cannot lower with an anchor that aliases a control operand");
      return failure();
    }
  lowerWithAnchorFallback(rewriter, phase, angle, polarities);
  return success();
}

struct LowerPhasePass
    : public cudaq::opt::impl::LowerPhaseBase<LowerPhasePass> {
  using LowerPhaseBase::LowerPhaseBase;

  void runOnOperation() override {
    SmallVector<cudaq::quake::PhaseOp> phases;
    getOperation().walk(
        [&](cudaq::quake::PhaseOp phase) { phases.push_back(phase); });

    IRRewriter rewriter(&getContext());
    for (cudaq::quake::PhaseOp phase : phases)
      if (failed(lowerPhase(rewriter, phase))) {
        signalPassFailure();
        return;
      }
  }
};

} // namespace
