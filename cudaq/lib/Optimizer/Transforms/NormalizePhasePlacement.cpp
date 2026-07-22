/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_NORMALIZEPHASEPLACEMENT
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

static SmallVector<Value> getWireInputs(cudaq::quake::PhaseOp phase) {
  SmallVector<Value> inputs;
  for (Value control : phase.getControls())
    if (isa<cudaq::quake::WireType>(control.getType()))
      inputs.push_back(control);
  if (isa<cudaq::quake::WireType>(phase.getTarget().getType()))
    inputs.push_back(phase.getTarget());
  assert(inputs.size() == phase.getWires().size() &&
         "phase result count does not match its wire operands");
  return inputs;
}

static bool hasUnambiguousLinearUse(Value value) {
  return !isa<cudaq::quake::WireType>(value.getType()) || value.use_empty() ||
         value.hasOneUse();
}

/// Return the output wire corresponding to \p input, or the input itself when
/// it is not a wire. A failure means that the operator's positional wire
/// convention could not be proved.
static FailureOr<Value> getThreadedValue(cudaq::quake::OperatorInterface op,
                                         Value input) {
  if (!isa<cudaq::quake::WireType>(input.getType()))
    return input;

  unsigned result = 0;
  std::optional<Value> threaded;
  for (Value control : op.getControls()) {
    if (!isa<cudaq::quake::WireType>(control.getType()))
      continue;
    if (control == input) {
      if (threaded)
        return failure();
      threaded = op.getWires()[result];
    }
    ++result;
  }
  for (Value target : op.getTargets()) {
    if (!isa<cudaq::quake::WireType>(target.getType()))
      continue;
    if (target == input) {
      if (threaded)
        return failure();
      threaded = op.getWires()[result];
    }
    ++result;
  }
  if (result != op.getWires().size())
    return failure();
  return threaded.value_or(input);
}

static bool isUnaliasedScalarAlloca(Value reference) {
  if (!isa<cudaq::quake::RefType>(reference.getType()))
    return false;
  auto allocation = reference.getDefiningOp<cudaq::quake::AllocaOp>();
  if (!allocation)
    return false;
  for (Operation *user : reference.getUsers())
    if (isa<cudaq::quake::ConcatOp>(user))
      return false;
  return true;
}

/// Return true only when the two values are proved to denote different
/// qubits. Distinct wire SSA values are precise. For reference semantics, use
/// the same deliberately narrow direct-alloca rule as PhaseFolding; function
/// arguments, extracts, and concatenated references may alias.
static bool canProveControlDisjointFromTarget(Value control, Value target) {
  if (control == target)
    return false;
  if (isa<cudaq::quake::ControlType>(control.getType()))
    return true;
  if (isa<cudaq::quake::WireType>(control.getType()))
    return isa<cudaq::quake::WireType>(target.getType());
  if (!isa<cudaq::quake::RefType>(control.getType()) ||
      !isa<cudaq::quake::RefType>(target.getType()))
    return false;
  return isUnaliasedScalarAlloca(control) && isUnaliasedScalarAlloca(target);
}

static bool mayTargetPhaseControl(cudaq::quake::OperatorInterface op,
                                  Value control) {
  return llvm::any_of(op.getTargets(), [&](Value target) {
    return !canProveControlDisjointFromTarget(control, target);
  });
}

/// Advance the phase's live controls and anchor through a known unitary
/// operator. A regular operator may use a phase control as a control, but may
/// not target it. PhaseOp's target is only an anchor, so another PhaseOp never
/// counts as changing a control.
static LogicalResult advanceAcrossOperator(cudaq::quake::OperatorInterface op,
                                           MutableArrayRef<Value> controls,
                                           Value &anchor) {
  bool bookkeepingPhase = isa<cudaq::quake::PhaseOp>(op.getOperation());

  for (Value &control : controls) {
    if (!hasUnambiguousLinearUse(control))
      return failure();
    // Without element-level alias information, a composite control might
    // overlap any target of an intervening operator. Statically sized vectors
    // are expected to have been expanded before this pass.
    if (!bookkeepingPhase &&
        isa<cudaq::quake::VeqType, cudaq::quake::StruqType>(control.getType()))
      return failure();
    if (!bookkeepingPhase && mayTargetPhaseControl(op, control))
      return failure();
    FailureOr<Value> threaded = getThreadedValue(op, control);
    if (failed(threaded))
      return failure();
    control = *threaded;
  }

  if (!hasUnambiguousLinearUse(anchor))
    return failure();
  FailureOr<Value> threadedAnchor = getThreadedValue(op, anchor);
  if (failed(threadedAnchor))
    return failure();
  anchor = *threadedAnchor;
  return success();
}

static bool hasQuantumValue(Operation *operation) {
  return llvm::any_of(operation->getOperandTypes(),
                      cudaq::quake::isQuantumType) ||
         llvm::any_of(operation->getResultTypes(), cudaq::quake::isQuantumType);
}

/// Calls, regions, terminators, non-unitary quantum operations, and operations
/// with unknown effects delimit safe straight-line placement sections.
static bool isSafeToCross(Operation *operation) {
  if (operation->hasTrait<OpTrait::IsTerminator>() ||
      operation->getNumRegions() != 0 || isa<CallOpInterface>(operation))
    return false;
  if (isa<cudaq::quake::OperatorInterface>(operation))
    return true;
  if (hasQuantumValue(operation))
    return false;
  return isMemoryEffectFree(operation);
}

static void replaceLiveWireUses(ValueRange inputs,
                                cudaq::quake::PhaseOp replacement) {
  unsigned result = 0;
  for (Value input : inputs) {
    if (!isa<cudaq::quake::WireType>(input.getType()))
      continue;
    input.replaceAllUsesExcept(replacement.getWires()[result++], replacement);
  }
  assert(result == replacement.getWires().size() &&
         "replacement phase result count mismatch");
}

/// Sink a phase to the end of the safe section that follows it. This routine
/// recreates the operation at its destination, forwards the old identity wire
/// results, and threads the new result through every live wire position.
static void sinkPhase(IRRewriter &rewriter, cudaq::quake::PhaseOp phase) {
  SmallVector<Value> controls;
  unsigned result = 0;
  for (Value control : phase.getControls()) {
    if (isa<cudaq::quake::WireType>(control.getType()))
      controls.push_back(phase.getWires()[result++]);
    else
      controls.push_back(control);
  }

  Value anchor = phase.getTarget();
  if (isa<cudaq::quake::WireType>(anchor.getType()))
    anchor = phase.getWires()[result++];
  if (result != phase.getWires().size())
    return;

  Operation *destination = phase->getNextNode();
  bool crossedOperation = false;
  for (Operation *cursor = phase->getNextNode(); cursor;
       cursor = cursor->getNextNode()) {
    if (!isSafeToCross(cursor)) {
      destination = cursor;
      break;
    }

    if (auto quantum = dyn_cast<cudaq::quake::OperatorInterface>(cursor))
      if (failed(advanceAcrossOperator(quantum, controls, anchor))) {
        destination = cursor;
        break;
      }

    crossedOperation = true;
    destination = cursor->getNextNode();
  }

  if (!crossedOperation)
    return;
  for (Value control : controls)
    if (!hasUnambiguousLinearUse(control))
      return;
  if (!hasUnambiguousLinearUse(anchor))
    return;

  if (destination)
    rewriter.setInsertionPoint(destination);
  else
    rewriter.setInsertionPointToEnd(phase->getBlock());

  auto resultTypes =
      getWireResultTypes(rewriter.getContext(), controls, anchor);
  auto moved = cudaq::quake::PhaseOp::create(
      rewriter, phase.getLoc(), resultTypes, phase.getIsAdjAttr(),
      phase.getParameters(), controls, ValueRange{anchor},
      phase.getNegatedQubitControlsAttr());

  SmallVector<Value> liveInputs(controls.begin(), controls.end());
  liveInputs.push_back(anchor);
  replaceLiveWireUses(liveInputs, moved);

  SmallVector<Value> oldInputs = getWireInputs(phase);
  rewriter.replaceOp(phase, oldInputs);
}

static Value mapThroughPhase(cudaq::quake::PhaseOp phase, Value value);

/// Phase operations commute with one another. Pure classical operations may
/// appear between them after earlier merges and are transparent to the
/// quantum predicate.
static bool isTransparentBetweenPhases(Operation *operation) {
  return isa<cudaq::quake::PhaseOp>(operation) ||
         (!hasQuantumValue(operation) && isSafeToCross(operation));
}

static bool haveSamePredicate(cudaq::quake::PhaseOp first,
                              cudaq::quake::PhaseOp second) {
  if (first.getControls().size() != second.getControls().size() ||
      first.getParameter().getType() != second.getParameter().getType() ||
      getControlPolarities(first) != getControlPolarities(second))
    return false;

  SmallVector<Value> secondControls(second.getControls().begin(),
                                    second.getControls().end());
  Operation *cursor = second->getPrevNode();
  for (; cursor && cursor != first.getOperation();
       cursor = cursor->getPrevNode()) {
    if (!isTransparentBetweenPhases(cursor))
      return false;
    if (auto phase = dyn_cast<cudaq::quake::PhaseOp>(cursor))
      for (Value &control : secondControls)
        control = mapThroughPhase(phase, control);
  }
  if (cursor != first.getOperation())
    return false;

  unsigned firstResult = 0;
  for (auto [firstControl, secondControl] :
       llvm::zip(first.getControls(), secondControls)) {
    if (isa<cudaq::quake::WireType>(firstControl.getType())) {
      if (!isa<cudaq::quake::WireType>(secondControl.getType()) ||
          secondControl != first.getWires()[firstResult++])
        return false;
      continue;
    }
    if (firstControl != secondControl)
      return false;
  }
  return true;
}

static Value mapThroughPhase(cudaq::quake::PhaseOp phase, Value value) {
  unsigned result = 0;
  for (Value input : phase.getControls()) {
    if (!isa<cudaq::quake::WireType>(input.getType()))
      continue;
    if (value == phase.getWires()[result])
      return input;
    ++result;
  }
  Value anchor = phase.getTarget();
  if (isa<cudaq::quake::WireType>(anchor.getType()) &&
      value == phase.getWires()[result])
    return anchor;
  return value;
}

static Value getSignedAngle(IRRewriter &rewriter, cudaq::quake::PhaseOp phase) {
  Value angle = phase.getParameter();
  if (phase.isAdj())
    angle = arith::NegFOp::create(rewriter, phase.getLoc(), angle);
  return angle;
}

/// Merge compatible corrections after placement. They may be separated by
/// other phase operations because all such corrections are diagonal and
/// commute. The merged operation stays at the later correction, uses its live
/// anchor, and bypasses the earlier identity anchor positionally.
static FailureOr<cudaq::quake::PhaseOp>
mergePair(IRRewriter &rewriter, cudaq::quake::PhaseOp first,
          cudaq::quake::PhaseOp second) {
  if (!haveSamePredicate(first, second))
    return failure();

  // Sinking normally places all classical angle definitions before the phase
  // run. If ambiguous wire use prevented that move, do not create an invalid
  // use by hoisting the later angle above its definition.
  if (Operation *definition = second.getParameter().getDefiningOp();
      definition && definition->getBlock() == first->getBlock() &&
      !definition->isBeforeInBlock(first))
    return failure();

  // Emit all classical angle arithmetic before the normalized phase run. This
  // keeps a second application of the pass from moving an earlier phase past
  // arithmetic introduced by the first application.
  rewriter.setInsertionPoint(first);
  Value firstAngle = getSignedAngle(rewriter, first);
  Value secondAngle = getSignedAngle(rewriter, second);
  Value angle =
      arith::AddFOp::create(rewriter, second.getLoc(), firstAngle, secondAngle);

  SmallVector<Value> firstInputs = getWireInputs(first);
  rewriter.replaceOp(first, firstInputs);

  rewriter.setInsertionPoint(second);
  SmallVector<Value> controls(second.getControls().begin(),
                              second.getControls().end());
  Value anchor = second.getTarget();
  auto resultTypes =
      getWireResultTypes(rewriter.getContext(), controls, anchor);
  auto merged = cudaq::quake::PhaseOp::create(
      rewriter, second.getLoc(), resultTypes, /*is_adj=*/false,
      ValueRange{angle}, controls, ValueRange{anchor},
      makeNegatedControlsAttr(rewriter, getControlPolarities(second)));

  rewriter.replaceOp(second, merged.getWires());
  return merged;
}

static void mergeCompatiblePhases(IRRewriter &rewriter, Block &block) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &operation : llvm::make_early_inc_range(block)) {
      auto first = dyn_cast<cudaq::quake::PhaseOp>(&operation);
      if (!first)
        continue;

      for (Operation *cursor = first->getNextNode(); cursor;
           cursor = cursor->getNextNode()) {
        if (!isTransparentBetweenPhases(cursor))
          break;
        auto second = dyn_cast<cudaq::quake::PhaseOp>(cursor);
        if (!second)
          continue;
        if (succeeded(mergePair(rewriter, first, second))) {
          changed = true;
          break;
        }
      }
      if (changed)
        break;
    }
  }
}

struct NormalizePhasePlacementPass
    : public cudaq::opt::impl::NormalizePhasePlacementBase<
          NormalizePhasePlacementPass> {
  using NormalizePhasePlacementBase::NormalizePhasePlacementBase;

  void runOnOperation() override {
    SmallVector<cudaq::quake::PhaseOp> phases;
    SmallVector<Block *> phaseBlocks;
    SmallPtrSet<Block *, 8> seenBlocks;
    getOperation().walk([&](cudaq::quake::PhaseOp phase) {
      phases.push_back(phase);
      if (seenBlocks.insert(phase->getBlock()).second)
        phaseBlocks.push_back(phase->getBlock());
    });

    IRRewriter rewriter(&getContext());
    for (cudaq::quake::PhaseOp phase : phases)
      sinkPhase(rewriter, phase);

    for (Block *block : phaseBlocks)
      mergeCompatiblePhases(rewriter, *block);
  }
};

} // namespace
