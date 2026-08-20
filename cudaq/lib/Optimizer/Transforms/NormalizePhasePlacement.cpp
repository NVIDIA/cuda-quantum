/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "QuakeOperatorUtilities.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <cassert>
#include <optional>

namespace cudaq::opt {
#define GEN_PASS_DEF_NORMALIZEPHASEPLACEMENT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// Wire values model linear dataflow and must have at most one live user while
/// being threaded. Reference-semantics values are not subject to that SSA-use
/// constraint; their possible aliasing is checked separately.
static bool hasUnambiguousWireUse(Value value) {
  if (!isa<cudaq::quake::WireType>(value.getType()))
    return true;
  return value.use_empty() || value.hasOneUse();
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

  // do not modify controls or anchor if any of them cannot be advanced!
  // create stage for controls and anchor and commit only after we've
  // verified that both the controls and anchor can be advanced past the
  // candidate op `op`

  SmallVector<Value> stagedControls(controls.begin(), controls.end());
  Value stagedAnchor = anchor;

  for (Value &control : stagedControls) {
    if (!hasUnambiguousWireUse(control))
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

  if (!hasUnambiguousWireUse(stagedAnchor))
    return failure();
  FailureOr<Value> threadedAnchor = getThreadedValue(op, stagedAnchor);
  if (failed(threadedAnchor))
    return failure();

  stagedAnchor = *threadedAnchor;

  // at this point, we've verified that all controls and anchors can safely be
  // moved past the candidate op

  for (unsigned i = 0; i < controls.size(); ++i)
    controls[i] = stagedControls[i];

  anchor = stagedAnchor;

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
  return true;
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
static cudaq::quake::PhaseOp sinkPhase(IRRewriter &rewriter,
                                       cudaq::quake::PhaseOp phase) {
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
    return phase;

  Operation *destination = phase->getNextNode();
  bool foundNewPlacement = false;
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

    foundNewPlacement = true;
    destination = cursor->getNextNode();
  }

  if (!foundNewPlacement)
    return phase;
  for (Value control : controls)
    if (!hasUnambiguousWireUse(control))
      return phase;
  if (!hasUnambiguousWireUse(anchor))
    return phase;

  if (destination)
    rewriter.setInsertionPoint(destination);
  else
    rewriter.setInsertionPointToEnd(phase->getBlock());

  auto resultTypes =
      cudaq::opt::getWireResultTypes(rewriter, controls, {anchor});
  auto moved = cudaq::quake::PhaseOp::create(
      rewriter, phase.getLoc(), resultTypes, phase.getIsAdjAttr(),
      phase.getParameters(), controls, ValueRange{anchor},
      phase.getNegatedQubitControlsAttr());

  SmallVector<Value> liveInputs(controls.begin(), controls.end());
  liveInputs.push_back(anchor);
  replaceLiveWireUses(liveInputs, moved);

  SmallVector<Value> oldInputs = cudaq::opt::getPhaseReplacements(
      phase, phase.getControls(), phase.getTarget());
  rewriter.replaceOp(phase, oldInputs);

  return moved;
}

static Value mapThroughPhase(cudaq::quake::PhaseOp phase, Value value);

/// Phase operations commute with one another. Pure classical operations may
/// appear between them after earlier merges and are transparent to the
/// quantum predicate.
static bool isTransparentBetweenPhases(Operation *operation) {
  return isa<cudaq::quake::PhaseOp>(operation) ||
         (!hasQuantumValue(operation) && isSafeToCross(operation));
}

/// Create a function that merges one group of same-type uncontrolled phases
/// Note that these phases ops can be merged _because_ they are uncontrolled
/// and are thus global phase shifts
static cudaq::quake::PhaseOp
mergeUncontrolledGroup(IRRewriter &rewriter,
                       ArrayRef<cudaq::quake::PhaseOp> uncontrolledPhases) {

  // input checks
  assert(!uncontrolledPhases.empty() &&
         "mergeUncontrolledGroup requires at least one phase");
  auto firstPhaseOp = uncontrolledPhases.front();
  auto angleType = firstPhaseOp.getParameter().getType();
  assert(llvm::all_of(uncontrolledPhases,
                      [&](auto phase) {
                        return phase.getControls().empty() &&
                               phase.getParameter().getType() == angleType;
                      }) &&
         "expected same-type uncontrolled phases");

  // this is the `quake.phase` that will accumulate all of the other phase
  // angles from `uncontrolledPhases`
  auto survivingPhase = uncontrolledPhases.back();

  rewriter.setInsertionPoint(survivingPhase);

  // add all of the phase angles together
  Value accumulatedAngle =
      cudaq::opt::getSignedAngle(rewriter, uncontrolledPhases.front());
  for (auto phaseOp : llvm::drop_begin(uncontrolledPhases, 1)) {
    accumulatedAngle =
        arith::AddFOp::create(rewriter, phaseOp.getLoc(), accumulatedAngle,
                              cudaq::opt::getSignedAngle(rewriter, phaseOp));
  }

  // remove all of the other uncontrolledPhases except for `survivingPhase`
  for (auto phaseOp : llvm::drop_end(uncontrolledPhases, 1)) {
    rewriter.replaceOp(
        phaseOp, cudaq::opt::getPhaseReplacements(
                     phaseOp, phaseOp.getControls(), phaseOp.getTarget()));
  }

  // create new `quake.phase` with accumulated angle and anchor
  // read this _after_ bypassing earlier phase ops; it might have been rewired
  auto survivingAnchor = survivingPhase.getTarget();

  auto resultTypes =
      cudaq::opt::getWireResultTypes(rewriter, {}, {survivingAnchor});

  auto mergedPhaseOp = cudaq::quake::PhaseOp::create(
      rewriter, survivingPhase.getLoc(), resultTypes,
      /*is_adj=*/false, ValueRange{accumulatedAngle}, ValueRange{},
      ValueRange{survivingAnchor}, DenseBoolArrayAttr{});

  rewriter.replaceOp(survivingPhase, mergedPhaseOp.getWires());

  return mergedPhaseOp;
}

/// iterate through the block to find batches of uncontrolled phases and then
/// merge those groups together.
static void batchUncontrolledPhases(IRRewriter &rewriter, Block &block) {
  using UncontrolledPhaseGroups =
      llvm::MapVector<Type, SmallVector<cudaq::quake::PhaseOp>>;

  Operation *cursor = block.empty() ? nullptr : &block.front();

  while (cursor) {
    // unsafe operations mark the boundary of a section and are not a
    // part of an uncontrolled phase group
    if (!isSafeToCross(cursor)) {
      cursor = cursor->getNextNode();
      continue;
    }

    // group all uncontrolled phases together between boundary of ops
    // that cannot be commuted with
    UncontrolledPhaseGroups groups;
    while (cursor && isSafeToCross(cursor)) {
      // if we find a phase op, add it to the appropriate group
      if (auto phaseOp = dyn_cast<cudaq::quake::PhaseOp>(cursor);
          phaseOp && phaseOp.getControls().empty()) {
        Type parameterType = phaseOp.getParameter().getType();
        groups[parameterType].push_back(phaseOp);
      }

      cursor = cursor->getNextNode();
    }

    // merge each group of uncontrolled phases together
    // 1. sink the last phase of the group initially
    // 2. then merge the rest of the phases of the group into the last phase
    for (auto &group : groups) {
      auto &groupPhases = group.second;
      auto representative = sinkPhase(rewriter, groupPhases.back());

      if (groupPhases.size() == 1)
        continue;

      // update last phase to newly sunken phase op
      groupPhases.back() = representative;

      // now merge all of 0..n-1 phases into the representative phase
      mergeUncontrolledGroup(rewriter, groupPhases);
    }

    if (cursor)
      cursor = cursor->getNextNode();
  }
}

/// check to see if two phase ops have the same controls
static bool haveSamePredicate(cudaq::quake::PhaseOp first,
                              cudaq::quake::PhaseOp second) {
  if (first.getControls().size() != second.getControls().size() ||
      first.getParameter().getType() != second.getParameter().getType() ||
      cudaq::opt::getControlPolarities(first) !=
          cudaq::opt::getControlPolarities(second))
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

/// Return the input that corresponds to a given output of a phase
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
  Value firstAngle = cudaq::opt::getSignedAngle(rewriter, first);
  Value secondAngle = cudaq::opt::getSignedAngle(rewriter, second);
  Value angle =
      arith::AddFOp::create(rewriter, second.getLoc(), firstAngle, secondAngle);

  SmallVector<Value> firstInputs = cudaq::opt::getPhaseReplacements(
      first, first.getControls(), first.getTarget());
  rewriter.replaceOp(first, firstInputs);

  rewriter.setInsertionPoint(second);
  SmallVector<Value> controls(second.getControls().begin(),
                              second.getControls().end());
  Value anchor = second.getTarget();
  auto resultTypes =
      cudaq::opt::getWireResultTypes(rewriter, controls, {anchor});
  auto merged = cudaq::quake::PhaseOp::create(
      rewriter, second.getLoc(), resultTypes, /*is_adj=*/false,
      ValueRange{angle}, controls, ValueRange{anchor},
      cudaq::opt::makeNegatedControlsAttr(
          rewriter, cudaq::opt::getControlPolarities(second)));

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
    // Find every block that initially contains at least one phase.
    SmallVector<Block *> phaseBlocks;
    SmallPtrSet<Block *, 8> seenBlocks;
    getOperation().walk([&](cudaq::quake::PhaseOp phase) {
      if (seenBlocks.insert(phase->getBlock()).second)
        phaseBlocks.push_back(phase->getBlock());
    });

    IRRewriter rewriter(&getContext());

    // Batch and sink all _uncontrolled_ phases first.
    for (Block *block : phaseBlocks)
      batchUncontrolledPhases(rewriter, *block);

    // Batching may erase/recreate PhaseOps, so take a fresh snapshot and sink
    // only the controlled phases with the existing algorithm.
    SmallVector<cudaq::quake::PhaseOp> controlledPhases;
    getOperation().walk([&](cudaq::quake::PhaseOp phase) {
      if (!phase.getControls().empty())
        controlledPhases.push_back(phase);
    });

    for (cudaq::quake::PhaseOp phase : controlledPhases)
      sinkPhase(rewriter, phase);

    // Keep the existing merger for now. It sees the merged uncontrolled
    // representatives and the normalized controlled phases.
    for (Block *block : phaseBlocks)
      mergeCompatiblePhases(rewriter, *block);
  }
};

} // namespace
