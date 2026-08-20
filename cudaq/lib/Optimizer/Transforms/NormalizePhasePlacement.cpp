/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "PhaseUtilities.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <algorithm>
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
      cudaq::quake::getWireResultTypes(rewriter, controls, {anchor});
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

/// Distinguishes real predicates from the sentinel keys required by DenseMap.
enum class PhasePredicateStorageKind { Normal, Empty, Tombstone };

/// A canonical conditional-phase predicate used as an active-merger map key.
/// The target is deliberately not part of the key: it is an ordering and
/// wire-flow anchor, not part of the conditional phase predicate.
struct PhasePredicate {
  Type parameterType;
  SmallVector<Value> controls;
  SmallVector<bool> polarities;
  PhasePredicateStorageKind kind = PhasePredicateStorageKind::Normal;

  /// Compare semantic fields for real keys and storage kind for sentinels.
  bool operator==(const PhasePredicate &other) const {
    if (kind != other.kind)
      return false;
    if (kind != PhasePredicateStorageKind::Normal)
      return true;
    return parameterType == other.parameterType && controls == other.controls &&
           polarities == other.polarities;
  }
};

/// Supplies DenseMap with sentinel keys, hashing, and equality for predicates.
struct PhasePredicateInfo {
  /// Return the sentinel key for an unused DenseMap bucket.
  static PhasePredicate getEmptyKey() {
    PhasePredicate key;
    key.kind = PhasePredicateStorageKind::Empty;
    return key;
  }

  /// Return the sentinel key for a deleted DenseMap bucket.
  static PhasePredicate getTombstoneKey() {
    PhasePredicate key;
    key.kind = PhasePredicateStorageKind::Tombstone;
    return key;
  }

  /// Return a hash that includes every semantic predicate field.
  static unsigned getHashValue(const PhasePredicate &predicate) {
    if (predicate.kind != PhasePredicateStorageKind::Normal)
      return static_cast<unsigned>(predicate.kind);

    assert(predicate.controls.size() == predicate.polarities.size() &&
           "every phase control must have a polarity");
    auto hash = llvm::hash_combine(predicate.parameterType);
    for (auto [control, polarity] :
         llvm::zip(predicate.controls, predicate.polarities))
      hash = llvm::hash_combine(hash, control, polarity);
    return static_cast<unsigned>(hash);
  }

  /// Return whether two keys denote the same predicate or sentinel.
  static bool isEqual(const PhasePredicate &lhs, const PhasePredicate &rhs) {
    return lhs == rhs;
  }
};

/// Maps phase-produced wire results to their canonical inputs in one section.
using WireAliases = llvm::DenseMap<Value, Value>;
/// Maps original operations to their stable rank within one block.
using OperationRanks = llvm::DenseMap<Operation *, unsigned>;

/// A merge candidate and its original block rank.
struct ActivePhase {
  cudaq::quake::PhaseOp phase;
  unsigned phaseRank;
};

/// Groups active merge candidates by their canonical phase predicate.
using ActivePhaseMap = llvm::DenseMap<PhasePredicate, SmallVector<ActivePhase>,
                                      PhasePredicateInfo>;

/// Return a wire's canonical representative and compress its alias path.
static Value canonicalizePhaseWire(Value value, WireAliases &aliases) {
  if (!isa<cudaq::quake::WireType>(value.getType()))
    return value;

  auto alias = aliases.find(value);
  if (alias == aliases.end())
    return value;

  Value canonical = alias->second;
  auto next = aliases.find(canonical);
  while (next != aliases.end()) {
    canonical = next->second;
    next = aliases.find(canonical);
  }

  alias->second = canonical;
  return canonical;
}

/// Build a predicate key by canonicalizing the phase's wire controls.
/// The target is intentionally excluded because it is only a placement anchor.
static PhasePredicate getCanonicalPhasePredicate(cudaq::quake::PhaseOp phase,
                                                 WireAliases &aliases) {
  PhasePredicate predicate;
  predicate.parameterType = phase.getParameter().getType();
  predicate.controls.reserve(phase.getControls().size());
  for (Value control : phase.getControls())
    predicate.controls.push_back(canonicalizePhaseWire(control, aliases));
  predicate.polarities = cudaq::opt::getControlPolarities(phase);
  return predicate;
}

/// Record the identity wire flow of a phase correction. This is the forward
/// counterpart to mapping a later control backward through earlier phases.
static void recordPhaseWireAliases(cudaq::quake::PhaseOp phase,
                                   WireAliases &aliases) {
  unsigned result = 0;
  auto record = [&](Value input) {
    if (!isa<cudaq::quake::WireType>(input.getType()))
      return;

    assert(result < phase.getWires().size() &&
           "phase wire result count mismatch");
    aliases[phase.getWires()[result++]] = canonicalizePhaseWire(input, aliases);
  };

  for (Value control : phase.getControls())
    record(control);
  record(phase.getTarget());

  assert(result == phase.getWires().size() &&
         "phase wire result count mismatch");
}

/// Assign stable, increasing ranks to the operations currently in a block.
static OperationRanks rankBlockOperations(Block &block) {
  OperationRanks ranks;
  unsigned rank = 1;
  for (Operation &operation : block)
    ranks[&operation] = rank++;
  return ranks;
}

/// Return a phase's stable rank from the block snapshot.
static unsigned getPhaseRank(cudaq::quake::PhaseOp phase,
                             const OperationRanks &ranks) {
  auto rank = ranks.find(phase.getOperation());
  assert(rank != ranks.end() && "phase must belong to the ranked block");
  return rank->second;
}

/// Return the earliest block rank at which a phase parameter is available.
/// A block argument or a definition outside this block is available before
/// every operation in the block.
static unsigned getAngleAvailabilityRank(cudaq::quake::PhaseOp phase,
                                         const OperationRanks &ranks) {
  Operation *definition = phase.getParameter().getDefiningOp();
  if (!definition || definition->getBlock() != phase->getBlock())
    return 0;

  auto rank = ranks.find(definition);
  assert(rank != ranks.end() &&
         "parameter definition must belong to the ranked block");
  return rank->second;
}

/// Merge phases whose canonical predicates already match. The caller proves
/// that the second angle is available before the first phase.
static cudaq::quake::PhaseOp
mergeKnownCompatiblePair(IRRewriter &rewriter, cudaq::quake::PhaseOp first,
                         cudaq::quake::PhaseOp second) {
  // Emit angle arithmetic before the normalized phase run. This keeps the
  // transform idempotent when it is applied again.
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
      cudaq::quake::getWireResultTypes(rewriter, controls, {anchor});
  auto merged = cudaq::quake::PhaseOp::create(
      rewriter, second.getLoc(), resultTypes, /*is_adj=*/false,
      ValueRange{angle}, controls, ValueRange{anchor},
      cudaq::opt::makeNegatedControlsAttr(
          rewriter, cudaq::quake::getControlPolarities(second)));

  rewriter.replaceOp(second, merged.getWires());
  return merged;
}

/// Merge compatible corrections with one forward scan. For each predicate,
/// active phases remain ordered by their original block positions. A later
/// phase merges the suffix whose angle definitions it can legally hoist.
static void mergeCompatiblePhases(IRRewriter &rewriter, Block &block) {
  OperationRanks ranks = rankBlockOperations(block);
  WireAliases aliases;
  ActivePhaseMap active;

  Operation *cursor = block.empty() ? nullptr : &block.front();
  while (cursor) {
    Operation *next = cursor->getNextNode();

    if (!isTransparentBetweenPhases(cursor)) {
      aliases.clear();
      active.clear();
      cursor = next;
      continue;
    }

    auto phase = dyn_cast<cudaq::quake::PhaseOp>(cursor);
    if (!phase) {
      cursor = next;
      continue;
    }

    PhasePredicate predicate = getCanonicalPhasePredicate(phase, aliases);
    unsigned phaseRank = getPhaseRank(phase, ranks);
    unsigned availabilityRank = getAngleAvailabilityRank(phase, ranks);
    auto &representatives = active[predicate];

    assert(std::is_sorted(representatives.begin(), representatives.end(),
                          [](const ActivePhase &lhs, const ActivePhase &rhs) {
                            return lhs.phaseRank < rhs.phaseRank;
                          }) &&
           "active phase representatives must be in block order");

    // A representative at rank R can absorb this phase exactly when this
    // phase's parameter was defined before R. The first merge places the
    // accumulated angle before its representative, so it also dominates every
    // following representative in the sorted suffix.
    auto firstEligible = std::upper_bound(
        representatives.begin(), representatives.end(), availabilityRank,
        [](unsigned availability, const ActivePhase &candidate) {
          return availability < candidate.phaseRank;
        });

    auto current = phase;
    for (auto iterator = firstEligible; iterator != representatives.end();
         ++iterator) {
      current = mergeKnownCompatiblePair(rewriter, iterator->phase, current);
    }

    representatives.erase(firstEligible, representatives.end());
    representatives.push_back({current, phaseRank});
    recordPhaseWireAliases(current, aliases);

    cursor = next;
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

    // Merge normalized phase corrections with a single forward sweep.
    for (Block *block : phaseBlocks)
      mergeCompatiblePhases(rewriter, *block);
  }
};

} // namespace
