/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/NewUnitaryOpGrouping.h"
#include "QubitIdentityAnalysis.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <queue>

#define DEBUG_TYPE "new-unitary-op-grouping-analysis"

using namespace mlir;

namespace cudaq::quake::detail {

/// Helper to mark unitary, measurement, and reset ops
/// Returns std::nullopt if the op is not one of these three types
/// - it is instead a segment boundary
static std::optional<SegmentOpRole> classifySegmentOpRole(Operation *op) {
  // ops with nested regions are hard boundaries and should not be grouped
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return std::nullopt;

  if (isa<cudaq::quake::MeasurementInterface>(op))
    return SegmentOpRole::MsmtDelimiter;

  // Reset has `QuantumGate` trait but it is not a unitary op
  if (isa<cudaq::quake::ResetOp>(op))
    return SegmentOpRole::ResetDelimiter;

  if (op->hasTrait<cudaq::QuantumGate>())
    return SegmentOpRole::Unitary;

  return std::nullopt;
}

/// The following are helpers to construct a dependency graph
/// Once a candidate segment of ops is identified in the current block,
/// we construct a segment dependency graph

/// Returns true if the op is in the segment
/// - remember only unitaries, msmts, and resets will be in the segment
static bool containsNode(const SegmentDependencyGraph &sdg, Operation *op) {
  return sdg.originalPositionByOp.contains(op);
}

/// Add a node to graph if not already present
static void addNode(SegmentDependencyGraph &sdg, Operation *op,
                    unsigned originalPos) {
  // classify role of op in segment
  std::optional<SegmentOpRole> role = classifySegmentOpRole(op);
  assert(role && "op must be of `SegmentOpRole` type");
  assert(!containsNode(sdg, op) && "node added more than once");

  // update sdg with new node
  sdg.nodesInBlockOrder.push_back(op);
  sdg.originalPositionByOp.try_emplace(op, originalPos);

  // Initialize slots for successors and predecessor count
  sdg.successorsByOp.try_emplace(op);
  sdg.predecessorCountByOp.try_emplace(op, 0);
}

/// Determine the `OrderingMode` for creating dependency sdg adj list
static OrderingMode determineOrderingMode(QuantumOpSegment &segment,
                                          QubitIdentityAnalysis &qia) {
  // iterate through the entire segment. if get scalar wire flow is null, go to
  // textual form if wireflow is good, check all inputs from wireflow to make
  // sure each quantum input has a qubit ID i.e., known identity

  for (auto &op : segment.opsInBlockOrder) {
    std::optional<ScalarWireFlow> flow = getScalarWireFlow(op);
    // if not valid wire flow or not scalar, fall back to textual IR ordering
    if (!flow)
      return OrderingMode::Textual;

    // if any of the wires have unknown identity (i.e., no QubitID),
    // fall back to textual IR ordering
    for (Value &quantumOperand : flow->inputs)
      if (!qia.getQubitId(quantumOperand))
        return OrderingMode::Textual;
  }

  return OrderingMode::WireDataflow;
}

/// If edge doesn't exist, add edge to adjacency list and increment
/// predCountByOp
static void addEdge(SegmentDependencyGraph &sdg, Operation *predecessor,
                    Operation *successor) {
  assert(containsNode(sdg, predecessor) && "predecessor node must be in graph");
  assert(containsNode(sdg, successor) && "successor node must be in graph");

  auto &successors = sdg.successorsByOp[predecessor];

  // don't duplicate edges!
  if (llvm::is_contained(successors, successor))
    return;

  successors.push_back(successor);

  ++sdg.predecessorCountByOp[successor];
}

/// Populate `successorsByOp` and update `predecessorCountByOp`
/// This function is for use with the `WireDataFlow` ordering mode
static void addIntraSegmentDefUseEdges(SegmentDependencyGraph &sdg) {
  for (Operation *consumer : sdg.nodesInBlockOrder) {
    for (Value operand : consumer->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (producer && containsNode(sdg, producer))
        addEdge(sdg, producer, consumer);
    }
  }
}

/// Add edges based on QubitIdentityAnalysis. This additional pass over the
/// segment ops accounts for any known edges that may come about from
/// definitions outside of the segment ops but within the containing block.
/// This function is for use with the `WireDataFlow` ordering mode
static void addQubitIdentityEdges(SegmentDependencyGraph &sdg,
                                  QubitIdentityAnalysis &qia) {

  // map to tell us what operation last touched the Qubit ID
  mlir::DenseMap<QubitIdentityAnalysis::QubitId, Operation *>
      lastTouchByQubitId;
  for (Operation *consumer : sdg.nodesInBlockOrder) {
    std::optional<ScalarWireFlow> flow = getScalarWireFlow(consumer);
    assert(flow && "requires valid scalar wire data flow");

    // a set of all of the Qubit IDs that have been touched by the current op
    // this set spans our check of the consumer's inputs
    llvm::SmallDenseSet<QubitIdentityAnalysis::QubitId, 4> touchedByThisOp;

    for (Value operand : flow->inputs) {
      // grab the qubit ID for each operand
      std::optional<QubitIdentityAnalysis::QubitId> qid =
          qia.getQubitId(operand);
      assert(qid && "all wires must have QubitID");

      // if this op has already seen this qubit ID, ignore
      if (!touchedByThisOp.insert(*qid).second)
        continue;

      // if this op hasn't seen this qubit ID, let's check to see if an op
      // in the containing block has touched it by consulting
      // lastTouchByQubitId.
      // if there's a key-value pair, then add an edge between that
      // producer and consumer
      auto producer = lastTouchByQubitId.find(*qid);
      if (producer != lastTouchByQubitId.end())
        addEdge(sdg, producer->second, consumer);
      // regardless of if there was a def'n or not, let's update
      // `lastTouchByQubitId` with this op that is consuming the operand
      lastTouchByQubitId[*qid] = consumer;
    }
  }
}

/// This function is for use with the `Textual` ordering mode
/// In this case, we simply add edges between consecutive ops to
/// preserve block order
/// NOTE: this function is actually unnecessary because
static void addConsecutiveIREdges(SegmentDependencyGraph &sdg) {
  SmallVectorImpl<Operation *> &opNodes = sdg.nodesInBlockOrder;
  for (std::size_t i = 1; i < opNodes.size(); ++i)
    addEdge(sdg, opNodes[i - 1], opNodes[i]);
}

/// Create a dependency graph based on the input QuantumOpSegment
static SegmentDependencyGraph
buildSegmentDependencyGraph(QuantumOpSegment &segment,
                            QubitIdentityAnalysis &qia) {
  SegmentDependencyGraph sdg;
  sdg.containingBlock = segment.containingBlock;

  // create nodes in the sdg
  for (auto [opIdxInBlock, op] : llvm::enumerate(segment.opsInBlockOrder)) {
    addNode(sdg, op, opIdxInBlock);
  }

  // determine ordering mode
  sdg.mode = determineOrderingMode(segment, qia);

  // depending on graph mode use wireflow or textual IR order for ordering
  switch (sdg.mode) {
  case OrderingMode::WireDataflow:
    addIntraSegmentDefUseEdges(sdg);
    addQubitIdentityEdges(sdg, qia);
    break;
  case OrderingMode::Textual:
    addConsecutiveIREdges(sdg);
    break;
  default:
    // same as textual mode
    addConsecutiveIREdges(sdg);
    break;
  }

  return sdg;
}

// "Ready" means that all predecessors of this op have been added to
// canonical order.
struct ReadyEntry {
  unsigned originalPos; // break ties between two ready ops
  Operation *op;
};

// returns true if lhs has lower priority
// in this case, lhs has lower priority if its originalPos > rhs
struct EarliestOriginalPositionFirst {
  bool operator()(const ReadyEntry &lhs, const ReadyEntry &rhs) const {
    return lhs.originalPos > rhs.originalPos;
  }
};

// Maintain a min heap to determine which op should go next in
// canonical ordering
using ReadyMinHeap = std::priority_queue<ReadyEntry, SmallVector<ReadyEntry>,
                                         EarliestOriginalPositionFirst>;

/// earliest means the op that shows up first in IR block order
static Operation *popEarliestReadyEntry(ReadyMinHeap &rmh) {
  assert(!rmh.empty() && "ReadyMinHeap must have an entry to pop!");
  Operation *earliestEntry = rmh.top().op;
  rmh.pop();
  return earliestEntry;
}

CanonicalSegmentOrder
NewUnitaryOpGroupingAnalysis::computeCanonicalSegmentOrder(
    SegmentDependencyGraph &sdg) {
  CanonicalSegmentOrder cso;
  cso.containingBlock = sdg.containingBlock;
  cso.mode = sdg.mode;

  // if mode is textual fallback, just assign
  // nodesInBlockOrder and assign that to opsInCanonicalOrder
  if (cso.mode == OrderingMode::Textual) {
    cso.opsInCanonicalOrder = sdg.nodesInBlockOrder;
    return cso;
  }

  // maintain two min heaps to pick from:
  ReadyMinHeap readyUnitaries;
  ReadyMinHeap readyDelimiters;

  // lambda to add an op to the appropriate min heap
  auto addReady = [&](Operation *op) {
    unsigned origPos = sdg.originalPositionByOp.lookup(op);
    ReadyEntry re{origPos, op};

    SegmentOpRole role = *classifySegmentOpRole(op);
    role == SegmentOpRole::Unitary ? readyUnitaries.push(re)
                                   : readyDelimiters.push(re);
  };

  // initially populate min heaps with ready ops
  for (Operation *op : sdg.nodesInBlockOrder) {
    if (sdg.predecessorCountByOp.lookup(op) == 0)
      addReady(op);
  }

  // while both queues not empty
  while (!readyUnitaries.empty() || !readyDelimiters.empty()) {
    // - determine what the next op should be
    Operation *next = !readyUnitaries.empty()
                          ? popEarliestReadyEntry(readyUnitaries)
                          : popEarliestReadyEntry(readyDelimiters);
    // - push op to cso.opsInCanonicalOrder
    cso.opsInCanonicalOrder.push_back(next);
    // - for each successor of the op that just got pushed back
    for (Operation *succ : sdg.successorsByOp.lookup(next)) {
      //   - decrement the predecessorCountByOp[op];
      auto &predCountByOp = sdg.predecessorCountByOp;
      unsigned &predCount = predCountByOp[succ];
      assert(predCount > 0 &&
             "predecessor count should for this op should have been > 0");
      //   - if the count is 0 after update, add successor to the appropriate
      //     minheap
      if (--predCount == 0)
        addReady(succ);
    }
  }

  // failsafe:
  // if the number of ops in canonical order != num ops in segment, just take
  // the order from the nodesInBlockOrder and assign that to opsInCanonicalOrder
  if (cso.opsInCanonicalOrder.size() != sdg.nodesInBlockOrder.size()) {
    assert(false && "dependency graph contains a cycle or graph bookkeeping is "
                    "inconsistent");
    cso.mode = OrderingMode::Textual;
    cso.opsInCanonicalOrder = sdg.nodesInBlockOrder;
  }

  return cso;
}

/// form unitary group helpers

/// Ingest a CanonicalSegmentOrder and form groups of unitary ops separated by
/// measurements, resets, or hard boundaries (e.g., op with nested region, block
/// terminator).
void NewUnitaryOpGroupingAnalysis::formUnitaryGroups(
    const CanonicalSegmentOrder &cso) {

  SmallVector<Operation *> currentUnitaryOps;
  SmallVector<Operation *> currentDelimiterOps;

  auto recordGroupMembership = [&](llvm::ArrayRef<Operation *> ops,
                                   unsigned groupIndex) {
    for (Operation *op : ops) {
      bool inserted = opToGroupIndex.try_emplace(op, groupIndex).second;
      assert(inserted && "operation added to multiple groups");
    }
  };

  auto flushCurrentGroups = [&]() {
    // if current run is empty, do nothing
    if (currentUnitaryOps.empty() && currentDelimiterOps.empty())
      return;

    // grab the index for the group we are forming
    // this will be added to `blockToGroupIndices`
    const unsigned groupIndex = static_cast<unsigned>(unitaryOpGroups.size());

    // populate `currUnitaryOpGroup` with currentRun ops
    NewUnitaryOpGroup currUnitaryOpGroup;
    currUnitaryOpGroup.containingBlock = cso.containingBlock;
    currUnitaryOpGroup.ops.append(currentUnitaryOps);
    currUnitaryOpGroup.trailingDelimiterOps.append(currentDelimiterOps);

    // add unitary group and index to analysis outputs
    unitaryOpGroups.push_back(std::move(currUnitaryOpGroup));
    blockToGroupIndices[cso.containingBlock].push_back(groupIndex);

    // populate analysis' `opToGroupIndex` with all ops
    const auto &unitaryGroup = unitaryOpGroups.back();
    recordGroupMembership(unitaryGroup.ops, groupIndex);
    recordGroupMembership(unitaryGroup.trailingDelimiterOps, groupIndex);

    currentUnitaryOps.clear();
    currentDelimiterOps.clear();
  };

  // iterate over all ops in the cso to create unitary groups
  // add consecutive measurements/resets to the group that they end
  for (Operation *op : cso.opsInCanonicalOrder) {

    auto role = classifySegmentOpRole(op);
    assert(role && "canonical segment cannot contain a hard boundary");

    if (*role == SegmentOpRole::Unitary) {
      // A unitary after a delimiter starts the next group
      if (!currentDelimiterOps.empty())
        flushCurrentGroups();

      currentUnitaryOps.push_back(op);
    } else {
      currentDelimiterOps.push_back(op);
    }
  }

  // The end of the vector of ops is also a boundary; make sure to add the last
  // group
  flushCurrentGroups();
}

/// end helpers

/// Main driver of the analysis
/// - find segments of ops, where each segment ends by hard boundaries (i.e.,
/// ops with nested regions or block terminators)
///   - note that segment ends are different than group ends; segments end
///   because of hard boundaries. groups end either because of hard boundary or
///   msmt/reset
/// -
void NewUnitaryOpGroupingAnalysis::analyzeBlock(Block &block) {
  // perform qubitIdentityAnalysis to elucidate relationships between
  // potentially aliasing wires created by `unwrap`s
  QubitIdentityAnalysis qia(block);
  // container for all of the ops for the current quantum op segment
  QuantumOpSegment currSegment;
  currSegment.containingBlock = &block;

  // lambda for helping break off a segment after encountering hard boundary
  auto finishCurrentSegment = [&]() {
    // at this point, we have hit a hard boundary
    // 1. build dependency graph
    SegmentDependencyGraph sdg = buildSegmentDependencyGraph(currSegment, qia);
    // 2. canonicalize order
    CanonicalSegmentOrder cso = computeCanonicalSegmentOrder(sdg);
    // 3. form unitary groups
    formUnitaryGroups(cso);

    // at the end, we know we want to flush the segment
    currSegment.opsInBlockOrder.clear();
  };

  for (Operation &op : block) {
    LLVM_DEBUG(llvm::dbgs()
               << "\tFound op: " << op.getName()
               << " and parent op: " << op.getParentOp()->getName() << "\n");

    // classify grouping role: unitary, reset, or msmt
    // if not null, we add it to the current segment and `continue` this loop
    std::optional<SegmentOpRole> role = classifySegmentOpRole(&op);
    if (role) {
      currSegment.opsInBlockOrder.push_back(&op);
      continue;
    }

    // else if null, the segment is done and we use that segment for creating
    // dependency graph, canonical order, and getting the unitary groups
    finishCurrentSegment();

    // after the above, it is possible that we hit an op with a nested boundary
    // this is where we will recursively search blocks within the nested regions
    for (auto &region : op.getRegions()) {
      for (auto &block : region)
        analyzeBlock(block);
    }
  }
}

void NewUnitaryOpGroupingAnalysis::performAnalysis(Operation *op) {
  auto funcOp = mlir::dyn_cast<func::FuncOp>(op);
  if (!funcOp)
    return;
  LLVM_DEBUG(llvm::dbgs() << "Found funcOp: " << funcOp.getName() << "\n");

  for (Region &region : funcOp->getRegions()) {
    for (Block &block : region) {
      // note that this will find nested regions and explore their
      // corresponding ops recursively
      analyzeBlock(block);
    }
  }
}

std::optional<unsigned>
NewUnitaryOpGroupingAnalysis::getGroupIndexForOp(mlir::Operation *op) const {
  if (!op)
    return std::nullopt;

  auto iter = opToGroupIndex.find(op);
  if (iter == opToGroupIndex.end())
    return std::nullopt;

  assert(iter->second < unitaryOpGroups.size() &&
         "group index must name a retained group");
  return iter->second;
}

const mlir::Block *NewUnitaryOpGroupingAnalysis::getBlockForGroup(
    const NewUnitaryOpGroup &group) const {
  return group.containingBlock;
}

const NewUnitaryOpGroup *
NewUnitaryOpGroupingAnalysis::getGroupContainingOp(mlir::Operation *op) const {
  auto groupIndex = getGroupIndexForOp(op);
  return groupIndex ? &unitaryOpGroups[*groupIndex] : nullptr;
}

mlir::SmallVector<const NewUnitaryOpGroup *>
NewUnitaryOpGroupingAnalysis::getGroupsIn(const mlir::Block *block) const {
  mlir::SmallVector<const NewUnitaryOpGroup *> groupsInBlock;
  if (!block)
    return groupsInBlock;

  auto iter = blockToGroupIndices.find(block);
  if (iter == blockToGroupIndices.end())
    return groupsInBlock;

  groupsInBlock.reserve(iter->second.size());
  for (unsigned groupIndex : iter->second) {
    assert(groupIndex < unitaryOpGroups.size() &&
           "block index must name a retained group");
    groupsInBlock.push_back(&unitaryOpGroups[groupIndex]);
  }

  return groupsInBlock;
}

bool NewUnitaryOpGroupingAnalysis::inSameGroup(mlir::Operation *lhs,
                                               mlir::Operation *rhs) const {
  auto lhsGroup = getGroupIndexForOp(lhs);
  auto rhsGroup = getGroupIndexForOp(rhs);
  return lhsGroup && rhsGroup && *lhsGroup == *rhsGroup;
}

} // namespace cudaq::quake::detail
