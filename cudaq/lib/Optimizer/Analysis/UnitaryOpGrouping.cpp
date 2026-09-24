/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "QubitIdentityAnalysis.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <queue>

#define DEBUG_TYPE "unitary-op-grouping-analysis"

using namespace mlir;

namespace cudaq::quake::detail {

/// Classify an operation for quantum-segment construction.
/// Return `std::nullopt` when the operation is a hard boundary.
static std::optional<SegmentOpRole> classifySegmentOpRole(Operation *op) {
  // Region-owning operations and terminators are always hard boundaries.
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return std::nullopt;

  if (isa<cudaq::quake::MeasurementInterface>(op))
    return SegmentOpRole::MsmtDelimiter;

  // Reset has the `QuantumGate` trait but is not unitary.
  if (isa<cudaq::quake::ResetOp>(op))
    return SegmentOpRole::ResetDelimiter;

  if (op->hasTrait<cudaq::QuantumGate>())
    return SegmentOpRole::Unitary;

  return std::nullopt;
}

/// Return whether \p op is a node in \p sdg.
static bool containsNode(const SegmentDependencyGraph &sdg, Operation *op) {
  return sdg.originalPositionByOp.contains(op);
}

/// Add a new graph node. The caller must not add duplicates.
static void addNode(SegmentDependencyGraph &sdg, Operation *op,
                    unsigned originalPos) {
  std::optional<SegmentOpRole> role = classifySegmentOpRole(op);
  assert(role && "op must be of `SegmentOpRole` type");
  assert(!containsNode(sdg, op) && "node added more than once");

  sdg.nodesInBlockOrder.push_back(op);
  sdg.originalPositionByOp.try_emplace(op, originalPos);

  // Materialize adjacency and in-degree entries for isolated nodes too.
  sdg.successorsByOp.try_emplace(op);
  sdg.predecessorCountByOp.try_emplace(op, 0);
}

/// Use wire-dataflow ordering only when every segment operation has scalar wire
/// flow and every quantum input has a known logical identity; otherwise use
/// textual order.
static OrderingMode determineOrderingMode(QuantumOpSegment &segment,
                                          QubitIdentityAnalysis &qia) {
  for (auto &op : segment.opsInBlockOrder) {
    std::optional<ScalarWireFlow> flow = getScalarWireFlow(op);
    if (!flow)
      return OrderingMode::Textual;

    for (Value &quantumOperand : flow->inputs)
      if (!qia.getQubitId(quantumOperand))
        return OrderingMode::Textual;
  }

  return OrderingMode::WireDataflow;
}

/// Record a unique predecessor-to-successor edge and increment the successor's
/// in-degree.
static void addEdge(SegmentDependencyGraph &sdg, Operation *predecessor,
                    Operation *successor) {
  assert(containsNode(sdg, predecessor) && "predecessor node must be in graph");
  assert(containsNode(sdg, successor) && "successor node must be in graph");

  auto &successors = sdg.successorsByOp[predecessor];

  // Def-use and qubit-identity reasoning may discover the same edge.
  if (llvm::is_contained(successors, successor))
    return;

  successors.push_back(successor);

  ++sdg.predecessorCountByOp[successor];
}

/// Add intra-segment SSA producer-to-consumer edges for wire-dataflow mode.
static void addIntraSegmentDefUseEdges(SegmentDependencyGraph &sdg) {
  for (Operation *consumer : sdg.nodesInBlockOrder) {
    for (Value operand : consumer->getOperands()) {
      Operation *producer = operand.getDefiningOp();
      if (producer && containsNode(sdg, producer))
        addEdge(sdg, producer, consumer);
    }
  }
}

/// Preserve original order between consecutive segment operations that touch
/// the same known logical qubit, including operations with distinct SSA roots.
static void addQubitIdentityEdges(SegmentDependencyGraph &sdg,
                                  QubitIdentityAnalysis &qia) {

  // Most recent segment operation touching each logical qubit.
  mlir::DenseMap<QubitIdentityAnalysis::QubitId, Operation *>
      lastTouchByQubitId;
  for (Operation *consumer : sdg.nodesInBlockOrder) {
    std::optional<ScalarWireFlow> flow = getScalarWireFlow(consumer);
    assert(flow && "requires valid scalar wire data flow");

    // Deduplicate identities within one operation to avoid a self-edge.
    llvm::SmallDenseSet<QubitIdentityAnalysis::QubitId, 4> touchedByThisOp;

    for (Value operand : flow->inputs) {
      std::optional<QubitIdentityAnalysis::QubitId> qid =
          qia.getQubitId(operand);
      assert(qid && "all wires must have QubitID");

      if (!touchedByThisOp.insert(*qid).second)
        continue;

      // Preserve order from the previous touch of this logical qubit.
      auto lastOpToTouchQid = lastTouchByQubitId.find(*qid);
      if (lastOpToTouchQid != lastTouchByQubitId.end())
        addEdge(sdg, lastOpToTouchQid->second, consumer);
      lastTouchByQubitId[*qid] = consumer;
    }
  }
}

/// Build the dependency graph for one quantum-operation segment.
static SegmentDependencyGraph
buildSegmentDependencyGraph(QuantumOpSegment &segment,
                            QubitIdentityAnalysis &qia) {
  SegmentDependencyGraph sdg;
  sdg.containingBlock = segment.containingBlock;

  for (auto [opIdxInBlock, op] : llvm::enumerate(segment.opsInBlockOrder)) {
    addNode(sdg, op, opIdxInBlock);
  }

  sdg.mode = determineOrderingMode(segment, qia);

  // Textual mode preserves nodesInBlockOrder directly and needs no edges.
  if (sdg.mode == OrderingMode::WireDataflow) {
    addIntraSegmentDefUseEdges(sdg);
    addQubitIdentityEdges(sdg, qia);
  }

  return sdg;
}

// "Ready" means that all predecessors of this op have been added to
// canonical order.
struct ReadyEntry {
  unsigned originalPos;
  Operation *op;
};

// `std::priority_queue` places the highest-priority element at the top.
// Treating later positions as lower priority makes the earliest original
// position top.
struct EarliestOriginalPositionFirst {
  bool operator()(const ReadyEntry &lhs, const ReadyEntry &rhs) const {
    return lhs.originalPos > rhs.originalPos;
  }
};

using ReadyMinHeap = std::priority_queue<ReadyEntry, SmallVector<ReadyEntry>,
                                         EarliestOriginalPositionFirst>;

static Operation *popEarliestReadyEntry(ReadyMinHeap &rmh) {
  assert(!rmh.empty() && "ReadyMinHeap must have an entry to pop!");
  Operation *earliestEntry = rmh.top().op;
  rmh.pop();
  return earliestEntry;
}

CanonicalSegmentOrder UnitaryOpGroupingAnalysis::computeCanonicalSegmentOrder(
    SegmentDependencyGraph &sdg) {
  CanonicalSegmentOrder cso;
  cso.containingBlock = sdg.containingBlock;
  cso.mode = sdg.mode;

  // Textual mode intentionally bypasses graph traversal.
  if (cso.mode == OrderingMode::Textual) {
    cso.opsInCanonicalOrder = sdg.nodesInBlockOrder;
    return cso;
  }

  // Prefer any ready unitary over a ready delimiter. Within either class,
  // prefer original segment order.
  ReadyMinHeap readyUnitaries;
  ReadyMinHeap readyDelimiters;

  auto addReady = [&](Operation *op) {
    unsigned origPos = sdg.originalPositionByOp.lookup(op);
    ReadyEntry re{origPos, op};

    SegmentOpRole role = *classifySegmentOpRole(op);
    role == SegmentOpRole::Unitary ? readyUnitaries.push(re)
                                   : readyDelimiters.push(re);
  };

  for (Operation *op : sdg.nodesInBlockOrder) {
    if (sdg.predecessorCountByOp.lookup(op) == 0)
      addReady(op);
  }

  // Emit the preferred ready node, then release newly ready successors.
  while (!readyUnitaries.empty() || !readyDelimiters.empty()) {
    Operation *next = !readyUnitaries.empty()
                          ? popEarliestReadyEntry(readyUnitaries)
                          : popEarliestReadyEntry(readyDelimiters);
    cso.opsInCanonicalOrder.push_back(next);
    for (Operation *succ : sdg.successorsByOp.lookup(next)) {
      auto &predCountByOp = sdg.predecessorCountByOp;
      unsigned &predCount = predCountByOp[succ];
      assert(predCount > 0 &&
             "predecessor count should for this op should have been > 0");
      if (--predCount == 0)
        addReady(succ);
    }
  }

  // A complete topological traversal emits every node. Debug builds diagnose a
  // cycle or bookkeeping error; release builds conservatively use block order.
  if (cso.opsInCanonicalOrder.size() != sdg.nodesInBlockOrder.size()) {
    assert(false && "dependency graph contains a cycle or graph bookkeeping is "
                    "inconsistent");
    cso.mode = OrderingMode::Textual;
    cso.opsInCanonicalOrder = sdg.nodesInBlockOrder;
  }

  return cso;
}

/// Partition canonical segment order into groups of unitaries followed by
/// consecutive measurement/reset delimiters. Leading delimiters form a
/// delimiter-only group.
void UnitaryOpGroupingAnalysis::formUnitaryGroups(
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

  auto flushCurrentGroup = [&]() {
    // Never emit an empty group.
    if (currentUnitaryOps.empty() && currentDelimiterOps.empty())
      return;

    const unsigned groupIndex = static_cast<unsigned>(unitaryOpGroups.size());

    UnitaryOpGroup currUnitaryOpGroup;
    currUnitaryOpGroup.block = cso.containingBlock;
    currUnitaryOpGroup.ops.append(currentUnitaryOps);
    currUnitaryOpGroup.trailingDelimiterOps.append(currentDelimiterOps);

    unitaryOpGroups.push_back(std::move(currUnitaryOpGroup));
    blockToGroupIndices[cso.containingBlock].push_back(groupIndex);

    // Index both unitary and delimiter operations as group members.
    const auto &unitaryGroup = unitaryOpGroups.back();
    recordGroupMembership(unitaryGroup.ops, groupIndex);
    recordGroupMembership(unitaryGroup.trailingDelimiterOps, groupIndex);

    currentUnitaryOps.clear();
    currentDelimiterOps.clear();
  };

  // Accumulate consecutive delimiters until the next unitary starts a new
  // group.
  for (Operation *op : cso.opsInCanonicalOrder) {

    auto role = classifySegmentOpRole(op);
    assert(role && "canonical segment cannot contain a hard boundary");

    if (*role == SegmentOpRole::Unitary) {
      // A unitary after a delimiter starts the next group.
      if (!currentDelimiterOps.empty())
        flushCurrentGroup();

      currentUnitaryOps.push_back(op);
    } else {
      currentDelimiterOps.push_back(op);
    }
  }

  // Flush any group remaining at segment end.
  flushCurrentGroup();
}

/// Analyze one block by forming maximal segments of classifiable quantum
/// operations. Every unclassified operation flushes the current segment; its
/// nested blocks are then analyzed recursively, so groups never cross blocks or
/// hard boundaries.
void UnitaryOpGroupingAnalysis::analyzeBlock(Block &block) {
  // Compute logical identities once per block so distinct wire SSA roots
  // produced by `unwrap` can still be ordered.
  QubitIdentityAnalysis qia(block);
  QuantumOpSegment currSegment;
  currSegment.containingBlock = &block;

  // Finish a segment at either a hard boundary or the end of the block.
  auto finishCurrentSegment = [&]() {
    if (currSegment.opsInBlockOrder.empty())
      return;

    // Canonicalize and group the completed segment.
    SegmentDependencyGraph sdg = buildSegmentDependencyGraph(currSegment, qia);
    CanonicalSegmentOrder cso = computeCanonicalSegmentOrder(sdg);
    formUnitaryGroups(cso);

    currSegment.opsInBlockOrder.clear();
  };

  for (Operation &op : block) {
    LLVM_DEBUG(llvm::dbgs()
               << "\tFound op: " << op.getName()
               << " and parent op: " << op.getParentOp()->getName() << "\n");

    std::optional<SegmentOpRole> role = classifySegmentOpRole(&op);
    if (role) {
      currSegment.opsInBlockOrder.push_back(&op);
      continue;
    }

    finishCurrentSegment();

    // Recurse after flushing so groups cannot cross a region-owning operation.
    for (auto &region : op.getRegions()) {
      for (auto &block : region)
        analyzeBlock(block);
    }
  }

  // Some regions permit blocks without terminators, so flush any trailing
  // segment that was not ended by a hard-boundary operation.
  finishCurrentSegment();
}

void UnitaryOpGroupingAnalysis::performAnalysis(Operation *op) {
  auto funcOp = mlir::dyn_cast<func::FuncOp>(op);
  if (!funcOp)
    return;
  LLVM_DEBUG(llvm::dbgs() << "Found funcOp: " << funcOp.getName() << "\n");

  for (Region &region : funcOp->getRegions()) {
    for (Block &block : region) {
      analyzeBlock(block);
    }
  }
}

std::optional<unsigned>
UnitaryOpGroupingAnalysis::getGroupIndexForOp(mlir::Operation *op) const {
  if (!op)
    return std::nullopt;

  auto iter = opToGroupIndex.find(op);
  if (iter == opToGroupIndex.end())
    return std::nullopt;

  assert(iter->second < unitaryOpGroups.size() &&
         "group index must name a retained group");
  return iter->second;
}

const mlir::Block *
UnitaryOpGroupingAnalysis::getBlockForGroup(const UnitaryOpGroup &group) const {
  return group.block;
}

const UnitaryOpGroup *
UnitaryOpGroupingAnalysis::getGroupContainingOp(mlir::Operation *op) const {
  auto groupIndex = getGroupIndexForOp(op);
  return groupIndex ? &unitaryOpGroups[*groupIndex] : nullptr;
}

mlir::SmallVector<const UnitaryOpGroup *>
UnitaryOpGroupingAnalysis::getGroupsIn(const mlir::Block *block) const {
  mlir::SmallVector<const UnitaryOpGroup *> groupsInBlock;
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

bool UnitaryOpGroupingAnalysis::inSameGroup(mlir::Operation *lhs,
                                            mlir::Operation *rhs) const {
  auto lhsGroup = getGroupIndexForOp(lhs);
  auto rhsGroup = getGroupIndexForOp(rhs);
  return lhsGroup && rhsGroup && *lhsGroup == *rhsGroup;
}

} // namespace cudaq::quake::detail
