/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <limits>
#include <set>

namespace cudaq::opt {
#define GEN_PASS_DEF_GROUPUNITARYOPS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

/// True when memory instances may require the earlier op to stay before the
/// later op in the original block order (Bernstein-style conflicts on the same
/// SSA `Value`). Two reads of the same value do not constrain order.
static bool memoryInstancesOrderDependent(const MemoryEffects::EffectInstance &a,
                                          const MemoryEffects::EffectInstance &b) {
  Value va = a.getValue();
  Value vb = b.getValue();
  if (!va || !vb)
    return true;
  if (va != vb)
    return false;

  auto isReadOnly = [](const MemoryEffects::EffectInstance &e) {
    return isa<MemoryEffects::Read>(e.getEffect());
  };
  if (isReadOnly(a) && isReadOnly(b))
    return false;
  return true;
}

/// Returns true if `earlier` (which appears before `later` in the original
/// block order) must stay before `later` to preserve memory semantics.
static bool memoryRequiresOriginalOrder(Operation *earlier, Operation *later) {
  if (isMemoryEffectFree(earlier) && isMemoryEffectFree(later))
    return false;

  auto fxEarlier = getEffectsRecursively(earlier);
  auto fxLater = getEffectsRecursively(later);
  if (!fxEarlier || !fxLater)
    return true;

  for (const auto &a : *fxEarlier)
    for (const auto &b : *fxLater)
      if (memoryInstancesOrderDependent(a, b))
        return true;
  return false;
}

/// Unknown memory behavior: keep this op's position relative to every other op
/// in the same block (still compatible with a forward-only edge DAG).
static bool mustPinInBlockOrder(Operation *op) {
  return hasUnknownEffects(op) || !getEffectsRecursively(op);
}

// each pair is (from, to)
using EdgeSet = std::set<std::pair<unsigned, unsigned>>;

static void addEdge(unsigned from, unsigned to, EdgeSet &edgeSet,
                    SmallVector<SmallVector<unsigned, 4>> &succs,
                    MutableArrayRef<unsigned> indegree) {
  if (!edgeSet.insert({from, to}).second)
    return;
  succs[from].push_back(to);
  indegree[to]++;
}

static void addSsaEdgesToSuccessors(Block *block, ArrayRef<Operation *> ops,
                                    EdgeSet &edgeSet,
                                    SmallVector<SmallVector<unsigned, 4>> &succs,
                                    MutableArrayRef<unsigned> indegree) {
  DenseMap<Operation *, unsigned> index;
  index.reserve(ops.size());
  for (auto [i, op] : llvm::enumerate(ops))
    index[op] = static_cast<unsigned>(i);

  for (auto [i, op] : llvm::enumerate(ops)) {
    for (Value v : op->getOperands()) {
      Operation *def = v.getDefiningOp();
      if (!def || def->getBlock() != block)
        continue;
      auto it = index.find(def);
      if (it == index.end())
        continue;
      const unsigned j = it->second;
      if (j == i)
        continue;
      addEdge(j, static_cast<unsigned>(i), edgeSet, succs, indegree);
    }
  }
}

static void addMemoryAndPinEdges(ArrayRef<Operation *> ops, EdgeSet &edgeSet,
                                 SmallVector<SmallVector<unsigned, 4>> &succs,
                                 MutableArrayRef<unsigned> indegree) {
  const unsigned n = static_cast<unsigned>(ops.size());
  for (unsigned i = 0; i < n; ++i) {
    if (mustPinInBlockOrder(ops[i])) {
      for (unsigned j = 0; j < i; ++j)
        addEdge(j, i, edgeSet, succs, indegree);
      for (unsigned j = i + 1; j < n; ++j)
        addEdge(i, j, edgeSet, succs, indegree);
      continue;
    }
    for (unsigned j = i + 1; j < n; ++j) {
      if (mustPinInBlockOrder(ops[j])) {
        addEdge(i, j, edgeSet, succs, indegree);
        continue;
      }
      if (memoryRequiresOriginalOrder(ops[i], ops[j]))
        addEdge(i, j, edgeSet, succs, indegree);
    }
  }
}

static bool isUnitaryOp(Operation *op) { return op->hasTrait<cudaq::Unitary>(); }

static bool isMeasurementOp(Operation *op) {
  return op->hasTrait<cudaq::QuantumMeasure>();
}

/// Among `ready`, pick the next op to schedule. Prefer ready ops that carry the
/// `Unitary` trait (smallest original index), then ready ops with the
/// `QuantumMeasure` trait, then the smallest original index. This pulls
/// independent unitaries and measurements earlier when the dependence graph
/// allows, analogous to the classical-vs-quantum tie-break in
/// `group-classical-and-quantum-ops`.
static unsigned pickNextReady(ArrayRef<unsigned> ready, ArrayRef<Operation *> ops) {
  assert(!ready.empty());
  unsigned bestUnitary = std::numeric_limits<unsigned>::max();
  for (unsigned idx : ready)
    if (isUnitaryOp(ops[idx]) && idx < bestUnitary)
      bestUnitary = idx;
  if (bestUnitary != std::numeric_limits<unsigned>::max())
    return bestUnitary;

  unsigned bestMeasure = std::numeric_limits<unsigned>::max();
  for (unsigned idx : ready)
    if (isMeasurementOp(ops[idx]) && idx < bestMeasure)
      bestMeasure = idx;
  if (bestMeasure != std::numeric_limits<unsigned>::max())
    return bestMeasure;

  unsigned best = ready.front();
  for (unsigned idx : ready)
    if (idx < best)
      best = idx;
  return best;
}

static LogicalResult scheduleBlock(Block &block, ArrayRef<Operation *> ops) {
  if (ops.size() <= 1)
    return success();

  const unsigned n = static_cast<unsigned>(ops.size());
  SmallVector<SmallVector<unsigned, 4>> succs(n);
  SmallVector<unsigned, 32> indegree(n, 0);
  EdgeSet edgeSet;

  addSsaEdgesToSuccessors(&block, ops, edgeSet, succs, indegree);
  addMemoryAndPinEdges(ops, edgeSet, succs, indegree);

  SmallVector<unsigned, 32> ready;
  ready.reserve(n);
  for (unsigned i = 0; i < n; ++i)
    if (indegree[i] == 0)
      ready.push_back(i);

  SmallVector<unsigned, 32> order;
  order.reserve(n);
  while (!ready.empty()) {
    const unsigned u = pickNextReady(ready, ops);
    llvm::erase(ready, u);
    order.push_back(u);

    for (unsigned v : succs[u]) {
      assert(indegree[v] > 0);
      if (--indegree[v] == 0)
        ready.push_back(v);
    }
  }

  if (order.size() != n) {
    mlir::emitError(block.getParentOp()->getLoc(),
                    "group-unitary-ops: cycle in dependence graph");
    return failure();
  }

  Operation *terminator = block.getTerminator();
  for (unsigned idx : order)
    ops[idx]->moveBefore(terminator);

  return success();
}

static LogicalResult reorderOperationsInBlock(Block &block) {
  SmallVector<Operation *> ops;
  for (Operation &op : block.without_terminator())
    ops.push_back(&op);
  return scheduleBlock(block, ops);
}

static void collectBlocks(Operation *root,
                          llvm::SmallVectorImpl<Block *> &blocks) {
  DenseSet<Block *> seen;
  auto add = [&](Block *b) {
    if (seen.insert(b).second)
      blocks.push_back(b);
  };

  root->walk([&](Operation *nested) {
    for (Region &region : nested->getRegions())
      for (Block &b : region)
        add(&b);
  });
}

struct GroupUnitaryOpsPass
    : public cudaq::opt::impl::GroupUnitaryOpsBase<GroupUnitaryOpsPass> {
  using GroupUnitaryOpsBase::GroupUnitaryOpsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    SmallVector<Block *> blocks;
    collectBlocks(func, blocks);

    for (Block *block : blocks) {
      if (failed(reorderOperationsInBlock(*block))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
