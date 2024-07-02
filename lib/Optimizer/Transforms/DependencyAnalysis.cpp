/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Threading.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_DEPENDENCYANALYSIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
inline bool isMeasureOp(Operation *op) {
  return dyn_cast<quake::MxOp>(*op) || dyn_cast<quake::MyOp>(*op) ||
         dyn_cast<quake::MzOp>(*op);
}

inline bool isBeginOp(Operation *op) {
  return dyn_cast<quake::UnwrapOp>(*op) || dyn_cast<quake::ExtractRefOp>(*op) ||
         dyn_cast<quake::NullWireOp>(*op);
}

inline bool isEndOp(Operation *op) {
  return dyn_cast<quake::DeallocOp>(*op) || dyn_cast<quake::SinkOp>(*op);
}

class LifeTime {
protected:
  uint begin;
  uint end;

public:
  LifeTime(uint _begin, uint _end) : begin(_begin), end(_end){};

  bool isAfter(LifeTime *other) { return begin > other->end; }

  bool isOverlapping(LifeTime *other) {
    return !isAfter(other) && !other->isAfter(this);
  }

  uint distance(LifeTime *other) {
    if (isOverlapping(other))
      return 0;
    return std::max(begin, other->begin) - std::min(end, other->end);
  }

  LifeTime *combine(LifeTime *other) {
    return new LifeTime(std::min(begin, other->begin),
                        std::max(end, other->end));
  }

  uint getBegin() { return begin; }
  uint getEnd() { return end; }
};

class DependencyNode {
  friend class DependencyGraph;

protected:
  SmallVector<DependencyNode *> successors;
  SmallVector<DependencyNode *> dependencies;
  SetVector<size_t> qids;
  Operation *associated;
  uint cycle = INT_MAX;

  void printNode() {
    llvm::outs() << "QIDs: ";
    bool printComma = false;
    for (auto qid : qids) {
      if (printComma)
        llvm::outs() << ", ";
      llvm::outs() << qid;
      printComma = true;
    }
    if (isScheduled())
      llvm::outs() << " @ " << cycle;
    llvm::outs() << " | ";
    if (isJoin())
      llvm::outs() << "join\n";
    else
      associated->dump();
  }

  // Print with tab index to should depth in graph
  void printSubGraph(int tabIndex) {
    for (int i = 0; i < tabIndex; i++) {
      llvm::outs() << "\t";
    }

    printNode();

    for (auto dependency : dependencies) {
      dependency->printSubGraph(tabIndex + 1);
    }
  }

  bool isScheduled() { return cycle != INT_MAX; }

  bool isJoin() { return associated == nullptr; }

  bool isRoot() { return successors.size() == 0; }

  bool skip() {
    return isJoin() || isEndOp(associated) || isBeginOp(associated);
  }

  bool isDependentOn(DependencyNode *other) {
    if (this == other)
      return true;

    for (auto dependency : dependencies) {
      if (dependency->isDependentOn(other))
        return true;
    }

    return false;
  }

  void propagateUp(SetVector<size_t> &qidsToAdd) {
    qids.set_union(qidsToAdd);

    for (auto successor : successors) {
      successor->propagateUp(qidsToAdd);
    }
  }

  void schedule(uint depth, uint total_height) {
    if (!skip()) {
      depth++;
      uint _cycle = total_height - depth;
      if (_cycle < cycle)
        cycle = _cycle;
    }

    for (auto dependency : dependencies)
      dependency->schedule(depth, total_height);
  }

  void performMapping(DenseMap<size_t, size_t> &mapping) {
    if (associated && associated->hasAttr("qid")) {
      // Lookup old qid
      auto old_qid = associated->getAttrOfType<IntegerAttr>("qid").getUInt();
      OpBuilder builder(associated);
      associated->setAttr("pqid", builder.getUI32IntegerAttr(mapping[old_qid]));
    }

    SetVector<size_t> new_qids;
    for (auto qid : qids) {
      new_qids.insert(mapping[qid]);
    }
    qids = new_qids;

    for (auto dependency : dependencies) {
      dependency->performMapping(mapping);
    }
  }

  uint findFirstUse(size_t qid) {
    if (!qids.contains(qid))
      return INT_MAX;

    auto min = cycle;
    for (auto dependency : dependencies) {
      auto res = dependency->findFirstUse(qid);
      if (res < min)
        min = res;
    }

    return min;
  }

public:
  DependencyNode(Operation *op, DependencyNode *from)
      : successors(), dependencies(), qids(), associated(op) {
    if (op->hasAttr("qid")) {
      // Lookup qid
      auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
      qids.insert(qid);
    }

    if (from)
      addSuccessor(from);
  };

  /// From must be non-null here (join node)
  DependencyNode(DependencyNode *from)
      : successors({from}), dependencies(), qids(), associated(nullptr) {
    addSuccessor(from);
  };

  void addSuccessor(DependencyNode *other) {
    successors.push_back(other);
    other->dependencies.push_back(this);
    other->propagateUp(qids);
  }

  /// Adds other as a successor only if it won't cause a loop
  void tryAddSuccessor(DependencyNode *other) {
    if (isDependentOn(other))
      return;
    addSuccessor(other);
  }

  void print() { printSubGraph(0); }

  uint getHeight() {
    uint max = 0;
    for (auto dependency : dependencies) {
      uint candidate = dependency->getHeight();
      if (candidate > max)
        max = candidate;
    }

    if (!skip())
      max++;

    return max;
  }
};

class DependencyGraph {
private:
  SetVector<DependencyNode *> roots;
  SetVector<size_t> qids;
  uint total_height;

  void gatherRoots(SetVector<DependencyNode *> &seen, DependencyNode *next) {
    if (seen.contains(next))
      return;

    if (next->isRoot()) {
      roots.insert(next);
      auto root_height = next->getHeight();
      if (root_height > total_height)
        total_height = root_height;
    }

    seen.insert(next);
    qids.set_union(next->qids);

    for (auto successor : next->successors) {
      gatherRoots(seen, successor);
    }
    for (auto dependency : next->dependencies) {
      gatherRoots(seen, dependency);
    }
  }

  void scheduleNodes() {
    for (auto root : roots)
      root->schedule(0, total_height);
    // Want roots right after all their dependencies to minimize lifetimes
    for (auto root : roots) {
      uint max = 0;
      for (auto dependency : root->dependencies)
        if (dependency->cycle > max)
          max = dependency->cycle;
      // Roots are always sinks and thus skipped while counting
      root->cycle = max;
    }
  }

public:
  DependencyGraph(DependencyNode *root) {
    total_height = 0;
    SetVector<DependencyNode *> seen;
    gatherRoots(seen, root);
    scheduleNodes();
    qids = SetVector<size_t>();
    for (auto root : roots) {
      qids.set_union(root->qids);
    }
  }

  SetVector<DependencyNode *> &getRoots() { return roots; }

  SetVector<size_t> &getQIDs() { return qids; }

  size_t getNumQIDs() { return qids.size(); }

  uint findFirstUse(size_t qid) {
    uint min = INT_MAX;
    for (auto root : roots) {
      uint candidate = root->findFirstUse(qid);
      if (candidate < min)
        min = candidate;
    }

    return min;
  }

  LifeTime *getLifeTimeForQID(size_t qid) {
    uint first = findFirstUse(qid);
    // TODO: conservative estimate for now
    auto last = total_height - 1;

    return new LifeTime(first, last);
  }

  void print() {
    llvm::outs() << "Graph Start\n";
    for (auto root : roots)
      root->print();
    llvm::outs() << "Graph End\n";
  }

  void performMapping(SmallVector<size_t> &new_qids) {
    assert(new_qids.size() == getNumQIDs());
    DenseMap<size_t, size_t> map;

    size_t i = 0;
    for (auto qid : qids) {
      map.insert({qid, new_qids[i++]});
    }

    for (auto root : roots) {
      root->performMapping(map);
    }
  }
};

class DependencyAnalysis {
private:
  SmallVector<DependencyNode *> perOp;
  DenseMap<BlockArgument, DependencyNode *> map;

  inline DependencyNode *getDNodeId(Operation *op) {
    if (op->hasAttr("dnodeid")) {
      auto id = op->getAttr("dnodeid").cast<IntegerAttr>().getUInt();
      auto dnode = perOp[id];
      return dnode;
    }

    return nullptr;
  }

public:
  DependencyAnalysis() : perOp(), map(){};

  DependencyNode *handleDependencyOp(Operation *op, DependencyNode *next) {
    // If we've already visited this operation, return memoized dnode
    auto dnodeid = getDNodeId(op);
    if (dnodeid) {
      if (dnodeid != next)
        dnodeid->addSuccessor(next);
      return dnodeid;
    }

    // Construct new dnode
    DependencyNode *newNode = new DependencyNode(op, next);

    // Dnodeid is the next slot of the dnode vector
    auto id = perOp.size();

    // Add dnodeid attribute
    OpBuilder builder(op);
    op->setAttr("dnodeid", builder.getUI32IntegerAttr(id));
    perOp.push_back(newNode);

    // Reached end of graph (beginning of circuit), don't visit children
    if (isBeginOp(op))
      return newNode;

    // Recursively visit children
    for (auto operand : op->getOperands()) {
      handleDependencyValue(operand, newNode);
    }

    return newNode;
  }

  DependencyNode *handleDependencyArg(BlockArgument arg, DependencyNode *next) {
    // If we've already handled this block argument, return memoized value
    if (auto prev = map.lookup(arg)) {
      prev->tryAddSuccessor(next);
      return prev;
    }

    auto block = arg.getParentBlock();
    DependencyNode *newNode = next;
    // TODO: better way to check for multiple predecessors?
    // TODO: get single or get unique?
    // If join point, insert join node
    if (!block->getSinglePredecessor()) {
      newNode = new DependencyNode(next);
      map.insert({arg, newNode});
    }

    // Look up operands from all branch instructions that can jump
    // to the parent block and recursively visit them
    for (auto predecessor : block->getPredecessors()) {
      if (auto branch =
              dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
        unsigned numSuccs = branch->getNumSuccessors();
        for (unsigned i = 0; i < numSuccs; ++i) {
          if (block && branch->getSuccessor(i) != block)
            continue;
          auto brArgs = branch.getSuccessorOperands(i).getForwardedOperands();
          auto operand = brArgs[arg.getArgNumber()];
          handleDependencyValue(operand, newNode);
        }
      }
    }

    return newNode;
  }

  DependencyNode *handleDependencyValue(Value v, DependencyNode *next) {
    if (!isa<quake::WireType>(v.getType()))
      return nullptr;

    // Block arguments do not have associated operations,
    // but may require inserting joins, so they are handled specially
    if (auto arg = dyn_cast<BlockArgument>(v))
      return handleDependencyArg(arg, next);

    auto defOp = v.getDefiningOp();
    if (defOp)
      return handleDependencyOp(defOp, next);

    // TODO: FAIL
    llvm::outs() << "UNKNOWN VALUE\n";
    v.dump();
    return nullptr;
  }
};

const int MAX_QUBITS = 1;

struct DependencyAnalysisPass
    : public cudaq::opt::impl::DependencyAnalysisBase<DependencyAnalysisPass> {
  using DependencyAnalysisBase::DependencyAnalysisBase;

  std::optional<size_t> findBestQubit(SmallVector<LifeTime *> lifetimes,
                                      LifeTime *lifetime) {
    std::optional<size_t> best;
    uint best_distance = INT_MAX;
    for (uint i = 0; i < lifetimes.size(); i++) {
      LifeTime *other = lifetimes[i];
      auto distance = lifetime->distance(other);
      if (!lifetime->isOverlapping(other) && distance < best_distance) {
        best = i;
        best_distance = distance;
      }
    }

    return best;
  }

  LogicalResult mapQubIts(SmallVector<DependencyGraph> &graphs) {
    SmallVector<LifeTime *> lifetimes;

    for (auto graph : graphs) {
      auto qids = graph.getQIDs();
      SmallVector<size_t> new_qids;
      for (auto qid : qids) {
        auto lifetime = graph.getLifeTimeForQID(qid);
        llvm::outs() << "QID " << qid << " is alive from cycles " << lifetime->getBegin() << " to "
                     << lifetime->getEnd() << "\n";
        auto new_qid = findBestQubit(lifetimes, lifetime);
        if (!new_qid) {
          new_qid = lifetimes.size();
          lifetimes.push_back(lifetime);
        }
        lifetimes[new_qid.value()] =
            lifetime->combine(lifetimes[new_qid.value()]);
        new_qids.push_back(new_qid.value());
      }

      graph.performMapping(new_qids);
    }

    return success();
  }

  void runOnOperation() override {
    auto func = getOperation();

    DependencyAnalysis engine;

    SetVector<DependencyNode *> roots;

    func.walk([&](quake::SinkOp sop) {
      // Assumption: wire is second result
      auto root = engine.handleDependencyOp(sop, nullptr);
      if (root)
        roots.insert(root);
    });

    // Construct graphs from roots
    SmallVector<DependencyGraph> graphs;
    while (!roots.empty()) {
      DependencyGraph new_graph(roots.front());
      roots.set_subtract(new_graph.getRoots());
      graphs.push_back(new_graph);
    }

    for (auto graph : graphs)
      graph.print();

    if (failed(mapQubIts(graphs))) {
      emitError(func.getLoc(),
                "function " + func.getName() + " exceeds max # of qubits.");
      signalPassFailure();
    }
  }
};

} // namespace
