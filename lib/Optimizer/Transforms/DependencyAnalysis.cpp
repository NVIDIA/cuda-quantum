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

  struct Edge {
    size_t result_idx;
    size_t operand_idx;
  };

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
    associated->dump();
  }

  // Print with tab index to should depth in graph
  void printSubGraph(int tabIndex) {
    for (int i = 0; i < tabIndex; i++) {
      llvm::outs() << "\t";
    }

    printNode();

    for (auto dependency : dependencies)
      if (dependency)
        dependency->printSubGraph(tabIndex + 1);
  }

  bool isScheduled() { return cycle != INT_MAX; }

  bool isRoot() { return successors.size() == 0; }

  bool isSkip() { return isEndOp(associated) || isBeginOp(associated); }

  void propagateUp(SetVector<size_t> &qidsToAdd) {
    qids.set_union(qidsToAdd);

    for (auto successor : successors)
      if (successor)
        successor->propagateUp(qidsToAdd);
  }

  void schedule(uint depth, uint total_height) {
    if (!isSkip()) {
      depth++;
      uint _cycle = total_height - depth;
      if (_cycle < cycle)
        cycle = _cycle;
    }

    for (auto dependency : dependencies)
      // Assumption: graph saturated
      dependency->schedule(depth, total_height);
  }

  /// Dependencies and successors are ordered lists:
  /// Dependencies are in the order of operands,
  /// successors in the order of results.
  /// This function returns the operand index represented by \p dependency ,
  /// and which result of \p dependency is relevant to \p this
  Edge getEdgeInfo(DependencyNode *dependency) {
    size_t operand = 0;
    for (; operand < dependencies.size(); operand++)
      if (dependencies[operand] == dependency)
        break;

    size_t res = 0;
    for (; res < dependency->successors.size(); res++)
      if (dependency->successors[res] == this)
        break;

    return Edge{res, operand};
  }

  /// Recursively find nodes scheduled at a given cycle
  SetVector<DependencyNode *> getNodesAtCycle(uint _cycle) {
    SetVector<DependencyNode *> nodes;
    if (cycle == _cycle) {
      nodes.insert(this);
      return nodes;
    }

    for (auto dependency : dependencies)
      nodes.set_union(dependency->getNodesAtCycle(_cycle));

    return nodes;
  }

  /// Generates a new operation for this node in the dependency graph
  /// Using the dependencies of the node as operands
  void codeGen(OpBuilder &builder) {
    SmallVector<mlir::Value> operands(associated->getNumOperands());
    for (auto dependency : dependencies) {
      auto edge = getEdgeInfo(dependency);
      operands[edge.operand_idx] =
          dependency->associated->getResult(edge.result_idx);
    }

    auto newOp = Operation::create(associated->getLoc(), associated->getName(),
                                   associated->getResultTypes(), operands,
                                   associated->getAttrs());
    builder.insert(newOp);
    associated->dropAllUses();
    associated->erase();
    associated = newOp;
  }

  /// Replaces the null_wire op for \p qid with \p init
  void initializeWire(size_t qid, Operation *init) {
    if (!qids.contains(qid))
      return;

    if (isBeginOp(associated)) {
      auto leftover = associated;
      associated = init;
      leftover->dropAllUses();
      leftover->erase();
      return;
    }

    for (auto dependency : dependencies) {
      dependency->initializeWire(qid, init);
    }
  }

public:
  DependencyNode(Operation *op, DependencyNode *from) : qids(), associated(op) {
    uint num_dependencies = op->getNumOperands();
    dependencies = SmallVector<DependencyNode *>(num_dependencies, nullptr);
    uint num_successors = op->getNumResults();
    successors = SmallVector<DependencyNode *>(num_successors, nullptr);
    if (op->hasAttr("qid")) {
      // Lookup qid
      auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
      qids.insert(qid);
    }

    if (from)
      addSuccessor(from);
  };

  /// Add \p successor as a successor according to which result(s) is passed
  /// to \p successor as operands.
  /// Similarly, adds \p this as a dependency of \p successor according
  /// to which operand(s) the result(s) is passed as.
  void addSuccessor(DependencyNode *successor) {
    auto otherOp = successor->associated;

    for (auto res : associated->getResults())
      for (auto user : res.getUsers())
        if (user == otherOp)
          successors[res.getResultNumber()] = successor;

    for (size_t i = 0; i < otherOp->getNumOperands(); i++)
      if (otherOp->getOperand(i).getDefiningOp() == associated)
        successor->dependencies[i] = this;

    successor->propagateUp(qids);
  }

  void print() { printSubGraph(0); }

  uint getHeight() {
    uint max = 0;
    for (auto dependency : dependencies) {
      if (!dependency)
        continue;
      uint candidate = dependency->getHeight();
      if (candidate > max)
        max = candidate;
    }

    if (!isSkip())
      max++;

    return max;
  }

  /// Generates code to clean up the qid when it is no longer in use
  void addCleanUp(OpBuilder &builder) {
    for (auto successor : successors) {
      if (isEndOp(successor->associated)) {
        auto edge = successor->getEdgeInfo(this);
        auto newOp = builder.create<quake::SinkOp>(
            builder.getUnknownLoc(), associated->getResult(edge.result_idx));
        newOp->setAttrs(successor->associated->getAttrs());
        successor->associated->erase();
        successor->associated = newOp;
      }
    }
  }
};

class DependencyGraph {
private:
  SetVector<DependencyNode *> roots;
  SetVector<size_t> qids;
  uint total_height;
  DenseMap<size_t, DependencyNode *> firstUses;
  bool isScheduled = false;

  /// Starting from \p next, searches through \p next's family
  /// (excluding already seen nodes) to find all the interconnected roots
  /// that this graph represents.
  /// Also fills in metadata about the height of the graph, and the qids in the
  /// graph.
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

    if (isBeginOp(next->associated)) {
      firstUses.insert({next->qids.front(), next->successors.front()});
      firstUses[next->qids.front()]->printNode();
    }

    for (auto successor : next->successors)
      if (successor)
        gatherRoots(seen, successor);
    for (auto dependency : next->dependencies)
      if (dependency)
        gatherRoots(seen, dependency);
  }

  void scheduleNodes() {
    for (auto root : roots)
      root->schedule(0, total_height);
  }

  SetVector<DependencyNode *> getNodesAtCycle(uint cycle) {
    SetVector<DependencyNode *> nodes;
    for (auto root : roots)
      nodes.set_union(root->getNodesAtCycle(cycle));
    return nodes;
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

  LifeTime *getLifeTimeForQID(size_t qid) {
    uint first = getFirstUseOf(qid)->cycle;
    auto last = getLastUseOf(qid)->cycle;

    return new LifeTime(first, last);
  }

  DependencyNode *getFirstUseOf(size_t qid) { return firstUses[qid]; }

  DependencyNode *getLastUseOf(size_t qid) {
    for (auto root : roots)
      if (root->associated->getAttr("qid").cast<IntegerAttr>().getUInt() == qid)
        return root->dependencies[0];
    return nullptr;
  }

  void codeGenAt(uint cycle, OpBuilder &builder) {
    SetVector<DependencyNode *> nodes = getNodesAtCycle(cycle);

    for (auto node : nodes)
      node->codeGen(builder);
  }

  uint getHeight() { return total_height; }

  SmallVector<size_t> getFirstUsedAtCycle(uint cycle) {
    SmallVector<size_t> cycles;
    for (auto qid : qids) {
      auto first = getFirstUseOf(qid);
      if (first->cycle == cycle)
        cycles.push_back(qid);
    }

    return cycles;
  }

  void initializeWire(size_t qid, OpBuilder &builder) {
    auto ctx = builder.getContext();
    auto wireTy = quake::WireType::get(ctx);
    auto initOp =
        builder.create<quake::NullWireOp>(builder.getUnknownLoc(), wireTy);
    getLastUseOf(qid)->initializeWire(qid, initOp);
  }

  void initializeWire(size_t qid, DependencyNode *init) {
    auto leftover = init->successors[0];
    leftover->associated->erase();
    getLastUseOf(qid)->initializeWire(qid, init->associated);
    delete leftover;
  }

  void print() {
    llvm::outs() << "Graph Start\n";
    for (auto root : roots)
      root->print();
    llvm::outs() << "Graph End\n";
  }
};

class DependencyAnalysis {
private:
  SmallVector<DependencyNode *> perOp;

  inline DependencyNode *getDNodeId(Operation *op) {
    if (op->hasAttr("dnodeid")) {
      auto id = op->getAttr("dnodeid").cast<IntegerAttr>().getUInt();
      auto dnode = perOp[id];
      return dnode;
    }

    return nullptr;
  }

public:
  DependencyAnalysis() : perOp(){};

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

  DependencyNode *handleDependencyValue(Value v, DependencyNode *next) {
    if (!isa<quake::WireType>(v.getType()))
      return nullptr;

    auto defOp = v.getDefiningOp();
    if (defOp)
      return handleDependencyOp(defOp, next);

    // TODO: FAIL
    // llvm::outs() << "UNKNOWN VALUE\n"
    assert(false && "UKKNOWN VALUE");
    return nullptr;
  }
};

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
      if (lifetime->isAfter(other) && distance < best_distance) {
        best = i;
        best_distance = distance;
      }
    }

    return best;
  }

  uint getTotalCycles(SmallVector<DependencyGraph> &graphs) {
    uint total = 0;
    SmallVector<LifeTime *> lifetimes;
    SmallVector<DependencyNode *> live_wires;

    for (auto graph : graphs)
      if (graph.getHeight() > total)
        total = graph.getHeight();

    return total;
  }

  void codeGen(SmallVector<DependencyGraph> &graphs, OpBuilder &builder) {
    SmallVector<LifeTime *> lifetimes;
    SmallVector<DependencyNode *> live_wires;
    uint cycles = getTotalCycles(graphs);

    for (uint cycle = 0; cycle < cycles; cycle++) {
      for (auto graph : graphs) {
        // For every "new" qubit, try to find an existing out-of-use qubit
        // that we can reuse. Failing that, use a new qubit.
        for (auto qid : graph.getFirstUsedAtCycle(cycle)) {
          auto lifetime = graph.getLifeTimeForQID(qid);
          auto new_qid = findBestQubit(lifetimes, lifetime);
          if (!new_qid) {
            // Can't reuse any qubits, have to allocate a new one
            new_qid = lifetimes.size();
            lifetimes.push_back(lifetime);
            live_wires.push_back(graph.getLastUseOf(qid));
            // Initialize the qubit with a null wire op
            graph.initializeWire(qid, builder);
          } else {
            // We found a qubit we can reuse!
            lifetimes[new_qid.value()] =
                lifetime->combine(lifetimes[new_qid.value()]);

            auto init = live_wires[new_qid.value()];
            // We assume that the result of the last use of the old qubit
            // must have been a null wire (e.g., it was reset),
            // so we can reuse that result as the initial value to reuse it
            graph.initializeWire(qid, init);
            live_wires[new_qid.value()] = graph.getLastUseOf(qid);
          }
        }

        graph.codeGenAt(cycle, builder);
      }
    }

    // Add teardown instructions
    for (auto node : live_wires)
      node->addCleanUp(builder);
  }

  void runOnOperation() override {
    auto func = getOperation();

    DependencyAnalysis engine;

    SetVector<DependencyNode *> roots;

    func.walk([&](quake::SinkOp sop) {
      auto root = engine.handleDependencyOp(sop, nullptr);
      if (root)
        roots.insert(root);
    });

    for (auto root : roots)
      root->print();

    // Construct graphs from roots
    SmallVector<DependencyGraph> graphs;
    while (!roots.empty()) {
      DependencyGraph new_graph(roots.front());
      roots.set_subtract(new_graph.getRoots());
      graphs.push_back(new_graph);
    }

    OpBuilder builder(func.getOperation());
    builder.setInsertionPoint(&func.back().back());
    codeGen(graphs, builder);
  }
};

} // namespace
