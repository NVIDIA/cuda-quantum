/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "dep-analysis"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//
namespace cudaq::opt {
#define GEN_PASS_DEF_DEPENDENCYANALYSIS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
#include <llvm/Support/Debug.h>
} // namespace cudaq::opt

namespace {
[[maybe_unused]] bool isMeasureOp(Operation *op) {
  return dyn_cast<quake::MxOp>(*op) || dyn_cast<quake::MyOp>(*op) ||
         dyn_cast<quake::MzOp>(*op);
}

[[maybe_unused]] bool isBeginOp(Operation *op) {
  return dyn_cast<quake::UnwrapOp>(*op) || dyn_cast<quake::ExtractRefOp>(*op) ||
         dyn_cast<quake::NullWireOp>(*op);
}

[[maybe_unused]] bool isEndOp(Operation *op) {
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
  // dependencies and successors are ordered lists:
  SmallVector<DependencyNode *> successors;
  // successors in the order of results of associated,
  SmallVector<DependencyNode *> dependencies;
  // dependencies are in the order of operands of associated
  SetVector<size_t> qids;
  Operation *associated;
  uint cycle = INT_MAX;
  bool hasCodeGen = false;

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
    if (!isEndOp(associated))
      qids.set_union(qidsToAdd);

    for (auto successor : successors)
      if (successor)
        successor->propagateUp(qidsToAdd);
  }

  void schedule(uint depth, uint total_height) {
    if (!isSkip()) {
      depth++;
      uint _cycle = total_height - depth;
      if (_cycle >= cycle)
        return;
      cycle = _cycle;
    }

    for (auto dependency : dependencies)
      if (dependency)
        dependency->schedule(depth, total_height);

    for (auto successor : successors)
      if (successor && !successor->isScheduled())
        successor->schedule(depth - 2, total_height);
  }

  /// Dependencies and successors are ordered lists:
  /// Dependencies are in the order of operands,
  /// successors in the order of results.
  /// This function returns the operand index represented by \p dependency
  /// and the relevant result index of \p dependency
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

    if (cycle < _cycle)
      return nodes;
    else if (cycle == _cycle)
      nodes.insert(this);

    for (auto dependency : dependencies)
      if (dependency)
        nodes.set_union(dependency->getNodesAtCycle(_cycle));

    return nodes;
  }

  /// Generates a new operation for this node in the dependency graph
  /// using the dependencies of the node as operands.
  void codeGen(OpBuilder &builder) {
    if (hasCodeGen)
      return;

    auto oldOp = associated;
    SmallVector<mlir::Value> operands(oldOp->getNumOperands());
    for (auto dependency : dependencies) {
      // Get relevant result from dependency's op
      // to use as the relevant operand
      auto edge = getEdgeInfo(dependency);
      operands[edge.operand_idx] =
          dependency->associated->getResult(edge.result_idx);
    }

    associated =
        Operation::create(oldOp->getLoc(), oldOp->getName(),
                          oldOp->getResultTypes(), operands, oldOp->getAttrs());
    builder.insert(associated);
    hasCodeGen = true;
  }

  /// Replaces the null_wire op for \p qid with \p init
  void initializeWire(size_t qid, Operation *init) {
    if (!qids.contains(qid))
      return;

    if (isBeginOp(associated)) {
      associated = init;
      return;
    }

    for (auto dependency : dependencies)
      if (dependency)
        dependency->initializeWire(qid, init);
  }

public:
  DependencyNode(Operation *op) : qids(), associated(op) {
    uint num_dependencies = op->getNumOperands();
    dependencies = SmallVector<DependencyNode *>(num_dependencies, nullptr);
    uint num_successors = op->getNumResults();
    successors = SmallVector<DependencyNode *>(num_successors, nullptr);

    if (op->hasAttr("qid")) {
      // Lookup qid
      auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
      qids.insert(qid);
    }
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

  void addCleanUp(OpBuilder &builder) {
    assert(isRoot() && "Can only call addCleanUp on a root node!");
    auto last_use = dependencies[0];
    auto edge = getEdgeInfo(last_use);
    auto wire = last_use->associated->getResult(edge.result_idx);
    auto newOp = builder.create<quake::SinkOp>(builder.getUnknownLoc(), wire);
    newOp->setAttrs(associated->getAttrs());
    associated = newOp;
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

    if (isBeginOp(next->associated))
      firstUses.insert({next->qids.front(), next->successors.front()});

    for (auto successor : next->successors)
      if (successor)
        gatherRoots(seen, successor);
    for (auto dependency : next->dependencies)
      if (dependency)
        gatherRoots(seen, dependency);
  }

  void scheduleNodes() {
    DependencyNode *tallest;
    int max = 0;
    for (auto root : roots) {
      int height = root->getHeight();
      if (height > max) {
        max = height;
        tallest = root;
      }
    }

    tallest->schedule(0, total_height);
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
    return getRootForQID(qid)->dependencies[0];
  }

  DependencyNode *getRootForQID(size_t qid) {
    for (auto root : roots)
      if (root->associated->getAttr("qid").cast<IntegerAttr>().getUInt() == qid)
        return root;
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

  void initializeWireFromRoot(size_t qid, DependencyNode *init) {
    auto lastOp = init->dependencies[0]->associated;
    getFirstUseOf(qid)->initializeWire(qid, lastOp);
  }

  void print() {
    llvm::outs() << "Graph Start\n";
    for (auto root : roots)
      root->print();
    llvm::outs() << "Graph End\n";
  }

  Location getIntroductionLoc(size_t qid) {
    return getFirstUseOf(qid)->associated->getLoc();
  }
};

struct DependencyAnalysisPass
    : public cudaq::opt::impl::DependencyAnalysisBase<DependencyAnalysisPass> {
  using DependencyAnalysisBase::DependencyAnalysisBase;
  SmallVector<DependencyNode *> perOp;

  /// Returns the dependency node for \p op
  /// Creates a new dependency node if it does not exist
  DependencyNode *visitOp(Operation *op, DependencyNode *next) {
    // If we've already visited this operation, return memoized dnode
    if (op->hasAttr("dnodeid")) {
      auto id = op->getAttr("dnodeid").cast<IntegerAttr>().getUInt();
      auto dnode = perOp[id];
      dnode->addSuccessor(next);
      return dnode;
    }

    // Skip classical operations
    bool quakeOp = false;
    for (auto type : op->getResultTypes())
      if (isa<quake::WireType>(type)) {
        quakeOp = true;
        break;
      }

    if (!quakeOp && !isEndOp(op))
      return nullptr;

    DependencyNode *newNode = new DependencyNode(op);

    // Dnodeid is the next slot of the dnode vector
    auto id = perOp.size();

    // Add dnodeid attribute
    OpBuilder builder(op);
    op->setAttr("dnodeid", builder.getUI32IntegerAttr(id));
    perOp.push_back(newNode);

    // Recursively visit children
    for (auto operand : op->getOperands())
      visitValue(operand, newNode);

    return newNode;
  }

  /// Returns the dependency node for the defining operation of \p v
  /// Creates a new dependency node if it does not exist
  DependencyNode *visitValue(Value v, DependencyNode *next) {
    // Skip classical values
    if (!isa<quake::WireType>(v.getType()))
      return nullptr;

    auto defOp = v.getDefiningOp();
    if (defOp)
      return visitOp(defOp, next);

    // TODO: FAIL
    assert(false && "UKKNOWN VALUE");
    return nullptr;
  }

  /// Given a set of qubit lifetimes and a candidate lifetime,
  /// tries to find a qubit to reuse.
  /// The result either contains the optimal qubit to reuse,
  /// or contains no value if no qubit can be reused
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

  /// Reorders the program based on the dependency graphs to reuse qubits
  void codeGen(SmallVector<DependencyGraph> &graphs, OpBuilder &builder) {
    SmallVector<LifeTime *> lifetimes;
    SmallVector<DependencyNode *> sinks;
    uint cycles = getTotalCycles(graphs);

    for (uint cycle = 0; cycle < cycles; cycle++) {
      for (auto graph : graphs) {
        // For every "new" qubit, try to find an existing out-of-use qubit
        // that we can reuse. Failing that, use a new qubit.
        for (auto qid : graph.getFirstUsedAtCycle(cycle)) {
          auto lifetime = graph.getLifeTimeForQID(qid);
          LLVM_DEBUG(llvm::dbgs() << "Qid " << qid);
          LLVM_DEBUG(llvm::dbgs()
                     << " is in use from cycle " << lifetime->getBegin());
          LLVM_DEBUG(llvm::dbgs() << " through cycle " << lifetime->getEnd());
          LLVM_DEBUG(llvm::dbgs() << "\n\n");

          auto new_qid = findBestQubit(lifetimes, lifetime);
          if (!new_qid) {
            // Can't reuse any qubits, have to allocate a new one
            new_qid = lifetimes.size();
            lifetimes.push_back(lifetime);
            sinks.push_back(graph.getRootForQID(qid));
            // Initialize the qubit with a null wire op
            graph.initializeWire(qid, builder);
          } else {
            // We found a qubit we can reuse!
            lifetimes[new_qid.value()] =
                lifetime->combine(lifetimes[new_qid.value()]);

            auto last_user = sinks[new_qid.value()];
            // We assume that the result of the last use of the old qubit
            // must have been a null wire (e.g., it was reset),
            // so we can reuse that result as the initial value to reuse it
            graph.initializeWireFromRoot(qid, last_user);
            sinks[new_qid.value()] = graph.getRootForQID(qid);
          }
        }

        graph.codeGenAt(cycle, builder);
      }
    }

    // Add teardown instructions
    for (auto sink : sinks)
      sink->addCleanUp(builder);
  }

  void runOnOperation() override {
    auto func = getOperation();

    SetVector<DependencyNode *> roots;
    for (auto &op : func.front().getOperations()) {
      if (dyn_cast<func::ReturnOp>(op))
        continue;
      auto root = visitOp(&op, nullptr);
      if (isEndOp(&op))
        roots.insert(root);
    }

    // Construct graphs from roots
    SmallVector<DependencyGraph> graphs;
    while (!roots.empty()) {
      DependencyGraph new_graph(roots.front());
      roots.set_subtract(new_graph.getRoots());
      graphs.push_back(new_graph);
    }

    // Setup new block to replace function body
    OpBuilder builder(func.getOperation());
    Block *oldBlock = &func.front();
    Block *newBlock = builder.createBlock(&func.getRegion());
    SmallVector<mlir::Location> locs;
    for (auto arg : oldBlock->getArguments())
      locs.push_back(arg.getLoc());
    newBlock->addArguments(oldBlock->getArgumentTypes(), locs);
    builder.setInsertionPointToStart(newBlock);
    // Generate optimized instructions in new block
    codeGen(graphs, builder);
    builder.create<func::ReturnOp>(builder.getUnknownLoc());
    // Replace old block
    oldBlock->erase();
  }
};

} // namespace
