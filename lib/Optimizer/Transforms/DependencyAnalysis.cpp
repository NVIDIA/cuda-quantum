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
  LifeTime(uint _begin, uint _end) : begin(_begin), end(_end) {
    assert(_end >= _begin && "invalid lifetime");
  };

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
  // Dependencies and successors are ordered lists:
  // successors in the order of results of associated,
  SmallVector<DependencyNode *> successors;
  // dependencies are in the order of operands of associated
  SmallVector<DependencyNode *> dependencies;
  // If a given dependency appears multiple times,
  // (e.g., multiple results of the dependency are used by associated),
  // it is important to know which result from the dependency
  // corresponds to which operand of associated.
  // Otherwise, the dependency will be code gen'ed first, and it will
  // be impossible to know (e.g., which result is a control and which is a
  // target). Result_idxs tracks this information, and therefore should be
  // exactly the same size as dependencies.
  SmallVector<size_t> result_idxs;
  SetVector<size_t> qids;
  Operation *associated;
  uint cycle = INT_MAX;
  bool hasCodeGen = false;
  uint height;

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

  bool isRoot() { return isEndOp(associated); }

  bool isLeaf() { return isBeginOp(associated); }

  bool isSkip() { return isRoot() || isLeaf(); }

  void schedule(uint level) {
    if (level < height)
      level = height;

    uint current = level;
    if (!isSkip()) {
      current--;
      cycle = current;
    }

    for (auto dependency : dependencies)
      if (dependency && !dependency->isScheduled() && !dependency->isLeaf())
        dependency->schedule(current);

    for (auto successor : successors)
      if (successor && !successor->isScheduled() && !successor->isRoot())
        successor->schedule(level + 1);
  }

  /// This function returns the index of the result of the dependency
  /// corresponding to the operand_idx'th operand of this node
  size_t getResultIdx(size_t operand_idx) { return result_idxs[operand_idx]; }

  /// This function returns the index of the operand of \p successor
  /// corresponding to the result_idx'th result of this node
  size_t getOperandIdx(size_t result_idx) {
    auto successor = successors[result_idx];
    assert(successor && "Cannot access operand of null successor");
    size_t operand = 0;
    auto operands = successor->associated->getOperands();
    for (; operand < operands.size(); operand++)
      if (operands[operand] == associated->getResult(result_idx))
        break;
    return operand;
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
    if (hasCodeGen || isRoot() || isLeaf())
      return;

    auto oldOp = associated;
    SmallVector<mlir::Value> operands(oldOp->getNumOperands());
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto dependency = dependencies[i];
      if (!dependency)
        continue;

      // Get relevant result from dependency's op
      // to use as the relevant operand
      auto result_idx = getResultIdx(i);
      operands[i] = dependency->associated->getResult(result_idx);
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

    if (isLeaf()) {
      associated = init;
      return;
    }

    for (auto dependency : dependencies)
      if (dependency)
        dependency->initializeWire(qid, init);
  }

  /// Ensures that the node is valid and has been scheduled.
  /// This should only be called after it has been added to a
  /// dependency graph.
  /// This is an expensive check of internal assumptions that will
  /// only be available during debugging.
  void validate() {
    // Validate metadata
    assert(associated && "Associated op is null");
    if (!isSkip()) {
      assert(isScheduled() && "Node part of graph but not scheduled");
      // A node is included in the calculation of its height, hence + 1
      assert(cycle + 1 >= height && "Node scheduled too early");
    }
    assert(dependencies.size() == associated->getNumOperands() &&
           "Wrong number of dependencies");
    assert(successors.size() == associated->getNumResults() &&
           "Wrong number of successors");
    // Ensure that the dependencies all make sense
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto dependency = dependencies[i];
      auto operand = associated->getOperand(i);
      if (!dependency) {
        assert(!isa<quake::WireType>(operand.getType()) &&
               "Successor for wire result cannot be null");
        continue;
      }

      auto expected = operand.getDefiningOp();
      if (!dependency->hasCodeGen)
        assert(dependency->associated == expected &&
               "Dependencies in wrong order");
    }
    // Ensure that the successors all make sense
    for (size_t i = 0; i < successors.size(); i++) {
      auto successor = successors[i];
      auto result = associated->getResult(i);
      if (!successor) {
        assert(!isa<quake::WireType>(result.getType()) &&
               "Successor for wire result cannot be null");
        continue;
      }

      auto operand_idx = getOperandIdx(i);
      if (!hasCodeGen)
      assert(successor->associated->getOperand(operand_idx) == result &&
             "Successors in wrong order");
    }
  }

public:
  DependencyNode(Operation *op, SmallVector<DependencyNode *> _dependencies)
      : dependencies(_dependencies), qids(), associated(op) {
    assert(op && "Cannot make dependency node for null op");
    assert(_dependencies.size() == op->getNumOperands() &&
           "Wrong # of dependencies to construct node");
    uint num_successors = op->getNumResults();
    successors = SmallVector<DependencyNode *>(num_successors, nullptr);
    result_idxs = SmallVector<size_t>(dependencies.size(), INT_MAX);

    if (isBeginOp(op) || isEndOp(op)) {
      // Should be ensured by assign-ids pass
      assert(op->hasAttr("qid") && "quake.null_wire or quake.sink missing qid");

      // Lookup qid
      auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
      qids.insert(qid);
    }

    height = 0;
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto dependency = dependencies[i];

      if (!dependency)
        continue;

      // Figure out result_idx
      size_t result_idx = 0;
      auto results = dependency->associated->getResults();
      for (; result_idx < results.size(); result_idx++)
        if (results[result_idx] == associated->getOperand(i))
          break;

      result_idxs[i] = result_idx;
      // Set relevant successor of dependency to this
      dependency->successors[result_idx] = this;
      // Update metadata
      if (dependency->height > height)
        height = dependency->height;
      if (!isEndOp(op))
        qids.set_union(dependency->qids);
    }

    if (!isSkip())
      height++;
  };

  void print() { printSubGraph(0); }

  uint getHeight() { return height; }

  void addCleanUp(OpBuilder &builder) {
    assert(isRoot() && isEndOp(associated) &&
           "Can only call addCleanUp on a root node");
    auto last_use = dependencies[0];
    auto result_idx = getResultIdx(0);
    auto wire = last_use->associated->getResult(result_idx);
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
  DependencyNode *tallest = nullptr;

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
      if (next->height > total_height || tallest == nullptr) {
        tallest = next;
        total_height = next->height;
      }
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

  void scheduleNodes() { tallest->schedule(total_height); }

  SetVector<DependencyNode *> getNodesAtCycle(uint cycle) {
    SetVector<DependencyNode *> nodes;
    for (auto root : roots)
      nodes.set_union(root->getNodesAtCycle(cycle));
    return nodes;
  }

  /// Ensures that the node is valid and is scheduled properly.
  /// This is an expensive check that should only be used for
  /// testing/debugging.
  void validateNode(DependencyNode *node, uint parent_cycle) {
    assert(node && "Null node in graph");
    if (!node->isSkip()) {
      assert(node->cycle < parent_cycle && "Node scheduled too late");
      parent_cycle = node->cycle;
    }
    node->validate();
    for (auto dependency : node->dependencies)
      validateNode(dependency, parent_cycle);
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

  /// Ensures that the graph is valid.
  /// This is an expensive check that should only be used for
  /// testing/debugging.
  void validate() {
    for (auto root : roots)
      validateNode(root, total_height);
  }
};

struct DependencyAnalysisPass
    : public cudaq::opt::impl::DependencyAnalysisBase<DependencyAnalysisPass> {
  using DependencyAnalysisBase::DependencyAnalysisBase;
  SmallVector<DependencyNode *> perOp;

  /// Validates that \p op meets the assumptions:
  /// * control flow operations are not allowed
  bool validateOp(Operation *op) {
    if (!quake::isLinearValueForm(op)) {
      op->emitOpError("dep-analysis requires all operations to be in value form");
      signalPassFailure();
      return false;
    }

    if (op->getRegions().size() != 0) {
      op->emitOpError(
          "control flow operations not currently supported in dep-analysis");
      signalPassFailure();
      return false;
    }

    if (dyn_cast<mlir::BranchOpInterface>(op)) {
      op->emitOpError(
          "branching operations not currently supported in dep-analysis");
      signalPassFailure();
      return false;
    }

    return true;
  }

  /// Validates that \p func meets the assumptions:
  /// * function bodies contain a single block
  /// * functions have no arguments
  /// * functions have no results
  bool validateFunc(func::FuncOp func) {
    if (func.getBlocks().size() != 1) {
      func.emitOpError("multiple blocks not currently supported in dep-analysis");
      signalPassFailure();
      return false;
    }

    if (func.getArguments().size() != 0) {
      func.emitOpError(
          "function arguments not currently supported in dep-analysis");
      signalPassFailure();
      return false;
    }

    if (func.getNumResults() != 0) {
      func.emitOpError(
          "non-void return types not currently supported in dep-analysis");
      signalPassFailure();
      return false;
    }
    return true;
  }

  /// Creates and returns a new dependency node for \p op, connecting it to the
  /// nodes created for the defining operations of the operands of \p op
  DependencyNode *visitOp(Operation *op) {
    if (!validateOp(op))
      return nullptr;

    SmallVector<DependencyNode *> dependencies(op->getNumOperands());
    for (uint i = 0; i < op->getNumOperands(); i++) {
      auto dependency = visitValue(op->getOperand(i));
      assert(dependency && "dependency node not found for dependency");
      dependencies[i] = dependency;
    }

    DependencyNode *newNode = new DependencyNode(op, dependencies);

    // Dnodeid is the next slot of the dnode vector
    auto id = perOp.size();

    // Add dnodeid attribute
    OpBuilder builder(op);
    op->setAttr("dnodeid", builder.getUI32IntegerAttr(id));
    perOp.push_back(newNode);

    return newNode;
  }

  /// Returns the dependency node for the defining operation of \p v
  /// Assumption: defining operation for \p v exists and already has been
  /// visited
  DependencyNode *visitValue(Value v) {
    // Skip classical values
    if (!isa<quake::WireType>(v.getType()))
      return nullptr;

    auto defOp = v.getDefiningOp();
    if (defOp) {
      // Since we walk forward through the ast, every value should be defined
      // before it is used, so we should have already visited defOp,
      // and thus should have a memoized dnode for defOp, fail if not
      assert(defOp->hasAttr("dnodeid") &&
             "Error: no dnodeid found for operation");

      auto id = defOp->getAttr("dnodeid").cast<IntegerAttr>().getUInt();
      auto dnode = perOp[id];
      return dnode;
    }

    // This means that v is a block argument which is not allowed
    // Return null so the error can be handled nicely by visitOp
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
    validateFunc(func);

    SetVector<DependencyNode *> roots;
    for (auto &op : func.front().getOperations()) {
      if (dyn_cast<func::ReturnOp>(op))
        continue;

      auto node = visitOp(&op);

      if (!node) {
        signalPassFailure();
        return;
      }

      if (isEndOp(&op))
        roots.insert(node);
    }

    // Construct graphs from roots
    SmallVector<DependencyGraph> graphs;
    while (!roots.empty()) {
      DependencyGraph new_graph(roots.front());
      roots.set_subtract(new_graph.getRoots());
      graphs.push_back(new_graph);
    }

    // Validate the graphs only in debug mode
    LLVM_DEBUG(for (auto graph : graphs) graph.validate(););

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
