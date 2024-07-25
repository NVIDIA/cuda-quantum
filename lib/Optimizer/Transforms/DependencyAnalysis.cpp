/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
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
  LifeTime(uint begin, uint end) : begin(begin), end(end) {
    assert(end >= begin && "invalid lifetime");
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

  void combine(LifeTime *other) {
    begin = std::min(begin, other->begin);
    end = std::max(end, other->end);
  }

  uint getBegin() { return begin; }
  uint getEnd() { return end; }
};

class LifeTimeSet {
private:
  StringRef set;
  SmallVector<SmallVector<LifeTime *>> lifetimes;
  SmallVector<SetVector<size_t>> frames;
  size_t width;

  bool isInCurrentFrame(size_t pqid) {
    return frames.back().contains(pqid);
  }

  void combineOrAdd(size_t pqid, LifeTime *lifetime) {
    if (isInCurrentFrame(pqid))
      lifetimes[pqid].back()->combine(lifetime);
    else
      lifetimes[pqid].push_back(lifetime);
  }
  
  /// Given a set of qubit lifetimes and a candidate lifetime,
  /// tries to find a qubit to reuse, otherwise allocates a new qubit
  size_t mapToPhysical(LifeTime *lifetime) {
    std::optional<size_t> best_reuse = std::nullopt;
    std::optional<size_t> empty = std::nullopt;
    uint best_distance = INT_MAX;
    for (uint i = 0; i < lifetimes.size(); i++) {
      if (lifetimes[i].empty()) {
        empty = i;
        continue;
      }

      LifeTime *other = lifetimes[i].back();
      auto distance = lifetime->distance(other);
      if (lifetime->isAfter(other) && distance < best_distance) {
        best_reuse = i;
        best_distance = distance;
      }
    }

    // Reuse a qubit based on a lifetime in the same scope
    if (best_reuse.has_value()) {
      combineOrAdd(best_reuse.value(), lifetime);
      return best_reuse.value();
    }

    // Reuse a qubit based on a lifetime in a different scope
    if (empty.has_value()) {
      lifetimes[empty.value()].push_back(lifetime);
      return empty.value();
    }

    // Fall back: allocate a new qubit
    lifetimes.push_back({lifetime});
    if (lifetimes.size() > width)
        width = lifetimes.size();
    return lifetimes.size() - 1;
  }

public:
  LifeTimeSet(StringRef set) : set(set), lifetimes(), frames(), width(0) {}

  quake::BorrowWireOp genBorrow(LifeTime *lifetime, size_t qid, OpBuilder &builder) {
    auto phys = mapToPhysical(lifetime);
    frames.back().insert(phys);

    auto wirety = quake::WireType::get(builder.getContext());
    return builder.create<quake::BorrowWireOp>(builder.getUnknownLoc(), wirety, set, phys);
  }

  void pushFrame() {
    frames.push_back({});
  }

  SetVector<size_t> popFrame() {
    auto pqids = frames.back();
    frames.pop_back();
    for (auto pqid : pqids) {
      lifetimes[pqid].pop_back();
    }
    return pqids;
  }

  void addOpaque(SetVector<size_t> pqids, LifeTime *lifetime) {
    for (auto pqid : pqids) {
      combineOrAdd(pqid, lifetime);
      frames.back().insert(pqid);
    }
  }

  size_t getCount() { return width; }

  void print() {
    llvm::outs() << "# qubits: " << width << ", # frames: " << frames.size() << ", cycles: ";
    for (size_t i = 0; i < lifetimes.size(); i++)
      if (lifetimes[i].empty())
        llvm::outs() << "E ";
      else
        llvm::outs() << lifetimes[i].back()->getEnd() << " ";
    llvm::outs() << "\n";
  }
};

class DependencyNode {
  friend class DependencyGraph;
  friend class OpDependencyNode;
  friend class IfDependencyNode;
  friend class ArgDependencyNode;
  friend class RootDependencyNode;
protected:
  SetVector<DependencyNode *> successors;
  // Dependencies are in the order of operands
  SmallVector<DependencyNode *> dependencies;
  // If a given dependency appears multiple times,
  // (e.g., multiple results of the dependency are used by this node),
  // it is important to know which result from the dependency
  // corresponds to which operand.
  // Otherwise, the dependency will be code gen'ed first, and it will
  // be impossible to know (e.g., which result is a control and which is a
  // target). Result_idxs tracks this information, and therefore should be
  // exactly the same size/order as dependencies.
  SmallVector<size_t> result_idxs;
  SetVector<size_t> qids;
  uint cycle = INT_MAX;
  bool hasCodeGen = false;
  uint height;
  bool isScheduled;

  virtual void printNode() = 0;

  void printSubGraph(int tabIndex) {
    for (int i = 0; i < tabIndex; i++) {
      llvm::outs() << "\t";
    }

    printNode();

    for (auto dependency : dependencies)
      dependency->printSubGraph(tabIndex + 1);
  }

  virtual bool isAlloc() { return false; }
  virtual bool isRoot() { return successors.size() == 0; };
  virtual bool isLeaf() { return dependencies.size() == 0; };
  virtual bool isSkip() { return numTicks() == 0; };
  virtual bool isQuantumDependent() {
    return qids.size() > 0;
  };
  virtual bool isQuantumOp() = 0;
  virtual uint numTicks() = 0;
  virtual Value getResult(uint resultidx) = 0;
  virtual ValueRange getResults() = 0;
  virtual ValueRange getOperands() = 0;
  virtual void codeGen(OpBuilder &builder, LifeTimeSet &set) = 0;
  
  /// Returns the index of the result of the dependency corresponding to the
  /// \p operand_idx'th operand of this node
  virtual size_t getResultIdx(size_t operand_idx) {
    return result_idxs[operand_idx];
  }

  /// Recursively find nodes scheduled at a given cycle
  SetVector<DependencyNode *> getNodesAtCycle(uint _cycle) {
    SetVector<DependencyNode *> nodes;

    if (cycle < _cycle)
      return nodes;
    else if (cycle == _cycle && !isSkip())
      nodes.insert(this);

    for (auto dependency : dependencies)
      nodes.set_union(dependency->getNodesAtCycle(_cycle));

    return nodes;
  }

  /// Replaces the null_wire op for \p qid with \p init
  virtual bool initializeWire(size_t qid, Value v) {
    if (!qids.contains(qid))
      return false;

    for (auto dependency : dependencies)
      if (dependency->initializeWire(qid, v))
        return true;

    return false;
  }

  /// This function guarantees that nodes are scheduled after their predecessors
  /// and before their successors, and that every node is scheduled at a cycle
  /// between 0 and the height of the graph to which they belong.
  ///
  /// The scheduling algorithm works by always following the longest path first.
  /// The longest path will always be "saturated" with an operation every cycle,
  /// so we know exactly when to schedule every operation along that path.
  /// Then, every successor (not on the path) of an operation on the path should
  /// be scheduled as early as possible, (the earliest an operation can be
  /// scheduled is determined by its height). Likewise, every dependency (not on
  /// the path) should be scheduled as late as possible. Because we work
  /// outwards from the longest path, this ensures that every other path is
  /// scheduled as "densely" as possible around the connections with the longest
  /// path, while still having a valid schedule.
  ///
  /// Always following the longest path first is essentially an implementation
  /// of a transitive reduction of the graph. The only auxiliary data structure
  /// used here is a sorted copy of the dependency list. The length of a path
  /// is equal to the height of the node which is metadata present from
  /// construction.
  ///
  /// \p level is essentially the depth from the tallest point in the graph
  void schedule(uint level) {
    isScheduled = true;
    // Ignore classical values that don't depend on quantum values
    if (!isQuantumDependent())
      return;

    // The height of a node (minus numTicks()) is the earliest a node can be
    // scheduled
    if (level < height)
      level = height;

    uint current = level;
    if (!isSkip()) {
      current -= numTicks();
      cycle = current;
    }

    // Sort dependencies by height to always follow the longest path first.
    // Without this, two dependencies may be scheduled at the same cycle,
    // even if one of the dependencies depends on the other.
    // This sort of mimics working over a transitive reduction of the graph.
    SmallVector<DependencyNode *> sorted(dependencies);
    std::sort(sorted.begin(), sorted.end(),
              [](DependencyNode *x, DependencyNode *y) {
                return x->getHeight() > y->getHeight();
              });

    // Schedule dependencies as late as possible (right before this operation)
    for (auto dependency : sorted)
      if (!dependency->isScheduled && !dependency->isLeaf())
        dependency->schedule(current);

    // Schedule unscheduled successors as early as possible
    for (auto successor : successors)
      if (!successor->isScheduled && !successor->isRoot())
        successor->schedule(current + numTicks() + successor->numTicks());
  }

public:
  DependencyNode() : successors(), dependencies({}), result_idxs({}),
  qids({}), height(0), isScheduled(false) {}

  uint getHeight() { return height; };

  void print() { printSubGraph(0); }

  virtual void genReturnWire(OpBuilder &builder, LifeTimeSet &set) {
    assert(false && "Called genReturnWire on a non root");
  };

  virtual void genTerminator(OpBuilder &builder, LifeTimeSet &set) {
    assert(false && "Called genTerminator on a non terminator");
  };
};


class DependencyGraph {
private:
  SetVector<DependencyNode *> roots;
  SetVector<size_t> qids;
  SetVector<size_t> allocs;
  uint total_height;
  DenseMap<size_t, DependencyNode *> firstUses;
  bool isScheduled = false;
  DependencyNode *tallest = nullptr;
  uint shift;

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

    if (next->isLeaf() && next->isQuantumOp())
      firstUses.insert({next->qids.front(), next->successors.front()});

    if (next->isAlloc())
      allocs.set_union(next->qids);

    for (auto successor : next->successors)
      if (next->isQuantumDependent() || !successor->isQuantumDependent())
        gatherRoots(seen, successor);
    for (auto dependency : next->dependencies)
      gatherRoots(seen, dependency);
  }

  void scheduleNodes() { tallest->schedule(total_height); }

  SetVector<DependencyNode *> getNodesAtCycle(uint cycle) {
    SetVector<DependencyNode *> nodes;
    for (auto root : roots)
      nodes.set_union(root->getNodesAtCycle(cycle));
    return nodes;
  }

public:
  DependencyGraph(DependencyNode *root) {
    shift = 0;
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
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    uint first = getFirstUseOf(qid)->cycle + shift;
    auto last = getLastUseOf(qid)->cycle + shift;

    return new LifeTime(first, last);
  }

  DependencyNode *getFirstUseOf(size_t qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    return firstUses[qid];
  }

  DependencyNode *getLastUseOf(size_t qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    return getRootForQID(qid)->dependencies[0];
  }

  DependencyNode *getRootForQID(size_t qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    for (auto root : roots)
      if (root->qids.contains(qid))
        return root;
    return nullptr;
  }

  void codeGenAt(uint cycle, OpBuilder &builder, LifeTimeSet &set) {
    SetVector<DependencyNode *> nodes = getNodesAtCycle(cycle);

    for (auto node : nodes)
      node->codeGen(builder, set);
  }

  uint getHeight() { return total_height; }

  SmallVector<size_t> getFirstUsedAtCycle(uint cycle) {
    SmallVector<size_t> fresh;
    for (auto qid : allocs) {
      auto first = getFirstUseOf(qid);
      if (first->cycle == cycle)
        fresh.push_back(qid);
    }

    return fresh;
  }

  SmallVector<size_t> getLastUsedAtCycle(uint cycle) {
    SmallVector<size_t> stale;
    for (auto qid : qids) {
      auto last = getLastUseOf(qid);
      if (last->cycle == cycle)
        stale.push_back(qid);
    }

    return stale;
  }

  /// Creates a new physical null wire to replace the
  /// "virtual" qubit represented by \p qid
  void initializeWire(size_t qid, Value wire) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    auto res = getFirstUseOf(qid)->initializeWire(qid, wire);
    assert(res && "Initializing wire failed");
  }

  void print() {
    llvm::outs() << "Graph Start\n";
    for (auto root : roots)
      root->print();
    llvm::outs() << "Graph End\n";
  }

  void genReturn(size_t qid, OpBuilder &builder, LifeTimeSet &set) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    getRootForQID(qid)->genReturnWire(builder, set);
  }

  // TODO: Is there a better way to get lifetimes to align
  void setCycle(uint cycle) {
    this->shift = cycle;
  }
};

class OpDependencyNode : public DependencyNode {
protected:
  Operation *associated;
  bool quantumOp;

  void printNode() override {
    llvm::outs() << "QIDs: ";
    bool printComma = false;
    for (auto qid : qids) {
      if (printComma)
        llvm::outs() << ", ";
      llvm::outs() << qid;
      printComma = true;
    }
    if (isScheduled)
      llvm::outs() << " @ " << cycle;
    llvm::outs() << " | " << height << ", " << numTicks() << " | ";
    associated->dump();
  }

  uint numTicks() override { return isQuantumOp() ? 1 : 0; }
  bool isQuantumOp() override { return quantumOp; }

  Value getResult(uint resultidx) override {
    return associated->getResult(resultidx);
  }

  ValueRange getResults() override {
    return associated->getResults();
  }

  ValueRange getOperands() override {
    return associated->getOperands();
  }

  SmallVector<mlir::Value> gatherOperands(size_t num, OpBuilder &builder, LifeTimeSet &set) {
    SmallVector<mlir::Value> operands(num);
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto dependency = dependencies[i];

      // Ensure classical values are available
      if (!dependency->isQuantumOp())
        dependency->codeGen(builder, set);

      assert(dependency->hasCodeGen &&
             "Generating code for successor before dependency");

      // Get relevant result from dependency's updated op
      // to use as the relevant operand
      auto result_idx = getResultIdx(i);
      operands[i] = dependency->getResult(result_idx);
    }
    return operands;
  }

  /// Generates a new operation for this node in the dependency graph
  /// using the dependencies of the node as operands.
  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {
    if (hasCodeGen)
      return;

    auto oldOp = associated;
    auto operands = gatherOperands(oldOp->getNumOperands(), builder, set);

    associated =
        Operation::create(oldOp->getLoc(), oldOp->getName(),
                          oldOp->getResultTypes(), operands, oldOp->getAttrs());
    associated->removeAttr("dnodeid");
    builder.insert(associated);
    hasCodeGen = true;
  }

public:
  OpDependencyNode(Operation *op, SmallVector<DependencyNode *> _dependencies)
      : associated(op) {
    dependencies = _dependencies;
    assert(op && "Cannot make dependency node for null op");
    assert(_dependencies.size() == op->getNumOperands() &&
           "Wrong # of dependencies to construct node");
    result_idxs = SmallVector<size_t>(dependencies.size(), INT_MAX);

    quantumOp = isQuakeOperation(op);
    if (dyn_cast<quake::DiscriminateOp>(op))
      quantumOp = false;

    height = 0;
    // Ingest dependencies, setting up metadata
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto dependency = dependencies[i];
      auto operand = associated->getOperand(i);

      assert(dependency && "Invalid dependency");

      // Figure out result_idx
      size_t result_idx = 0;
      auto results = dependency->getResults();
      for (; result_idx < results.size(); result_idx++)
        if (results[result_idx] == operand)
          break;

      assert(result_idx < results.size() &&
             "Node passed as dependency isn't actually a dependency!");

      result_idxs[i] = result_idx;
      // Add this as a successor to each dependency
      dependency->successors.insert(this);

      // Update metadata
      if (dependency->height > height)
        height = dependency->height;
      if (!isEndOp(op))
        qids.set_union(dependency->qids);
    }

    height += numTicks();
  };

  void print() { printSubGraph(0); }

  uint getHeight() { return height; }
};

class InitDependencyNode : public DependencyNode {
protected:
  Value wire;

  void printNode() override {
    llvm::outs() << "Initial value for QID ";
    for (auto qid : qids)
      llvm::outs() << qid;
    llvm::outs() << ": ";
    wire.dump();
  }

  bool isAlloc() override { return true; }
  bool isQuantumDependent() override { return true; }
  uint numTicks() override { return 0; }
  bool isQuantumOp() override { return true; }

  Value getResult(uint resultidx) override {
    assert(resultidx == 0 && "Illegal resultidx");
    return wire;
  }

  ValueRange getResults() override {
    return ValueRange({wire});
  }

  ValueRange getOperands() override {
    return ValueRange({});
  }

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {}

  /// Replaces the null_wire op for \p qid with \p init
  bool initializeWire(size_t qid, Value v) override {
    if (!qids.contains(qid))
      return false;

    wire = v;
    hasCodeGen = true;
    return true;
  }

public:
  InitDependencyNode(quake::NullWireOp op) : wire(op.getResult()) {
    // Should be ensured by assign-ids pass
    assert(op->hasAttr("qid") && "quake.null_wire missing qid");

    // Lookup qid
    auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
    qids.insert(qid);
  };
};

class RootDependencyNode : public OpDependencyNode {
protected:
  void printNode() override {
    llvm::outs() << "Sink for QID ";
    for (auto qid : qids)
      llvm::outs() << qid;
    llvm::outs() << ": ";
    associated->dump();
  }

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {}

public:
  RootDependencyNode(quake::SinkOp op, SmallVector<DependencyNode *> dependencies)
      : OpDependencyNode(op, dependencies){
    // Should be ensured by assign-ids pass
    assert(op->hasAttr("qid") && "quake.sink missing qid");

    qids.clear();

    // Lookup qid
    auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
    qids.insert(qid);
  };

  void genReturnWire(OpBuilder &builder, LifeTimeSet &set) override {
    auto result_idx = getResultIdx(0);
    auto wire = dependencies[0]->getResult(result_idx);
    auto newOp = builder.create<quake::ReturnWireOp>(builder.getUnknownLoc(), wire);
    newOp->setAttrs(associated->getAttrs());
    associated->removeAttr("dnodeid");
    associated = newOp;
    hasCodeGen = true;
  }
};

class ArgDependencyNode : public DependencyNode {
  friend class DependencyBlock;
protected:
  BlockArgument barg;

  void printNode() override { barg.dump(); }

  bool isRoot() override { return false; }
  bool isLeaf() override { return true; }
  bool isQuantumOp() override { return quake::isQuantumType(barg.getType()); }
  uint numTicks() override { return 0; }

  Value getResult(uint resultidx) override {
    assert(resultidx == 0 && "Invalid resultidx");
    return barg;
  }

  ValueRange getResults() override {
    return ValueRange({barg});
  }

  ValueRange getOperands() override {
    return ValueRange({});
  }

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override { };

public:
  ArgDependencyNode(BlockArgument arg, DependencyNode *val)
    : barg(arg) {
      if (!val)
        return;
      qids = val->qids;
    }
};

class DependencyBlock {
private:
  SmallVector<ArgDependencyNode *> argdnodes;
  SmallVector<DependencyGraph *> graphs;
  Block *block;
  DependencyNode *terminator;
  uint height;
  SetVector<size_t> pqids;

public:
  DependencyBlock(SmallVector<ArgDependencyNode *> argdnodes,
                  SmallVector<DependencyGraph *> graphs, Block *block, DependencyNode *terminator)
    : argdnodes(argdnodes), graphs(graphs), block(block), terminator(terminator), pqids() {
    height = 0;
    for (auto graph : graphs)
      if (graph->getHeight() > height)
        height = graph->getHeight();
  }

  uint getHeight() {
    return height;
  }

  void setCycle(uint cycle) {
    for (auto graph : graphs)
      graph->setCycle(cycle);
  }

  /// Up to caller to move builder outside block after construction
  Block *codeGen(OpBuilder &builder, Region *region, LifeTimeSet &set) {
    set.pushFrame();
    Block *newBlock = builder.createBlock(region);
    SmallVector<mlir::Location> locs;
    for (auto arg : block->getArguments())
      locs.push_back(arg.getLoc());
    newBlock->addArguments(block->getArgumentTypes(), locs);
    for (uint i = 0; i < newBlock->getNumArguments(); i++) {
      argdnodes[i]->barg = newBlock->getArgument(i);
      argdnodes[i]->hasCodeGen = true;
    }
    builder.setInsertionPointToStart(newBlock);

    for (uint cycle = 0; cycle < height; cycle++) {
      for (auto graph : graphs) {
        // For every "new" qubit, try to find an existing out-of-use qubit
        // that we can reuse. Failing that, use a new qubit.
        for (auto qid : graph->getFirstUsedAtCycle(cycle)) {
          auto lifetime = graph->getLifeTimeForQID(qid);
          LLVM_DEBUG(llvm::dbgs() << "Qid " << qid);
          LLVM_DEBUG(llvm::dbgs()
                      << " is in use from cycle " << lifetime->getBegin());
          LLVM_DEBUG(llvm::dbgs() << " through cycle " << lifetime->getEnd());
          LLVM_DEBUG(llvm::dbgs() << "\n\n");

          auto borrowOp = set.genBorrow(lifetime, qid, builder);
          graph->initializeWire(qid, borrowOp.getResult());
        }

        graph->codeGenAt(cycle, builder, set);

        for (auto qid : graph->getLastUsedAtCycle(cycle))
          graph->genReturn(qid, builder, set);
      }
    }

    terminator->genTerminator(builder, set);

    block = newBlock;
    pqids = set.popFrame();

    return newBlock;
  }

  SetVector<size_t> getPQids() {
    return pqids;
  }

  void print() {
    llvm::outs() << "Block:\n";
    block->dump();
    llvm::outs() << "Block graphs:\n";
    for (auto graph : graphs)
      graph->print();
    llvm::outs() << "End block\n";
  }
};

class IfDependencyNode : public OpDependencyNode {
  friend class ArgDependencyNode;
protected:
  DependencyBlock *then_block;
  DependencyBlock *else_block;

  // TODO: figure out nice way to display
  /*void printNode() override {
    this->OpDependencyNode::printNode();
    //llvm::outs() << "Then: ";
    //then_block->print();
    //llvm::outs() << "Else: ";
    //else_block->print();
  }*/

  uint numTicks() override {
    return std::max(then_block->getHeight(), else_block->getHeight());
  }

  bool isQuantumOp() override {
    for (auto type : associated->getResultTypes())
      if (quake::isQuantumType(type))
        return true;

    return numTicks() > 0;
  }

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {
    if (hasCodeGen)
      return;

    then_block->setCycle(cycle);
    else_block->setCycle(cycle);

    cudaq::cc::IfOp oldOp = dyn_cast<cudaq::cc::IfOp>(associated);
    auto operands = gatherOperands(oldOp->getNumOperands(), builder, set);

    auto newif = builder.create<cudaq::cc::IfOp>(oldOp->getLoc(), oldOp->getResultTypes(), operands);
    auto *then_region = &newif.getThenRegion();
    then_block->codeGen(builder, then_region, set);

    auto *else_region = &newif.getElseRegion();
    else_block->codeGen(builder, else_region, set);

    auto pqids = then_block->getPQids();
    pqids.set_union(else_block->getPQids());
    set.addOpaque(pqids, new LifeTime(cycle, cycle+numTicks()));

    associated = newif;
    builder.setInsertionPointAfter(associated);
    hasCodeGen = true;
  };

public:
  IfDependencyNode(cudaq::cc::IfOp op, SmallVector<DependencyNode *> dependencies,
                   DependencyBlock *then_block, DependencyBlock *else_block)
    : OpDependencyNode(op.getOperation(), dependencies), then_block(then_block),
      else_block(else_block) {
        // Num ticks won't be properly calculated by OpDependencyNode constructor
        // So have to recompute height here
        height = 0;
        for (auto dependency : dependencies)
          if (dependency->height > height)
            height = dependency->height;
        height += numTicks();
      }
};

class TerminatorDependencyNode : public OpDependencyNode {
protected:
  void printNode() override {
    llvm::outs() << "Block Terminator With QIDs ";
    bool printComma = false;
    for (auto qid : qids) {
      if (printComma)
        llvm::outs() << ", ";
      llvm::outs() << qid;
      printComma = true;
    }
    llvm::outs() << ": ";
    associated->dump();
  }

public:
  TerminatorDependencyNode(Operation *terminator,
                           SmallVector<DependencyNode *> dependencies)
    : OpDependencyNode(terminator, dependencies) {
      assert(terminator->hasTrait<mlir::OpTrait::ReturnLike>() && "Invalid terminator");
    }

  void genReturnWire(OpBuilder &builder, LifeTimeSet &set) override {
    // Cleanup is someone elses responsibility
  };

  void genTerminator(OpBuilder &builder, LifeTimeSet &set) override {
    codeGen(builder, set);
  }
};

/// Validates that \p op meets the assumptions:
/// * control flow operations are not allowed
bool validateOp(Operation *op) {
  if (isQuakeOperation(op) && !quake::isLinearValueForm(op) &&
      !dyn_cast<quake::DiscriminateOp>(op)) {
    op->emitOpError(
        "dep-analysis requires all quake operations to be in value form");
    return false;
  }

  if (op->getRegions().size() != 0 && !dyn_cast<cudaq::cc::IfOp>(op)) {
    op->emitOpError(
        "control flow operations not currently supported in dep-analysis");
    return false;
  }

  if (dyn_cast<mlir::BranchOpInterface>(op)) {
    op->emitOpError(
        "branching operations not currently supported in dep-analysis");
    return false;
  }

  if (dyn_cast<mlir::CallOpInterface>(op)) {
    op->emitOpError("function calls not currently supported in dep-analysis");
    return false;
  }

  return true;
}

/// Validates that \p func meets the assumptions:
/// * function bodies contain a single block
/// * functions have no arguments
/// * functions have no results
[[maybe_unused]] bool validateFunc(func::FuncOp func) {
  if (func.getBlocks().size() != 1) {
    func.emitOpError(
        "multiple blocks not currently supported in dep-analysis");
    return false;
  }

  if (func.getArguments().size() != 0) {
    func.emitOpError(
        "function arguments not currently supported in dep-analysis");
    return false;
  }

  if (func.getNumResults() != 0) {
    func.emitOpError(
        "non-void return types not currently supported in dep-analysis");
    return false;
  }
  return true;
}

class DependencyAnalysisEngine {
private:
  SmallVector<DependencyNode *> perOp;
  DenseMap<BlockArgument, ArgDependencyNode *> argMap;

public:
  DependencyAnalysisEngine() : perOp({}), argMap({}) {}

  DependencyBlock *visitBlock(mlir::Block *b, SmallVector<DependencyNode *> dependencies) {
    SmallVector<ArgDependencyNode *> argdnodes;
    for (auto targ : b->getArguments()) {
      auto dnode = new ArgDependencyNode(targ, dependencies[targ.getArgNumber()]);
      argMap[targ] = dnode;
      argdnodes.push_back(dnode);
    }

    SetVector<DependencyNode *> roots;
    DependencyNode *terminator;
    for (auto &op : b->getOperations()) {
      bool isTerminator = (&op == b->getTerminator());
      auto node = visitOp(&op, isTerminator);

      if (!node)
        return nullptr;

      if (isEndOp(&op))
        roots.insert(node);

      if (isTerminator) {
        roots.insert(node);
        terminator = node;
      } 
    }

    SmallVector<DependencyGraph *> graphs;
    while (!roots.empty()) {
      DependencyGraph *new_graph = new DependencyGraph(roots.front());
      roots.set_subtract(new_graph->getRoots());
      graphs.push_back(new_graph);
    }

    return new DependencyBlock(argdnodes, graphs, b, terminator);
  }

  /// Creates and returns a new dependency node for \p op, connecting it to the
  /// nodes created for the defining operations of the operands of \p op
  DependencyNode *visitOp(Operation *op, bool isTerminator) {
    if (!validateOp(op))
      return nullptr;

    SmallVector<DependencyNode *> dependencies(op->getNumOperands());
    for (uint i = 0; i < op->getNumOperands(); i++) {
      auto dependency = visitValue(op->getOperand(i));
      assert(dependency && "dependency node not found for dependency");
      dependencies[i] = dependency;
    }

    DependencyNode *newNode;

    if (isTerminator)
      newNode = new TerminatorDependencyNode(op, dependencies);
    else if (auto init = dyn_cast<quake::NullWireOp>(op))
      newNode = new InitDependencyNode(init);
    else if (auto sink = dyn_cast<quake::SinkOp>(op))
      newNode = new RootDependencyNode(sink, dependencies);
    else if (auto ifop = dyn_cast<cudaq::cc::IfOp>(op)) {
      auto then_block = visitBlock(ifop.getThenEntryBlock(), dependencies);
      auto else_block = visitBlock(ifop.getElseEntryBlock(), dependencies);
      if (!then_block || !else_block)
        return nullptr;
      newNode = new IfDependencyNode(ifop, dependencies, then_block, else_block);
    }
    else
      newNode = new OpDependencyNode(op, dependencies);

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

    if (auto barg = dyn_cast<BlockArgument>(v))
      return argMap[barg];

    // Return null so the error can be handled nicely by visitOp
    return nullptr;
  }
};

struct DependencyAnalysisPass
    : public cudaq::opt::impl::DependencyAnalysisBase<DependencyAnalysisPass> {
  using DependencyAnalysisBase::DependencyAnalysisBase;
  void runOnOperation() override {
    auto mod = getOperation();
    if (mod.getBody()->getOperations().size() != 1) {
      mod.emitOpError("multi-function modules not currently supported in dep-analysis");
      signalPassFailure();
      return;
    }

    if (auto func = dyn_cast<func::FuncOp>(mod.front())) {
      validateFunc(func);
      Block *oldBlock = &func.front();

      auto engine = DependencyAnalysisEngine();

      auto body = engine.visitBlock(oldBlock, SmallVector<DependencyNode *>(oldBlock->getNumArguments(), nullptr));

      if (!body) {
        signalPassFailure();
        return;
      }

      OpBuilder builder(func);
      auto name = "wires";
      LifeTimeSet set(name);
      body->codeGen(builder, &func.getRegion(), set);
      builder.setInsertionPointToStart(mod.getBody());
      builder.create<quake::WireSetOp>(builder.getUnknownLoc(), name, set.getCount(), ElementsAttr{});

      // Replace old block
      oldBlock->erase();
    } else {
      mod.front().emitOpError("expected func");
      signalPassFailure();
      return;
    }
  }
};

} // namespace
