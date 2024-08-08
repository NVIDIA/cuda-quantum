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
typedef size_t PhysicalQID;
typedef size_t VirtualQID;

[[maybe_unused]] bool isMeasureOp(Operation *op) {
  return isa<quake::MxOp>(*op) || isa<quake::MyOp>(*op) ||
         isa<quake::MzOp>(*op);
}

[[maybe_unused]] bool isBeginOp(Operation *op) {
  return isa<quake::NullWireOp>(*op);
}

[[maybe_unused]] bool isEndOp(Operation *op) { return isa<quake::SinkOp>(*op); }

[[maybe_unused]] size_t getOperandIDXFromResultIDX(size_t resultidx,
                                                   Operation *op) {
  if (isMeasureOp(op))
    return 0;
  if (isa<quake::SwapOp>(op))
    return (resultidx == 0 ? 1 : 0);
  // Currently, all classical operands precede all quantum operands
  for (auto type : op->getOperandTypes()) {
    if (!quake::isQuantumType(type))
      resultidx++;
    else
      break;
  }
  return resultidx;
}

[[maybe_unused]] size_t getResultIDXFromOperandIDX(size_t operand_idx,
                                                   Operation *op) {
  if (isMeasureOp(op))
    return 1;
  if (isa<quake::SwapOp>(op))
    return (operand_idx == 0 ? 1 : 0);
  // Currently, all classical operands precede all quantum operands
  for (auto type : op->getOperandTypes()) {
    if (!quake::isQuantumType(type))
      operand_idx--;
    else
      break;
  }
  return operand_idx;
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
  StringRef name;
  SmallVector<SmallVector<LifeTime *>> lifetimes;
  SmallVector<SetVector<PhysicalQID>> frames;
  // DenseMap<size_t,LifeTime *> perQID;
  DenseMap<VirtualQID, PhysicalQID> virToPhys;
  size_t width;

  bool isInCurrentFrame(PhysicalQID pqid) {
    return frames.back().contains(pqid);
  }

  void combineOrAdd(PhysicalQID pqid, LifeTime *lifetime) {
    if (isInCurrentFrame(pqid))
      lifetimes[pqid].back()->combine(lifetime);
    else
      lifetimes[pqid].push_back(lifetime);
  }

  /// Given a set of qubit lifetimes and a candidate lifetime,
  /// tries to find a qubit to reuse, otherwise allocates a new qubit
  PhysicalQID allocatePhysical(LifeTime *lifetime) {
    std::optional<PhysicalQID> best_reuse = std::nullopt;
    std::optional<PhysicalQID> empty = std::nullopt;
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

    PhysicalQID physical = INT_MAX;

    // Reuse a qubit based on its lifetime in the same scope
    if (best_reuse.has_value()) {
      physical = best_reuse.value();
    } else if (empty.has_value()) {
      // Reuse a qubit based on its lifetime in a different scope
      physical = empty.value();
    } else {
      // Fall back: allocate a new qubit
      physical = lifetimes.size();
      lifetimes.push_back({});
      if (lifetimes.size() > width)
        width = lifetimes.size();
    }

    combineOrAdd(physical, lifetime);
    return physical;
  }

public:
  LifeTimeSet(StringRef name) : name(name), lifetimes(), frames(), width(0) {}

  PhysicalQID mapToPhysical(VirtualQID qid, LifeTime *lifetime) {
    if (virToPhys.count(qid) == 1)
      return virToPhys[qid];

    auto phys = allocatePhysical(lifetime);
    frames.back().insert(phys);
    virToPhys[qid] = phys;
    return phys;
  }

  // quake::BorrowWireOp genBorrow(VirtualQID qid, OpBuilder &builder) {
  //   // auto phys = mapToPhysical(lifetime);
  //   // frames.back().insert(phys);

  //   auto wirety = quake::WireType::get(builder.getContext());
  //   return builder.create<quake::BorrowWireOp>(builder.getUnknownLoc(),
  //   wirety, set, virToPhys[qid]);
  // }

  void pushFrame() { frames.push_back({}); }

  SetVector<PhysicalQID> popFrame() {
    auto pqids = frames.back();
    frames.pop_back();
    for (auto pqid : pqids) {
      lifetimes[pqid].pop_back();
    }
    return pqids;
  }

  void addOpaque(SetVector<PhysicalQID> pqids, LifeTime *lifetime) {
    for (auto pqid : pqids) {
      combineOrAdd(pqid, lifetime);
      frames.back().insert(pqid);
    }
  }

  size_t getCount() { return width; }

  void print() {
    llvm::outs() << "# qubits: " << width << ", # frames: " << frames.size()
                 << ", cycles: ";
    for (size_t i = 0; i < lifetimes.size(); i++)
      if (lifetimes[i].empty())
        llvm::outs() << "E ";
      else
        llvm::outs() << lifetimes[i].back()->getEnd() << " ";
    llvm::outs() << "\n";
  }

  StringRef getName() { return name; }
};

class DependencyNode {
  friend class DependencyGraph;
  friend class OpDependencyNode;
  friend class IfDependencyNode;
  friend class ArgDependencyNode;
  friend class RootDependencyNode;

public:
  struct DependencyEdge {
  public:
    DependencyNode *node;
    // If a given dependency appears multiple times,
    // (e.g., multiple results of the dependency are used by this node),
    // it is important to know which result from the dependency
    // corresponds to which operand.
    // Otherwise, the dependency will be code gen'ed first, and it will
    // be impossible to know (e.g., which result is a control and which is a
    // target). Resultidx tracks this information.
    size_t resultidx;
    std::optional<VirtualQID> qid;

    DependencyEdge() : node(nullptr), resultidx(INT_MAX), qid(std::nullopt) {}

    DependencyEdge(DependencyNode *node, size_t resultidx)
        : node(node), resultidx(resultidx) {
      assert(node && "DependencyEdge: node cannot be null");
      qid = node->getQIDForResult(resultidx);
    }

    /// Returns the underlying DependencyNode * without attached metadata
    DependencyNode *operator->() { return node; }

    /// Returns the value represented by this DependencyEdge
    Value getValue() { return node->getResult(resultidx); }
  };

protected:
  SetVector<DependencyNode *> successors;
  // Dependencies are in the order of operands
  SmallVector<DependencyEdge> dependencies;
  SetVector<VirtualQID> qids;
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
  virtual bool isQuantumOp() = 0;
  virtual bool isContainer() { return false; }
  virtual uint numTicks() = 0;
  virtual Value getResult(uint resultidx) = 0;
  virtual ValueRange getResults() = 0;
  virtual SetVector<PhysicalQID> mapToPhysical(LifeTimeSet &set) { return {}; }
  virtual void codeGen(OpBuilder &builder, LifeTimeSet &set) = 0;

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

  /// Assigns cycles to quantum operations. A node must be scheduled after all
  /// of its dependencies, and before all of its successors. A node cannot be
  /// scheduled at a negative cycle, nor can it be scheduled at a cycle greater
  /// than or equal to the height of the graph to which it belongs.
  ///
  /// The scheduling algorithm (as currently implemented) works by always
  /// following the longest path first.
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
    SmallVector<DependencyEdge> sorted(dependencies);
    std::sort(sorted.begin(), sorted.end(),
              [](DependencyEdge x, DependencyEdge y) {
                return x.node->getHeight() > y.node->getHeight();
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

  virtual std::optional<std::string> getOpName() = 0;

  virtual bool equivalentTo(DependencyNode *other) {
    if (!getOpName() || !other->getOpName())
      return false;
    if (getOpName().value() != other->getOpName().value())
      return false;
    if (height != other->height)
      return false;
    if (dependencies.size() != other->dependencies.size())
      return false;
    for (uint i = 0; i < dependencies.size(); i++) {
      if (!dependencies[i].node->equivalentTo(other->dependencies[i].node))
        return false;
    }
    return true;
  }

  virtual std::optional<VirtualQID> getQIDForResult(size_t resultidx) = 0;

  virtual void performLifting() {}

  virtual void computeHeight() {
    height = 0;
    for (auto edge : dependencies) {
      if (edge->getHeight() > height)
        height = edge->getHeight();
    }
    height += numTicks();
  }

public:
  DependencyNode()
      : successors(), dependencies({}), qids({}), height(0),
        isScheduled(false) {}

  uint getHeight() { return height; };

  void print() { printSubGraph(0); }

  virtual bool isQuantumDependent() { return qids.size() > 0; };

  virtual void performAnalysis(LifeTimeSet &set) {
    assert(false && "performAnalysis can only be called on an IfDependencyNode");
  }
};

class InitDependencyNode : public DependencyNode {
  friend class DependencyGraph;

protected:
  Value wire;
  PhysicalQID pqid = INT_MAX;

  void printNode() override {
    llvm::outs() << "Initial value for QID " << getQID();
    if (pqid != INT_MAX)
      llvm::outs() << "=" << pqid;
    llvm::outs() << ": ";
    wire.dump();
  }

  bool isAlloc() override { return true; }
  uint numTicks() override { return 0; }
  bool isQuantumOp() override { return true; }

  Value getResult(uint resultidx) override {
    assert(resultidx == 0 && "Illegal resultidx");
    return wire;
  }

  ValueRange getResults() override { return ValueRange({wire}); }

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {
    auto wirety = quake::WireType::get(builder.getContext());
    auto alloc = builder.create<quake::BorrowWireOp>(
        builder.getUnknownLoc(), wirety, set.getName(), pqid);
    wire = alloc.getResult();
    hasCodeGen = true;
  }

  void performMapping(PhysicalQID phys) { pqid = phys; }

  VirtualQID getQID() { return qids.front(); }

  virtual std::optional<std::string> getOpName() override {
    return std::optional("init");
  };

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    assert(resultidx == 0 && "Invalid resultidx");
    return std::optional(getQID());
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

class OpDependencyNode : public DependencyNode {
  friend class IfDependencyNode;

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

  ValueRange getResults() override { return associated->getResults(); }

  SmallVector<mlir::Value> gatherOperands(size_t num, OpBuilder &builder,
                                          LifeTimeSet &set) {
    SmallVector<mlir::Value> operands(num);
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto dependency = dependencies[i];

      // Ensure classical values are available and that any allocs are added
      if (dependency->isSkip())
        dependency->codeGen(builder, set);

      assert(dependency->hasCodeGen &&
             "Generating code for successor before dependency");

      // Get relevant result from dependency's updated op
      // to use as the relevant operand
      operands[i] = dependency->getResult(dependency.resultidx);
    }

    return operands;
  }

  /// Generates a new operation for this node in the dependency graph
  /// using the dependencies of the node as operands.
  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {
    if (hasCodeGen)
      return;

    // Non-quake operations have code generated aggressively
    // This ensures that code gen is not too aggressive
    if (isSkip())
      for (auto dependency : dependencies)
        if (!dependency->hasCodeGen)
          return;

    auto oldOp = associated;
    auto operands = gatherOperands(oldOp->getNumOperands(), builder, set);

    associated =
        Operation::create(oldOp->getLoc(), oldOp->getName(),
                          oldOp->getResultTypes(), operands, oldOp->getAttrs());
    associated->removeAttr("dnodeid");
    builder.insert(associated);
    hasCodeGen = true;

    // Ensure classical values are generated
    for (auto successor : successors)
      if (successor->isSkip())
        successor->codeGen(builder, set);
  }

  std::optional<std::string> getOpName() override {
    if (isa<arith::ConstantOp>(associated)) {
      if (auto cstf = dyn_cast<arith::ConstantFloatOp>(associated)) {
        auto value = cstf.getValue().cast<FloatAttr>().getValueAsDouble();
        return std::optional(std::to_string(value));
      } else if (auto csti = dyn_cast<arith::ConstantIndexOp>(associated)) {
        auto value = cstf.getValue().cast<IntegerAttr>().getInt();
        return std::optional(std::to_string(value));
      } else if (auto csti = dyn_cast<arith::ConstantIntOp>(associated)) {
        auto value = cstf.getValue().cast<IntegerAttr>().getInt();
        return std::optional(std::to_string(value));
      }
    }
    return std::optional(associated->getName().getStringRef().str());
  };

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    if (!isQuantumOp())
      return std::nullopt;
    return dependencies[getOperandIDXFromResultIDX(resultidx, associated)].qid;
  }

public:
  OpDependencyNode(Operation *op, SmallVector<DependencyEdge> _dependencies)
      : associated(op) {
    assert(op && "Cannot make dependency node for null op");
    assert(_dependencies.size() == op->getNumOperands() &&
           "Wrong # of dependencies to construct node");

    dependencies = _dependencies;

    quantumOp = isQuakeOperation(op);
    if (dyn_cast<quake::DiscriminateOp>(op))
      quantumOp = false;

    height = 0;
    // Ingest dependencies, setting up metadata
    for (size_t i = 0; i < dependencies.size(); i++) {
      auto edge = dependencies[i];

      assert(edge->getResult(edge.resultidx) == associated->getOperand(i) &&
             "Dependency isn't actually a dependency!");
      // Add this as a successor to each dependency
      edge->successors.insert(this);

      // Update metadata
      if (!isEndOp(op) && edge.qid.has_value())
        qids.insert(edge.qid.value());
    }

    computeHeight();
  };

  void print() { printSubGraph(0); }

  uint getHeight() { return height; }

  DependencyEdge getDependencyForResult(size_t resultidx) {
    return dependencies[getOperandIDXFromResultIDX(resultidx, associated)];
  }

  size_t getResultForDependency(size_t operandidx) {
    return getResultIDXFromOperandIDX(operandidx, associated);
  }

  /// Remove this dependency node by replacing successor dependencies with
  /// the relevant dependency from this node.
  void erase() {
    for (auto successor : successors) {
      for (uint j = 0; j < successor->dependencies.size(); j++) {
        auto edge = successor->dependencies[j];
        if (edge.node == this) {
          successor->dependencies[j] = getDependencyForResult(edge.resultidx);
          getDependencyForResult(edge.resultidx)->successors.insert(successor);
        }
      }
    }
  }
};

class DependencyGraph {
private:
  SetVector<DependencyNode *> roots;
  DenseMap<VirtualQID, InitDependencyNode *> allocs;
  DenseMap<VirtualQID, DependencyNode *> leafs;
  SetVector<VirtualQID> qids;
  uint total_height;
  bool isScheduled = false;
  DependencyNode *tallest = nullptr;
  uint shift;
  SetVector<DependencyNode *> containers;

  /// Starting from \p next, searches through \p next's family
  /// (excluding already seen nodes) to find all the interconnected roots
  /// that this graph represents.
  /// Also fills in metadata about the height of the graph, and the qids in the
  /// graph.
  void gatherRoots(SetVector<DependencyNode *> &seen, DependencyNode *next) {
    if (seen.contains(next) || !next->isQuantumDependent())
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
      leafs.insert({next->qids.front(), next});

    if (next->isAlloc()) {
      auto init = static_cast<InitDependencyNode *>(next);
      assert(init &&
             "DependencyNode claiming to be alloc is not actually an alloc");
      allocs[init->getQID()] = init;
    }

    if (next->isContainer())
      containers.insert(next);

    for (auto successor : next->successors)
      gatherRoots(seen, successor);
    for (auto dependency : next->dependencies)
      gatherRoots(seen, dependency.node);
  }

  SetVector<DependencyNode *> getNodesAtCycle(uint cycle) {
    SetVector<DependencyNode *> nodes;
    for (auto root : roots)
      nodes.set_union(root->getNodesAtCycle(cycle));
    return nodes;
  }

  /*void performLifting(SetVector<DependencyNode *> &seen, DependencyNode *next) {
    if (seen.contains(next))
      return;

    seen.insert(next);

    if (!next->isQuantumDependent())
      return;

    next->performLifting();

    for (auto edge : next->dependencies)
      performLifting(seen, edge.node);
  }*/

public:
  DependencyGraph(DependencyNode *root) {
    shift = 0;
    total_height = 0;
    SetVector<DependencyNode *> seen;
    gatherRoots(seen, root);
    if (roots.size() == 0)
      return;

    qids = SetVector<size_t>();
    for (auto root : roots) {
      qids.set_union(root->qids);
    }
  }

  SetVector<DependencyNode *> &getRoots() { return roots; }

  SetVector<VirtualQID> &getQIDs() { return qids; }

  size_t getNumQIDs() { return qids.size(); }

  LifeTime *getLifeTimeForQID(VirtualQID qid) {
    uint first = getFirstUseOf(qid)->cycle + shift;
    auto last = getLastUseOf(qid)->cycle + shift;

    return new LifeTime(first, last);
  }

  OpDependencyNode *getFirstUseOf(VirtualQID qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    return static_cast<OpDependencyNode *>(leafs[qid]->successors[0]);
  }

  OpDependencyNode *getLastUseOf(VirtualQID qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    return static_cast<OpDependencyNode *>(
        getRootForQID(qid)->dependencies[0].node);
  }

  DependencyNode *getRootForQID(VirtualQID qid) {
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

  SetVector<PhysicalQID> mapToPhysicalAt(uint cycle, LifeTimeSet &set) {
    SetVector<DependencyNode *> nodes = getNodesAtCycle(cycle);

    SetVector<PhysicalQID> pqids;

    for (auto node : nodes)
      pqids.set_union(node->mapToPhysical(set));

    return pqids;
  }

  uint getHeight() { return total_height; }

  SmallVector<VirtualQID> getFirstUsedAtCycle(uint cycle) {
    SmallVector<VirtualQID> fresh;
    for (auto [qid, _] : allocs)
      if (getFirstUseOf(qid)->cycle == cycle)
        fresh.push_back(qid);

    return fresh;
  }

  SmallVector<VirtualQID> getLastUsedAtCycle(uint cycle) {
    SmallVector<VirtualQID> stale;
    for (auto [qid, _] : allocs)
      if (getLastUseOf(qid)->cycle == cycle)
        stale.push_back(qid);

    return stale;
  }

  SetVector<VirtualQID> getAllocs() {
    SetVector<VirtualQID> allocated;
    for (auto [qid, _] : allocs)
      allocated.insert(qid);
    return allocated;
  }

  void performMapping(VirtualQID qid, PhysicalQID phys) {
    if (allocs.count(qid) == 1)
      allocs[qid]->performMapping(phys);
  }

  /*void performLifting() {
    SetVector<DependencyNode *> seen;
    for (auto root : roots)
      performLifting(seen, root);
  }*/
  
  void performScheduling() { tallest->schedule(total_height); }

  void print() {
    llvm::outs() << "Graph Start\n";
    for (auto root : roots)
      root->print();
    llvm::outs() << "Graph End\n";
  }

  void setCycleOffset(uint cycle) { this->shift = cycle; }

  void performAnalysis(LifeTimeSet &set) {
    for (auto container : containers)
      container->performAnalysis(set);
  }
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

  bool isSkip() override { return true; }

  uint numTicks() override { return 0; }

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {
    assert(!hasCodeGen && "Returning same wire twice");
    auto wire = dependencies[0].getValue();
    auto newOp =
        builder.create<quake::ReturnWireOp>(builder.getUnknownLoc(), wire);
    newOp->setAttrs(associated->getAttrs());
    newOp->removeAttr("dnodeid");
    associated = newOp;
    hasCodeGen = true;
  }

public:
  RootDependencyNode(quake::SinkOp op, SmallVector<DependencyEdge> dependencies)
      : OpDependencyNode(op, dependencies) {
    // Should be ensured by assign-ids pass
    assert(op->hasAttr("qid") && "quake.sink missing qid");
    // It's useful to know precisely which VirtualQID this sinkOp is for
    qids.clear();
    // Lookup qid
    auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
    qids.insert(qid);

    // numTicks won't be properly calculated by OpDependencyNode constructor,
    // so have to recompute height here
    computeHeight();
  };
};

class ArgDependencyNode : public DependencyNode {
  friend class DependencyBlock;
  friend class IfDependencyNode;

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

  ValueRange getResults() override { return ValueRange({barg}); }

  virtual std::optional<std::string> getOpName() override {
    return std::optional(std::to_string(barg.getArgNumber()).append("arg"));
  };

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override{};

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    assert(resultidx == 0 && "Invalid resultidx");
    return std::optional(qids.front());
  }

public:
  ArgDependencyNode(BlockArgument arg, DependencyEdge val) : barg(arg) {
    auto qid = val->getQIDForResult(val.resultidx);
    if (qid.has_value())
      qids.insert(qid.value());
  }

  ArgDependencyNode(BlockArgument arg) : barg(arg) {}
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

  bool isQuantumOp() override { return !qids.empty(); }

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override{};

public:
  TerminatorDependencyNode(Operation *terminator,
                           SmallVector<DependencyEdge> dependencies)
      : OpDependencyNode(terminator, dependencies) {
    assert(terminator->hasTrait<mlir::OpTrait::ReturnLike>() &&
           "Invalid terminator");
  }

  void genTerminator(OpBuilder &builder, LifeTimeSet &set) {
    OpDependencyNode::codeGen(builder, set);
  }
};

class DependencyBlock {
private:
  SmallVector<ArgDependencyNode *> argdnodes;
  SmallVector<DependencyGraph *> graphs;
  DenseMap<VirtualQID, DependencyGraph *> graphMap;
  Block *block;
  TerminatorDependencyNode *terminator;
  uint height;
  SetVector<size_t> pqids;

public:
  DependencyBlock(SmallVector<ArgDependencyNode *> argdnodes,
                  SmallVector<DependencyGraph *> graphs, Block *block,
                  TerminatorDependencyNode *terminator)
      : argdnodes(argdnodes), graphs(graphs), block(block),
        terminator(terminator), pqids() {
    // Each Graph should contain distinct VirtualQIDs
    for (auto graph : graphs)
      for (auto qid : graph->getQIDs())
        graphMap[qid] = graph;

    computeHeight();
  }

  uint getHeight() { return height; }

  void setCycle(uint cycle) {
    for (auto graph : graphs)
      graph->setCycleOffset(cycle);
  }

  SetVector<VirtualQID> getAllocs() {
    SetVector<VirtualQID> allocated;
    for (auto graph : graphs)
      allocated.set_union(graph->getAllocs());

    return allocated;
  }

  SetVector<VirtualQID> getQIDs() {
    SetVector<VirtualQID> qids;
    for (auto graph : graphs)
      qids.set_union(graph->getQIDs());

    return qids;
  }

  OpDependencyNode *getFirstUseOf(VirtualQID qid) {
    return graphMap[qid]->getFirstUseOf(qid);
  }

  SetVector<PhysicalQID> mapToPhysical(LifeTimeSet &set) {
    set.pushFrame();

    for (uint cycle = 0; cycle < height; cycle++) {
      for (auto graph : graphs) {
        for (auto qid : graph->getFirstUsedAtCycle(cycle)) {
          auto lifetime = graph->getLifeTimeForQID(qid);
          LLVM_DEBUG(llvm::dbgs() << "Qid " << qid);
          LLVM_DEBUG(llvm::dbgs()
                     << " is in use from cycle " << lifetime->getBegin());
          LLVM_DEBUG(llvm::dbgs() << " through cycle " << lifetime->getEnd());
          LLVM_DEBUG(llvm::dbgs() << "\n");

          auto phys = set.mapToPhysical(qid, lifetime);
          LLVM_DEBUG(llvm::dbgs()
                     << "\tIt is mapped to the physical qubit " << phys);
          LLVM_DEBUG(llvm::dbgs() << "\n\n");

          graph->performMapping(qid, phys);
        }

        // New physical qubits will be captured in the LifeTimeSet frame,
        // no need to capture here
        graph->mapToPhysicalAt(cycle, set);
      }
    }

    return set.popFrame();
  }

  /// Up to caller to move builder outside block after construction
  Block *codeGen(OpBuilder &builder, Region *region, LifeTimeSet &set) {
    // graphs[0]->print();
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

    for (uint cycle = 0; cycle < height; cycle++)
      for (auto graph : graphs)
        graph->codeGenAt(cycle, builder, set);

    terminator->genTerminator(builder, set);

    block = newBlock;

    return newBlock;
  }

  void print() {
    llvm::outs() << "Block:\n";
    block->dump();
    llvm::outs() << "Block graphs:\n";
    for (auto graph : graphs)
      graph->print();
    llvm::outs() << "End block\n";
  }

  void computeHeight() {
    height = 0;
    for (auto graph : graphs) {
      if (graph->getHeight() > height)
        height = graph->getHeight();
    }
  }

  void performAnalysis(LifeTimeSet &set) {
    for (auto graph : graphs)
      graph->performAnalysis(set);
  }

  void performScheduling() {
    for (auto graph : graphs)
      graph->performScheduling();
  }
};

class IfDependencyNode : public OpDependencyNode {
  friend class ArgDependencyNode;

protected:
  DependencyBlock *then_block;
  DependencyBlock *else_block;

  // TODO: figure out nice way to display
  void printNode() override {
    this->OpDependencyNode::printNode();
    llvm::outs() << "Then ";
    then_block->print();
    llvm::outs() << "Else ";
    else_block->print();
  }

  uint numTicks() override {
    return std::max(then_block->getHeight(), else_block->getHeight());
  }

  bool isQuantumOp() override {
    for (auto type : associated->getResultTypes())
      if (quake::isQuantumType(type))
        return true;

    return numTicks() > 0;
  }

  virtual bool isContainer() override { return true; }

  void liftOp(OpDependencyNode *op) {
    auto newDeps = SmallVector<DependencyEdge>();

    for (uint i = 0; i < op->dependencies.size(); i++) {
      auto dependency = op->dependencies[i];
      assert(!dependency->isAlloc() && "TODO");

      if (!dependency->isQuantumOp()) {
        newDeps.push_back(dependency);
      } else if (dependency->isLeaf()) {
        ArgDependencyNode *arg =
            static_cast<ArgDependencyNode *>(dependency.node);
        auto num = arg->barg.getArgNumber();
        auto newDep = dependencies[num + 1];
        newDeps.push_back(newDep);
        newDep->successors.remove(this);
        newDep->successors.insert(op);

        dependencies[num + 1] =
            DependencyEdge{op, op->getResultForDependency(i)};
        // Patch successors
        for (auto successor : op->successors) {
          for (uint j = 0; j < successor->dependencies.size(); j++) {
            auto edge = successor->dependencies[j];
            if (edge.node == op)
              successor->dependencies[j] =
                  op->getDependencyForResult(edge.resultidx);
          }
          op->successors.remove(successor);
          successors.insert(successor);
        }
      }
    }

    op->successors.insert(this);
    op->dependencies = newDeps;
  }

  void performLifting() override {
    // Calculate qids from outside used inside
    auto then_allocs = then_block->getAllocs();
    auto else_allocs = else_block->getAllocs();
    auto inner_qids = then_block->getQIDs();
    inner_qids.set_union(else_block->getQIDs());
    inner_qids.set_subtract(then_allocs);
    inner_qids.set_subtract(else_allocs);

    DenseMap<VirtualQID, DependencyEdge> depMap;
    for (auto edge : dependencies)
      if (edge.qid.has_value())
        depMap[edge.qid.value()] = edge;

    for (auto qid : inner_qids) {
      auto then_use = then_block->getFirstUseOf(qid);
      auto else_use = else_block->getFirstUseOf(qid);

      if (!then_use || !else_use)
        continue;

      if (then_use->equivalentTo(else_use)) {
        liftOp(then_use);
        else_use->erase();
      }
    }

    // Alloc case todo
    // for (auto then_alloc : then_allocs) {
    //   llvm::outs() << "QID: " << then_alloc << "\n";
    //   auto then_use = then_block->getFirstUseOf(then_alloc);
    //   // if (then_use->cycle > 0)
    //   //   continue;
    //   then_use->printNode();
    //   for (auto else_alloc : else_allocs) {
    //     auto else_use = else_block->getFirstUseOf(else_alloc);
    //     else_use->printNode();
    //     if (then_use->equivalentTo(else_use))
    //       llvm::outs() << "The operation on alloc " << then_alloc << "/" <<
    //       else_alloc << " can be lifted!\n";
    //   }
    // }
  }

  SetVector<PhysicalQID> mapToPhysical(LifeTimeSet &set) override {
    then_block->setCycle(cycle);
    else_block->setCycle(cycle);
    auto pqids1 = then_block->mapToPhysical(set);
    auto pqids2 = else_block->mapToPhysical(set);
    // TODO: function for combining pqids
    pqids1.set_union(pqids2);
    set.addOpaque(pqids1, new LifeTime(cycle, cycle + numTicks()));
    return pqids1;
  };

  void codeGen(OpBuilder &builder, LifeTimeSet &set) override {
    if (hasCodeGen)
      return;

    if (isSkip())
      for (auto dependency : dependencies)
        if (!dependency->hasCodeGen)
          return;

    cudaq::cc::IfOp oldOp = dyn_cast<cudaq::cc::IfOp>(associated);
    auto operands = gatherOperands(oldOp->getNumOperands(), builder, set);

    auto newIf = builder.create<cudaq::cc::IfOp>(
        oldOp->getLoc(), oldOp->getResultTypes(), operands);
    auto *then_region = &newIf.getThenRegion();
    then_block->codeGen(builder, then_region, set);

    auto *else_region = &newIf.getElseRegion();
    else_block->codeGen(builder, else_region, set);

    associated = newIf;
    builder.setInsertionPointAfter(associated);
    hasCodeGen = true;

    // Ensure classical values are generated
    for (auto successor : successors) {
      if (!successor->isSkip())
        successor->codeGen(builder, set);
    }
  };

  void computeHeight() override {
    height = 0;
    for (auto edge : dependencies)
      if (edge->getHeight() > height)
        height = edge->getHeight();
    height += numTicks();
  }

public:
  IfDependencyNode(cudaq::cc::IfOp op, SmallVector<DependencyEdge> dependencies,
                   DependencyBlock *then_block, DependencyBlock *else_block)
      : OpDependencyNode(op.getOperation(), dependencies),
        then_block(then_block), else_block(else_block) {
    // numTicks won't be properly calculated by OpDependencyNode constructor,
    // so have to recompute height here
    computeHeight();
  }

  void performAnalysis(LifeTimeSet &set) override {
    // First, recur to settle Ifs inside blocks
    then_block->performAnalysis(set);
    else_block->performAnalysis(set);
    // Lift operations as possible
    performLifting();
    // Recompute block heights after lifting
    then_block->computeHeight();
    else_block->computeHeight();
    // Schedule blocks once operations inside are finalized
    then_block->performScheduling();
    else_block->performScheduling();
    // TODO: mapToPhysical - update with context
    auto newPQIDs = mapToPhysical(set);
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
    func.emitOpError("multiple blocks not currently supported in dep-analysis");
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

  DependencyBlock *
  visitBlock(mlir::Block *b,
             SmallVector<DependencyNode::DependencyEdge> dependencies) {
    SmallVector<ArgDependencyNode *> argdnodes;
    for (auto targ : b->getArguments()) {
      ArgDependencyNode *dnode;
      // Entry block has no argument dependencies
      if (dependencies.size() > 0)
        dnode =
            new ArgDependencyNode(targ, dependencies[targ.getArgNumber() + 1]);
      else
        dnode = new ArgDependencyNode(targ);
      argMap[targ] = dnode;
      argdnodes.push_back(dnode);
    }

    SetVector<DependencyNode *> roots;
    TerminatorDependencyNode *terminator;
    for (auto &op : b->getOperations()) {
      bool isTerminator = (&op == b->getTerminator());
      auto node = visitOp(&op, isTerminator);

      if (!node)
        return nullptr;

      if (isEndOp(&op))
        roots.insert(node);

      if (isTerminator) {
        assert(op.hasTrait<mlir::OpTrait::IsTerminator>() &&
               "Illegal terminator op!");
        terminator = static_cast<TerminatorDependencyNode *>(node);
        if (terminator->isQuantumDependent())
          roots.insert(terminator);
      }
    }

    SmallVector<DependencyGraph *> graphs;
    while (!roots.empty()) {
      DependencyGraph *new_graph = new DependencyGraph(roots.front());
      roots.set_subtract(new_graph->getRoots());
      if (new_graph->getRoots().size() > 0)
        graphs.push_back(new_graph);
    }

    return new DependencyBlock(argdnodes, graphs, b, terminator);
  }

  /// Creates and returns a new dependency node for \p op, connecting it to the
  /// nodes created for the defining operations of the operands of \p op
  DependencyNode *visitOp(Operation *op, bool isTerminator) {
    if (!validateOp(op))
      return nullptr;

    SmallVector<DependencyNode::DependencyEdge> dependencies;
    for (uint i = 0; i < op->getNumOperands(); i++)
      dependencies.push_back(visitValue(op->getOperand(i)));

    DependencyNode *newNode;

    if (auto init = dyn_cast<quake::NullWireOp>(op))
      newNode = new InitDependencyNode(init);
    else if (auto sink = dyn_cast<quake::SinkOp>(op))
      newNode = new RootDependencyNode(sink, dependencies);
    else if (auto ifop = dyn_cast<cudaq::cc::IfOp>(op)) {
      auto then_block = visitBlock(ifop.getThenEntryBlock(), dependencies);
      auto else_block = visitBlock(ifop.getElseEntryBlock(), dependencies);
      if (!then_block || !else_block)
        return nullptr;
      newNode =
          new IfDependencyNode(ifop, dependencies, then_block, else_block);
    } else
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
  DependencyNode::DependencyEdge visitValue(Value v) {
    if (auto barg = dyn_cast<BlockArgument>(v))
      return DependencyNode::DependencyEdge{argMap[barg], 0};

    auto defOp = v.getDefiningOp();

    auto resultidx = dyn_cast<OpResult>(v).getResultNumber();
    assert(defOp &&
           "Cannot handle value that is neither a BlockArgument nor OpResult");
    // Since we walk forward through the ast, every value should be defined
    // before it is used, so we should have already visited defOp,
    // and thus should have a memoized dnode for defOp, fail if not
    assert(defOp->hasAttr("dnodeid") && "No dnodeid found for operation");

    auto id = defOp->getAttr("dnodeid").cast<IntegerAttr>().getUInt();
    auto dnode = perOp[id];
    return DependencyNode::DependencyEdge{dnode, resultidx};
  }
};

struct DependencyAnalysisPass
    : public cudaq::opt::impl::DependencyAnalysisBase<DependencyAnalysisPass> {
  using DependencyAnalysisBase::DependencyAnalysisBase;
  void runOnOperation() override {
    auto mod = getOperation();
    if (mod.getBody()->getOperations().size() != 1) {
      mod.emitOpError(
          "multi-function modules not currently supported in dep-analysis");
      signalPassFailure();
      return;
    }

    if (auto func = dyn_cast<func::FuncOp>(mod.front())) {
      validateFunc(func);
      Block *oldBlock = &func.front();

      auto engine = DependencyAnalysisEngine();

      auto body = engine.visitBlock(
          oldBlock, SmallVector<DependencyNode::DependencyEdge>());

      if (!body) {
        signalPassFailure();
        return;
      }

      OpBuilder builder(func);
      auto name = "wires";
      LifeTimeSet set(name);
      // A function body is a container with only one inner block,
      // so we repeat the "performAnalysis" function from IfDependencyNode,
      // but with only one block instead of two
      body->performAnalysis(set);
      body->computeHeight();
      body->performScheduling();
      body->mapToPhysical(set);
      // Now, perform code generation
      body->codeGen(builder, &func.getRegion(), set);
      builder.setInsertionPointToStart(mod.getBody());
      builder.create<quake::WireSetOp>(builder.getUnknownLoc(), name,
                                       set.getCount(), ElementsAttr{});

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
