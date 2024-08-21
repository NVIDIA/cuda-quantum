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

#define RAW(X) quake::X
#define RAW_MEASURE_OPS MEASURE_OPS(RAW)
#define RAW_GATE_OPS GATE_OPS(RAW)
#define RAW_QUANTUM_OPS QUANTUM_OPS(RAW)

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

size_t getOperandIDXFromResultIDX(size_t resultidx, Operation *op) {
  if (isa<RAW_MEASURE_OPS>(op))
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

size_t getResultIDXFromOperandIDX(size_t operand_idx, Operation *op) {
  if (isa<RAW_MEASURE_OPS>(op))
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

class LifeTimeAnalysis {
private:
  StringRef name;
  SmallVector<LifeTime *> lifetimes;
  SetVector<PhysicalQID> frame;
  // DenseMap<VirtualQID, PhysicalQID> allocMap;

  /// Given a candidate lifetime, tries to find a qubit to reuse,
  /// and otherwise allocates a new qubit
  PhysicalQID allocatePhysical(LifeTime *lifetime) {
    std::optional<size_t> best_reuse = std::nullopt;
    std::optional<size_t> empty = std::nullopt;
    uint best_distance = INT_MAX;

    for (uint i = 0; i < lifetimes.size(); i++) {
      if (!lifetimes[i]) {
        empty = i;
        continue;
      }

      auto other = lifetimes[i];
      auto distance = lifetime->distance(other);
      if (!lifetime->isOverlapping(other) && distance < best_distance) {
        best_reuse = i;
        best_distance = distance;
      }
    }

    // Reuse a qubit based on its lifetime in the same scope
    if (best_reuse) {
      auto physical = best_reuse.value();
      lifetimes[physical]->combine(lifetime);
      return physical;
    }

    // Reuse a qubit without a lifetime (used in a different frame)
    if (empty) {
      auto physical = empty.value();
      lifetimes[physical] = lifetime;
      return physical;
    }

    // Fall back: allocate a new qubit
    lifetimes.push_back(lifetime);
    return lifetimes.size() - 1;
  }

public:
  LifeTimeAnalysis(StringRef name) : name(name), lifetimes(), frame() {}

  PhysicalQID allocatePhysical(VirtualQID qid, LifeTime *lifetime) {
    auto phys = allocatePhysical(lifetime);
    frame.insert(phys);
    return phys;
  }

  SetVector<PhysicalQID> getAllocated() {
    for (uint i = 0; i < lifetimes.size(); i++)
      lifetimes[i] = nullptr;
    auto pqids = SetVector<PhysicalQID>(frame);
    frame.clear();
    return pqids;
  }

  void reallocatePhysical(PhysicalQID phys, LifeTime *lifetime) {
    lifetimes[phys] = lifetime;
  }

  size_t getCount() { return lifetimes.size(); }

  void print() {
    llvm::outs() << "# qubits: " << getCount() << ", cycles: ";
    for (size_t i = 0; i < lifetimes.size(); i++)
      llvm::outs() << lifetimes[i]->getBegin() << " - "
                   << lifetimes[i]->getEnd() << " ";
    llvm::outs() << "\n";
  }

  StringRef getName() { return name; }
};

class DependencyGraph;

class DependencyNode {
  friend class DependencyGraph;
  friend class OpDependencyNode;
  friend class IfDependencyNode;
  friend class ArgDependencyNode;
  friend class RootDependencyNode;
  friend class InitDependencyNode;

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
    std::optional<PhysicalQID> qubit;

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
  std::optional<uint> cycle = std::nullopt;
  bool hasCodeGen = false;
  uint height;

  virtual void printNode() = 0;

  void printSubGraph(int tabIndex) {
    for (int i = 0; i < tabIndex; i++) {
      llvm::outs() << "\t";
    }

    printNode();

    for (auto dependency : dependencies)
      dependency->printSubGraph(tabIndex + 1);
  }

  virtual bool isRoot() { return successors.size() == 0; };
  virtual bool isLeaf() { return dependencies.size() == 0; };
  virtual bool isSkip() { return numTicks() == 0; };
  virtual bool isQuantumOp() = 0;
  virtual uint numTicks() = 0;
  virtual Value getResult(uint resultidx) = 0;
  virtual ValueRange getResults() = 0;
  virtual SetVector<PhysicalQID> mapToPhysical(LifeTimeAnalysis &set) {
    return {};
  }
  virtual void codeGen(OpBuilder &builder, LifeTimeAnalysis &set) = 0;

  /// Recursively find nodes scheduled at a given cycle
  SetVector<DependencyNode *>
  getNodesAtCycle(uint _cycle, SetVector<DependencyNode *> &seen) {
    SetVector<DependencyNode *> nodes;

    if (seen.contains(this))
      return nodes;

    seen.insert(this);

    if (!isSkip()) {
      assert(cycle.has_value() && "Trying to use cycle of unscheduled node");

      if (cycle.value() < _cycle)
        return nodes;
      else if (cycle.value() == _cycle) {
        nodes.insert(this);
        return nodes;
      }
    }

    for (auto dependency : dependencies)
      nodes.set_union(dependency->getNodesAtCycle(_cycle, seen));

    return nodes;
  }

  virtual bool prefixEquivalentTo(DependencyNode *other) {
    if (getOpName() != other->getOpName())
      return false;
    if (height != other->height)
      return false;
    if (dependencies.size() != other->dependencies.size())
      return false;
    for (uint i = 0; i < dependencies.size(); i++) {
      if (dependencies[i].qid != other->dependencies[i].qid) {
        if (!dependencies[i].qubit.has_value())
          return false;
        if (dependencies[i].qubit != other->dependencies[i].qubit)
          return false;
      }
      if (!dependencies[i].node->prefixEquivalentTo(
              other->dependencies[i].node))
        return false;
    }
    return true;
  }

  virtual bool postfixEquivalentTo(DependencyNode *other) {
    if (getOpName() != other->getOpName())
      return false;
    if (dependencies.size() != other->dependencies.size())
      return false;
    for (uint i = 0; i < dependencies.size(); i++) {
      if (dependencies[i].qid != other->dependencies[i].qid) {
        if (!dependencies[i].qubit.has_value())
          return false;
        if (dependencies[i].qubit != other->dependencies[i].qubit)
          return false;
      }
    }
    return true;
  }

  virtual void updateHeight() {
    height = 0;
    for (auto edge : dependencies) {
      if (edge->getHeight() > height)
        height = edge->getHeight();
    }
    height += numTicks();
  }

  virtual SetVector<PhysicalQID> getQubits() {
    return SetVector<PhysicalQID>();
  }

  void replaceWith(DependencyEdge other) {
    for (auto successor : successors) {
      for (uint j = 0; j < successor->dependencies.size(); j++) {
        auto edge = successor->dependencies[j];
        if (edge.node == this) {
          successor->dependencies[j] = other;
          other->successors.remove(this);
          other->successors.insert(successor);
        }
      }
    }
  }

  virtual void updateWithPhysical(VirtualQID qid, PhysicalQID qubit) {
    for (auto dependency : dependencies) {
      if (dependency.qid && dependency.qid == qid) {
        dependency.qubit = qubit;
        break;
      }
    }

    for (auto successor : successors)
      if (successor->qids.contains(qid))
        successor->updateWithPhysical(qid, qubit);
  }

  void updateQID(VirtualQID old_qid, VirtualQID new_qid) {
    qids.remove(old_qid);
    qids.insert(new_qid);
    for (auto dependency : dependencies) {
      if (dependency.qid && dependency.qid == old_qid) {
        dependency.qid = new_qid;
        break;
      }
    }

    for (auto successor : successors)
      if (successor->qids.contains(old_qid))
        successor->updateQID(old_qid, new_qid);
  }

public:
  DependencyNode() : successors(), dependencies({}), qids({}), height(0) {}

  virtual bool isAlloc() { return false; }

  uint getHeight() { return height; };

  void print() { printSubGraph(0); }

  virtual bool isQuantumDependent() {
    if (isQuantumOp())
      return true;
    for (auto dependency : dependencies)
      if (dependency->isQuantumDependent())
        return true;
    return false;
  };

  virtual void contractAllocsPass() {
    assert(false &&
           "contractAllocPass can only be called on an IfDependencyNode");
  }

  virtual void performAnalysis(LifeTimeAnalysis &set,
                               DependencyGraph *parent_graph) {
    assert(false &&
           "performAnalysis can only be called on an IfDependencyNode");
  }

  virtual void lowerAlloc(DependencyNode *init, DependencyNode *root,
                          VirtualQID alloc) {
    assert(false && "lowerAlloc can only be called on an IfDependencyNode");
  }

  virtual void liftAlloc(DependencyNode *init, DependencyNode *root,
                         VirtualQID alloc) {
    assert(false && "liftAlloc can only be called on an IfDependencyNode");
  }

  virtual std::string getOpName() = 0;

  virtual bool isContainer() { return false; }

  /// Remove this dependency node from the path for \p qid by replacing
  /// successor dependencies on \p qid with the relevant dependency from this
  /// node.
  virtual void eraseQID(VirtualQID qid) = 0;

  virtual std::optional<VirtualQID> getQIDForResult(size_t resultidx) = 0;
};

class InitDependencyNode : public DependencyNode {
  friend class DependencyGraph;

protected:
  Value wire;
  std::optional<PhysicalQID> qubit = std::nullopt;

  void printNode() override {
    llvm::outs() << "Initial value for QID " << getQID();
    if (qubit)
      llvm::outs() << " -> phys: " << qubit.value();
    llvm::outs() << ": ";
    wire.dump();
  }

  uint numTicks() override { return 0; }
  bool isQuantumOp() override { return true; }

  Value getResult(uint resultidx) override {
    assert(resultidx == 0 && "Illegal resultidx");
    return wire;
  }

  ValueRange getResults() override { return ValueRange({wire}); }

  void codeGen(OpBuilder &builder, LifeTimeAnalysis &set) override {
    assert(qubit.has_value() && "Trying to codeGen a virtual allocation "
                                "without a physical qubit assigned!");
    auto wirety = quake::WireType::get(builder.getContext());
    auto alloc = builder.create<quake::BorrowWireOp>(
        builder.getUnknownLoc(), wirety, set.getName(), qubit.value());
    wire = alloc.getResult();
    hasCodeGen = true;
  }

  void assignToPhysical(PhysicalQID phys) {
    qubit = phys;
    updateWithPhysical(getQID(), phys);
  }

  VirtualQID getQID() { return qids.front(); }

public:
  InitDependencyNode(quake::BorrowWireOp op) : wire(op.getResult()) {
    // Should be ensured by assign-ids pass

    // Lookup qid
    auto qid = op.getIdentity();
    qids.insert(qid);
  };

  bool isAlloc() override { return true; }

  std::string getOpName() override { return "init"; };

  bool prefixEquivalentTo(DependencyNode *other) override {
    if (!other->isAlloc())
      return false;

    auto other_init = static_cast<InitDependencyNode *>(other);

    return qubit && other_init->qubit && qubit == other_init->qubit;
  }

  void eraseQID(VirtualQID qid) override {
    assert(false && "Can't call eraseQID with an InitDependencyNode");
  }

  SetVector<PhysicalQID> getQubits() override {
    SetVector<PhysicalQID> qubits;
    if (qubit)
      qubits.insert(qubit.value());
    return qubits;
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    assert(resultidx == 0 && "Invalid resultidx");
    return std::optional(getQID());
  }
};

class OpDependencyNode : public DependencyNode {
  friend class IfDependencyNode;

protected:
  Operation *associated;
  bool quantumOp;

  virtual void printNode() override {
    llvm::outs() << "QIDs: ";
    bool printComma = false;
    for (auto qid : qids) {
      if (printComma)
        llvm::outs() << ", ";
      llvm::outs() << qid;
      printComma = true;
    }
    if (cycle.has_value())
      llvm::outs() << " @ " << cycle.value();
    llvm::outs() << " | " << height << ", " << numTicks() << " | ";
    associated->dump();
  }

  virtual uint numTicks() override { return isQuantumOp() ? 1 : 0; }
  virtual bool isQuantumOp() override { return quantumOp; }

  Value getResult(uint resultidx) override {
    return associated->getResult(resultidx);
  }

  ValueRange getResults() override { return associated->getResults(); }

  SmallVector<mlir::Value> gatherOperands(OpBuilder &builder,
                                          LifeTimeAnalysis &set) {
    SmallVector<mlir::Value> operands(dependencies.size());
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

  virtual void genOp(OpBuilder &builder, LifeTimeAnalysis &set) {
    auto oldOp = associated;
    auto operands = gatherOperands(builder, set);

    associated =
        Operation::create(oldOp->getLoc(), oldOp->getName(),
                          oldOp->getResultTypes(), operands, oldOp->getAttrs());
    associated->removeAttr("dnodeid");
    builder.insert(associated);
  }

  /// Generates a new operation for this node in the dependency graph
  /// using the dependencies of the node as operands.
  virtual void codeGen(OpBuilder &builder, LifeTimeAnalysis &set) override {
    if (hasCodeGen && isQuantumDependent())
      return;

    // Non-quake operations have code generated aggressively
    // This ensures that code gen is not too aggressive
    if (isSkip())
      for (auto dependency : dependencies)
        if (!dependency->hasCodeGen && dependency->isQuantumDependent())
          // Wait for quantum op dependency to be codeGen'ed
          return;

    genOp(builder, set);
    hasCodeGen = true;

    // Ensure classical values are generated
    for (auto successor : successors)
      if (successor->isSkip() && isQuantumDependent())
        successor->codeGen(builder, set);
  }

public:
  OpDependencyNode(Operation *op, SmallVector<DependencyEdge> _dependencies)
      : associated(op) {
    assert(op && "Cannot make dependency node for null op");
    assert(_dependencies.size() == op->getNumOperands() &&
           "Wrong # of dependencies to construct node");

    dependencies = _dependencies;

    quantumOp = isQuakeOperation(op);
    if (isa<quake::DiscriminateOp>(op))
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
      if (edge.qid.has_value() && quantumOp)
        qids.insert(edge.qid.value());
    }

    updateHeight();
  };

  void print() { printSubGraph(0); }

  uint getHeight() { return height; }

  DependencyEdge getDependencyForResult(size_t resultidx) {
    return dependencies[getOperandIDXFromResultIDX(resultidx, associated)];
  }

  size_t getResultForDependency(size_t operandidx) {
    return getResultIDXFromOperandIDX(operandidx, associated);
  }

  virtual void eraseQID(VirtualQID qid) override {
    qids.remove(qid);
    for (auto successor : successors) {
      for (uint j = 0; j < successor->dependencies.size(); j++) {
        auto edge = successor->dependencies[j];
        if (edge.node == this && edge.qid == qid) {
          auto operandIDX =
              getOperandIDXFromResultIDX(edge.resultidx, associated);
          auto dep = dependencies[operandIDX];
          successor->dependencies[j] = dep;
          dependencies.erase(dependencies.begin() + operandIDX);
          // TODO: only remove if all paths from dep are removed
          dep->successors.remove(this);
          dep->successors.insert(successor);
          return;
        }
      }
    }
  }

  /// Remove this dependency node from the graph by replacing all successor
  /// dependencies with the relevant dependency from this node.
  void erase() {
    for (auto successor : successors) {
      bool remove = true;
      for (uint j = 0; j < successor->dependencies.size(); j++) {
        auto edge = successor->dependencies[j];
        if (edge.node == this) {
          // If the output isn't a linear type, then don't worry about moving it
          if (quake::isQuantumType(edge.getValue().getType())) {
            auto dep = getDependencyForResult(edge.resultidx);
            successor->dependencies[j] = dep;
            dep->successors.insert(successor);
          } else {
            remove = false;
          }
        }
      }

      if (remove)
        successors.remove(successor);
    }

    for (auto dependency : dependencies) {
      dependency->successors.remove(this);
      if (dependency->successors.empty() && !dependency->isLeaf())
        static_cast<OpDependencyNode *>(dependency.node)->erase();
    }
  }

  std::string getOpName() override {
    if (isa<arith::ConstantOp>(associated)) {
      if (auto cstf = dyn_cast<arith::ConstantFloatOp>(associated)) {
        auto value = cstf.getValue().cast<FloatAttr>().getValueAsDouble();
        return std::to_string(value);
      } else if (auto csti = dyn_cast<arith::ConstantIndexOp>(associated)) {
        auto value = cstf.getValue().cast<IntegerAttr>().getInt();
        return std::to_string(value);
      } else if (auto csti = dyn_cast<arith::ConstantIntOp>(associated)) {
        auto value = cstf.getValue().cast<IntegerAttr>().getInt();
        return std::to_string(value);
      }
    }
    return associated->getName().getStringRef().str();
  };

  virtual std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    if (!isQuantumOp())
      return std::nullopt;
    auto operand = getOperandIDXFromResultIDX(resultidx, associated);
    if (operand >= dependencies.size())
      return std::nullopt;
    return dependencies[operand].qid;
  }
};

class DependencyGraph {
private:
  SetVector<DependencyNode *> roots;
  DenseMap<VirtualQID, InitDependencyNode *> allocs;
  DenseMap<VirtualQID, DependencyNode *> leafs;
  SetVector<VirtualQID> qids;
  DenseMap<PhysicalQID, DependencyNode *> qubits;
  uint total_height;
  bool isScheduled = false;
  DependencyNode *tallest = nullptr;
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

    if (next->isLeaf() && next->isQuantumOp()) {
      leafs.insert({next->qids.front(), next});
      qids.insert(next->qids.front());
    }

    if (next->isAlloc()) {
      auto init = static_cast<InitDependencyNode *>(next);
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
    SetVector<DependencyNode *> seen;
    for (auto root : roots)
      nodes.set_union(root->getNodesAtCycle(cycle, seen));
    return nodes;
  }

  void updateHeight(SetVector<DependencyNode *> &seen, DependencyNode *next) {
    if (seen.contains(next))
      return;

    seen.insert(next);

    for (auto dependency : next->dependencies)
      updateHeight(seen, dependency.node);

    next->updateHeight();
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
  void schedule(SetVector<DependencyNode *> &seen, DependencyNode *next,
                uint level) {
    // Ignore classical values that don't depend on quantum values
    if (seen.contains(next) || !next->isQuantumDependent())
      return;

    seen.insert(next);

    // The height of a node (minus numTicks()) is the earliest a node can be
    // scheduled
    if (level < next->height)
      level = next->height;

    uint current = level;
    if (!next->isSkip()) {
      current -= next->numTicks();
      next->cycle = current;
    }

    // Sort dependencies by height to always follow the longest path first.
    // Without this, two dependencies may be scheduled at the same cycle,
    // even if one of the dependencies depends on the other.
    // This sort of mimics working over a transitive reduction of the graph.
    SmallVector<DependencyNode::DependencyEdge> sorted(next->dependencies);
    std::sort(sorted.begin(), sorted.end(), [](auto x, auto y) {
      return x.node->getHeight() > y.node->getHeight();
    });

    // Schedule dependencies as late as possible
    for (auto dependency : sorted)
      if (!dependency->isLeaf())
        schedule(seen, dependency.node, current);

    // Schedule unscheduled successors as early as possible
    for (auto successor : next->successors)
      if (!successor->isRoot())
        schedule(seen, successor,
                 current + next->numTicks() + successor->numTicks());
  }

  void replaceLeaf(VirtualQID old_qid, VirtualQID new_qid,
                   DependencyNode *new_leaf) {
    assert(new_leaf->isLeaf() && "Invalid leaf!");

    if (leafs.count(old_qid) == 1) {
      auto first_use = getFirstUseOfQID(old_qid);
      auto old_leaf = leafs[old_qid];

      // TODO: use replaceWith
      for (uint i = 0; i < first_use->dependencies.size(); i++)
        if (first_use->dependencies[i].node == old_leaf)
          first_use->dependencies[i] =
              DependencyNode::DependencyEdge(new_leaf, 0);
      old_leaf->successors.remove(first_use);
      new_leaf->successors.insert(first_use);
      if (old_leaf->isAlloc())
        allocs.erase(allocs.find(old_qid));
    }

    leafs[new_qid] = new_leaf;
    if (new_leaf->isAlloc()) {
      auto alloc = static_cast<InitDependencyNode *>(new_leaf);
      allocs[new_qid] = alloc;
      if (alloc->qubit)
        qubits[alloc->qubit.value()] = alloc;
    }
  }

  void replaceRoot(VirtualQID qid, VirtualQID new_qid,
                   DependencyNode *new_root) {
    assert(new_root->isRoot() && "Invalid root!");

    if (qids.contains(qid)) {
      DependencyNode *old_root = getRootForQID(qid);
      auto last_use = getLastUseOfQID(qid);

      for (uint i = 0; i < old_root->dependencies.size(); i++) {
        auto edge = old_root->dependencies[i];
        if (edge.qid == qid) {
          new_root->dependencies.push_back(edge);
          old_root->dependencies.erase(old_root->dependencies.begin() + i);
          break;
        }
      }

      if (old_root->dependencies.size() == 0)
        roots.remove(old_root);

      last_use->successors.remove(old_root);
      last_use->successors.insert(new_root);
      old_root->qids.remove(qid);
    }

    new_root->qids.insert(new_qid);
    new_root->qids.insert(qid);
    roots.insert(new_root);
  }

public:
  DependencyGraph(DependencyNode *root) {
    total_height = 0;
    SetVector<DependencyNode *> seen;
    qids = SetVector<VirtualQID>();
    gatherRoots(seen, root);
    if (roots.size() == 0)
      return;
  }

  SetVector<DependencyNode *> &getRoots() { return roots; }

  SetVector<VirtualQID> getQIDs() { return SetVector<VirtualQID>(qids); }

  size_t getNumQIDs() { return qids.size(); }

  LifeTime *getLifeTimeForQID(VirtualQID qid) {
    auto first_use = getFirstUseOfQID(qid);
    auto last_use = getLastUseOfQID(qid);
    assert(first_use->cycle.has_value() &&
           "Graph must be scheduled before lifetimes can be ascertained");
    assert(last_use->cycle.has_value() &&
           "Graph must be scheduled before lifetimes can be ascertained");
    uint first = first_use->cycle.value();
    auto last = last_use->cycle.value();

    return new LifeTime(first, last);
  }

  LifeTime *getLifeTimeForQubit(PhysicalQID qubit) {
    DependencyNode *first_use = getFirstUseOfQubit(qubit);
    DependencyNode *last_use = getLastUseOfQubit(qubit);
    assert(first_use->cycle.has_value() &&
           "Graph must be scheduled before lifetimes can be ascertained");
    assert(last_use->cycle.has_value() &&
           "Graph must be scheduled before lifetimes can be ascertained");
    uint first = first_use->cycle.value();
    auto last = last_use->cycle.value();

    return new LifeTime(first, last);
  }

  OpDependencyNode *getFirstUseOfQID(VirtualQID qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    DependencyNode *firstUse = leafs[qid]->successors[0];
    if (firstUse->isRoot())
      return nullptr;
    return static_cast<OpDependencyNode *>(firstUse);
  }

  OpDependencyNode *getLastUseOfQID(VirtualQID qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    DependencyNode *root = getRootForQID(qid);
    DependencyNode *lastUse;
    for (auto dependency : root->dependencies) {
      if (dependency.qid == qid) {
        lastUse = dependency.node;
        break;
      }
    }
    if (lastUse->isLeaf())
      return nullptr;
    return static_cast<OpDependencyNode *>(lastUse);
  }

  OpDependencyNode *getFirstUseOfQubit(PhysicalQID qubit) {
    assert(qubits.count(qubit) == 1 && "Given qubit not in dependency graph");
    auto defining = qubits[qubit];
    // Qubit is defined here, return the first use
    if (defining->isAlloc()) {
      auto alloc = static_cast<InitDependencyNode *>(defining);
      return getFirstUseOfQID(alloc->getQID());
    }

    // Qubit is defined in a container which is an OpDependencyNode
    return static_cast<OpDependencyNode *>(defining);
  }

  OpDependencyNode *getLastUseOfQubit(PhysicalQID qubit) {
    assert(qubits.count(qubit) == 1 && "Given qubit not in dependency graph");
    auto defining = qubits[qubit];
    // Qubit is defined here, return the last use
    if (defining->isAlloc()) {
      auto alloc = static_cast<InitDependencyNode *>(defining);
      return getLastUseOfQID(alloc->getQID());
    }

    // Qubit is defined in a container which is an OpDependencyNode
    return static_cast<OpDependencyNode *>(defining);
  }

  DependencyNode *getRootForQID(VirtualQID qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    for (auto root : roots)
      if (root->qids.contains(qid))
        return root;
    return nullptr;
  }

  InitDependencyNode *getAllocForQID(VirtualQID qid) {
    assert(allocs.count(qid) == 1 && "Given qid not allocated in graph");
    return allocs[qid];
  }

  void codeGenAt(uint cycle, OpBuilder &builder, LifeTimeAnalysis &set) {
    SetVector<DependencyNode *> nodes = getNodesAtCycle(cycle);

    for (auto node : nodes)
      node->codeGen(builder, set);
  }

  uint getHeight() { return total_height; }

  SetVector<VirtualQID> getAllocs() {
    SetVector<VirtualQID> allocated;
    for (auto [qid, _] : allocs)
      allocated.insert(qid);
    return allocated;
  }

  SetVector<PhysicalQID> getQubits() {
    auto allocated = SetVector<PhysicalQID>();
    for (auto [qubit, _] : qubits)
      allocated.insert(qubit);
    return allocated;
  }

  SetVector<PhysicalQID> getAllocatedQubits() {
    auto allocated = SetVector<PhysicalQID>();
    for (auto [qubit, definining] : qubits)
      if (definining->isAlloc())
        allocated.insert(qubit);
    return allocated;
  }

  void assignToPhysical(VirtualQID qid, PhysicalQID phys) {
    if (allocs.count(qid) == 1)
      allocs[qid]->assignToPhysical(phys);
    qubits[phys] = allocs[qid];
  }

  void addPhysicalAllocation(DependencyNode *container, PhysicalQID qubit) {
    assert(containers.contains(container) &&
           "Illegal container in addPhysicalAllocation");
    qubits[qubit] = container;
  }

  /// Qubits allocated within a dependency block that are only used inside an
  /// `if` in that block, can be moved inside the `if`.
  ///
  /// Works outside-in, to contract as tightly as possible.
  void contractAllocsPass() {
    for (auto container : containers)
      container->contractAllocsPass();
  }

  /// Assigns a cycle to every quantum operation in each dependency graph
  /// (including `if`s containing quantum operations).
  ///
  /// Currently works inside-out, but scheduling is order-agnostic
  /// as inner-blocks don't rely on parent schedules, and vice-versa.
  void schedulingPass() {
    if (!tallest) {
      assert(roots.empty() &&
             "updateHeight not invoked before scheduling graph!");
      return;
    }
    SetVector<DependencyNode *> seen;
    schedule(seen, tallest, total_height);
  }

  void print() {
    llvm::outs() << "Graph Start\n";
    for (auto root : roots)
      root->print();
    llvm::outs() << "Graph End\n";
  }

  void performAnalysis(LifeTimeAnalysis &set) {
    for (auto container : containers)
      container->performAnalysis(set, this);
  }

  void removeVirtualAlloc(VirtualQID qid) {
    // Ignore already removed qid
    if (allocs.count(qid) == 1)
      allocs.erase(allocs.find(qid));

    if (qids.contains(qid)) {
      auto toRemove = getRootForQID(qid);
      roots.remove(toRemove);
    }
  }

  /// Simultaneously replaces the leaf and root nodes for a given qid, or
  /// adds them if the qid was not present before.
  /// The operations are separate, but doing them together makes it harder
  /// to produce an invalid graph
  void replaceLeafAndRoot(VirtualQID qid, DependencyNode *new_leaf,
                          DependencyNode *new_root) {
    auto new_qid = qid;
    if (!new_leaf->qids.empty())
      new_qid = new_leaf->qids.front();

    replaceLeaf(qid, new_qid, new_leaf);
    replaceRoot(qid, new_qid, new_root);

    qids.insert(new_qid);

    if (new_qid != qid) {
      qids.remove(qid);
      new_leaf->updateQID(qid, new_qid);
    }
  }

  void removeQID(VirtualQID qid) {
    leafs.erase(leafs.find(qid));
    qids.remove(qid);
  }

  void updateHeight() {
    total_height = 0;
    tallest = nullptr;
    SetVector<DependencyNode *> seen;
    for (auto root : roots) {
      updateHeight(seen, root);
      if (!tallest || root->height > total_height) {
        tallest = root;
        total_height = root->height;
      }
    }
  }

  // void updateLeafs() {
  //   for (auto qid : qids) {
  //     auto leaf = leafs[qid];
  //     if (leaf->successors.empty()) {
  //       leafs.erase(leafs.find(qid));
  //       if (allocs.count(qid) == 1)
  //         allocs.erase(allocs.find(qid));
  //       qids.remove(qid);
  //     }
  //   }
  // }
};

class RootDependencyNode : public OpDependencyNode {
protected:
  void printNode() override {
    llvm::outs() << "Dealloc for QID ";
    for (auto qid : qids)
      llvm::outs() << qid;
    llvm::outs() << ": ";
    associated->dump();
  }

  bool isSkip() override { return true; }

  uint numTicks() override { return 0; }

  void genOp(OpBuilder &builder, LifeTimeAnalysis &set) override {
    auto wire = dependencies[0].getValue();
    auto newOp =
        builder.create<quake::ReturnWireOp>(builder.getUnknownLoc(), wire);
    newOp->setAttrs(associated->getAttrs());
    newOp->removeAttr("dnodeid");
    associated = newOp;
  }

public:
  RootDependencyNode(quake::ReturnWireOp op,
                     SmallVector<DependencyEdge> dependencies)
      : OpDependencyNode(op, dependencies) {
    // numTicks won't be properly calculated by OpDependencyNode constructor,
    // so have to recompute height here
    updateHeight();
  };

  void eraseQID(VirtualQID qid) override {
    if (qids.contains(qid))
      dependencies.clear();
  }
};

class ArgDependencyNode : public DependencyNode {
  friend class DependencyBlock;
  friend class IfDependencyNode;

protected:
  BlockArgument barg;
  uint argNum;

  void printNode() override {
    if (qids.size() > 0)
      llvm::outs() << "QID: " << qids.front() << ", ";
    barg.dump();
  }

  bool isRoot() override { return false; }
  bool isLeaf() override { return true; }
  bool isQuantumOp() override { return quake::isQuantumType(barg.getType()); }
  uint numTicks() override { return 0; }

  Value getResult(uint resultidx) override {
    assert(resultidx == 0 && "Invalid resultidx");
    return barg;
  }

  ValueRange getResults() override { return ValueRange({barg}); }

  void codeGen(OpBuilder &builder, LifeTimeAnalysis &set) override{};

public:
  ArgDependencyNode(BlockArgument arg)
      : barg(arg), argNum(arg.getArgNumber()) {}

  ArgDependencyNode(BlockArgument arg, uint num) : barg(arg), argNum(num) {}

  ArgDependencyNode(BlockArgument arg, DependencyEdge val)
      : ArgDependencyNode(arg) {
    auto qid = val->getQIDForResult(val.resultidx);
    if (qid.has_value())
      qids.insert(qid.value());
  }

  ArgDependencyNode(BlockArgument arg, DependencyEdge val, uint num)
      : barg(arg), argNum(num) {
    auto qid = val->getQIDForResult(val.resultidx);
    if (qid.has_value())
      qids.insert(qid.value());
  }

  virtual std::string getOpName() override {
    return std::to_string(barg.getArgNumber()).append("arg");
  };

  void eraseQID(VirtualQID qid) override {
    assert(false && "Can't call eraseQID with an ArgDependencyNode");
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    assert(resultidx == 0 && "Invalid resultidx");
    if (qids.size() == 1)
      return std::optional(qids.front());
    return std::nullopt;
  }

  uint getArgNumber() { return argNum; }
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

  uint numTicks() override { return 0; }

  bool isQuantumOp() override { return qids.size() > 0; }

  void codeGen(OpBuilder &builder, LifeTimeAnalysis &set) override{};

public:
  TerminatorDependencyNode(Operation *terminator,
                           SmallVector<DependencyEdge> dependencies)
      : OpDependencyNode(terminator, dependencies) {
    assert(terminator->hasTrait<mlir::OpTrait::ReturnLike>() &&
           "Invalid terminator");
    for (auto dependency : dependencies)
      if (dependency.qid.has_value())
        qids.insert(dependency.qid.value());
  }

  void genTerminator(OpBuilder &builder, LifeTimeAnalysis &set) {
    OpDependencyNode::codeGen(builder, set);
  }

  void eraseQID(VirtualQID qid) override {
    for (uint i = 0; i < dependencies.size(); i++)
      if (dependencies[i].qid == qid)
        dependencies.erase(dependencies.begin() + i);
    qids.remove(qid);
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    assert(resultidx < dependencies.size() && "Invalid ressultidx");
    return dependencies[resultidx].qid;
  }
};

class DependencyBlock {
private:
  SmallVector<ArgDependencyNode *> argdnodes;
  DependencyGraph *graph;
  Block *block;
  TerminatorDependencyNode *terminator;
  uint height;
  SetVector<size_t> pqids;

public:
  DependencyBlock(SmallVector<ArgDependencyNode *> argdnodes,
                  DependencyGraph *graph, Block *block,
                  TerminatorDependencyNode *terminator)
      : argdnodes(argdnodes), graph(graph), block(block),
        terminator(terminator), pqids() {
    height = graph->getHeight();
  }

  uint getHeight() { return height; }

  SetVector<VirtualQID> getAllocs() { return graph->getAllocs(); }

  SetVector<VirtualQID> getQIDs() { return graph->getQIDs(); }

  OpDependencyNode *getFirstUseOfQID(VirtualQID qid) {
    return graph->getFirstUseOfQID(qid);
  }

  OpDependencyNode *getLastUseOfQID(VirtualQID qid) {
    return graph->getLastUseOfQID(qid);
  }

  OpDependencyNode *getFirstUseOfQubit(PhysicalQID qubit) {
    return graph->getFirstUseOfQubit(qubit);
  }

  OpDependencyNode *getLastUseOfQubit(PhysicalQID qubit) {
    return graph->getLastUseOfQubit(qubit);
  }

  DependencyNode *getRootForQID(VirtualQID qid) {
    return graph->getRootForQID(qid);
  }

  void allocatePhyiscalQubits(LifeTimeAnalysis &set) {
    for (auto qubit : graph->getQubits()) {
      auto lifetime = graph->getLifeTimeForQubit(qubit);
      set.reallocatePhysical(qubit, lifetime);
    }

    // New physical qubits will be captured by `set`
    for (auto qid : getAllocs()) {
      auto leaf = graph->getAllocForQID(qid);
      if (!leaf->getQubits().empty())
        continue;

      auto lifetime = graph->getLifeTimeForQID(qid);
      LLVM_DEBUG(llvm::dbgs() << "Qid " << qid);
      LLVM_DEBUG(llvm::dbgs()
                 << " is in use from cycle " << lifetime->getBegin());
      LLVM_DEBUG(llvm::dbgs() << " through cycle " << lifetime->getEnd());
      LLVM_DEBUG(llvm::dbgs() << "\n");

      auto phys = set.allocatePhysical(qid, lifetime);
      LLVM_DEBUG(llvm::dbgs()
                 << "\tIt is mapped to the physical qubit " << phys);
      LLVM_DEBUG(llvm::dbgs() << "\n\n");

      graph->assignToPhysical(qid, phys);
    }
  }

  /// Up to caller to move builder outside block after construction
  Block *codeGen(OpBuilder &builder, Region *region, LifeTimeAnalysis &set) {
    Block *newBlock = builder.createBlock(region);
    for (uint i = 0; i < argdnodes.size(); i++) {
      auto old_barg = argdnodes[i]->barg;
      argdnodes[i]->barg =
          newBlock->addArgument(old_barg.getType(), old_barg.getLoc());
      assert(argdnodes[i]->barg.getArgNumber() == argdnodes[i]->argNum);
      argdnodes[i]->hasCodeGen = true;
    }

    builder.setInsertionPointToStart(newBlock);

    for (uint cycle = 0; cycle < height; cycle++)
      graph->codeGenAt(cycle, builder, set);

    terminator->genTerminator(builder, set);

    block = newBlock;

    return newBlock;
  }

  void print() {
    llvm::outs() << "Block with (" << argdnodes.size() << ") args:\n";
    // block->dump();
    // llvm::outs() << "Block graph:\n";
    graph->print();
    llvm::outs() << "End block\n";
  }

  void updateHeight() {
    graph->updateHeight();
    height = graph->getHeight();
  }

  void performAnalysis(LifeTimeAnalysis &set) {
    // The analysis works inside-out, so first resolve all nested `if`s
    graph->performAnalysis(set);

    // Update metadata after the analysis
    // graph->updateLeafs();
    updateHeight();
    // Schedule the nodes for lifetime analysis
    schedulingPass();
    // Finally, perform lifetime analysis and allocate physical qubits
    // Allocations will be captured in `set`
    allocatePhyiscalQubits(set);
  }

  /// Checks to see if qubits allocated within a block are only used
  /// inside an `if` in that block, in which case they can be moved
  /// inside the `if`.
  ///
  /// Works outside-in, to contract as tightly as possible.
  void contractAllocsPass() {
    // Look for contract-able allocations in this block
    for (auto alloc : getAllocs()) {
      auto first_use = getFirstUseOfQID(alloc);
      auto last_use = getLastUseOfQID(alloc);
      if (first_use == last_use && first_use->isContainer()) {
        // Move alloc inside
        auto root = graph->getRootForQID(alloc);
        auto init = graph->getAllocForQID(alloc);
        first_use->lowerAlloc(init, root, alloc);
        // Qid is no longer used in this block, remove related metadata
        graph->removeVirtualAlloc(alloc);
        graph->removeQID(alloc);
      }
    }

    // Outside-in, so recur only after applying pass to this block
    graph->contractAllocsPass();
  }

  void lowerAlloc(DependencyNode *init, DependencyNode *root, VirtualQID qid) {
    removeArgument(qid);
    graph->replaceLeafAndRoot(qid, init, root);
  }

  void liftAlloc(VirtualQID qid, DependencyNode *lifted_alloc) {
    auto new_edge = DependencyNode::DependencyEdge{lifted_alloc, 0};
    auto new_argdnode = addArgument(new_edge);

    graph->replaceLeafAndRoot(qid, new_argdnode, terminator);
  }

  void schedulingPass() { graph->schedulingPass(); }

  void removeQID(VirtualQID qid) {
    removeArgument(qid);

    terminator->eraseQID(qid);
    graph->removeQID(qid);
  }

  SetVector<PhysicalQID> getQubits() { return graph->getQubits(); }

  SetVector<PhysicalQID> getAllocatedQubits() {
    return graph->getAllocatedQubits();
  }

  DependencyNode *addArgument(DependencyNode::DependencyEdge incoming) {
    auto new_barg = block->addArgument(incoming.getValue().getType(),
                                       incoming.getValue().getLoc());
    auto new_argdnode =
        new ArgDependencyNode(new_barg, incoming, argdnodes.size());
    argdnodes.push_back(new_argdnode);
    return new_argdnode;
  }

  DependencyNode *addArgument(Type wireTy, mlir::Location loc) {
    auto new_barg = block->addArgument(wireTy, loc);
    auto new_argdnode = new ArgDependencyNode(new_barg);
    argdnodes.push_back(new_argdnode);
    return new_argdnode;
  }

  void removeArgument(VirtualQID qid) {
    for (uint i = 0; i < argdnodes.size(); i++)
      if (argdnodes[i]->qids.contains(qid)) {
        argdnodes.erase(argdnodes.begin() + i);
        break;
      }
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) {
    return terminator->getQIDForResult(resultidx);
  }
};

class IfDependencyNode : public OpDependencyNode {
protected:
  DependencyBlock *then_block;
  DependencyBlock *else_block;
  SmallVector<Type> results;

  // TODO: figure out nice way to display
  void printNode() override {
    this->OpDependencyNode::printNode();
    // llvm::outs() << "If with results:\n";
    // for (auto result : results)
    //   result.dump();
    llvm::outs() << "Then ";
    then_block->print();
    llvm::outs() << "Else ";
    else_block->print();
  }

  uint numTicks() override {
    return std::max(then_block->getHeight(), else_block->getHeight());
  }

  bool isSkip() override { return numTicks() == 0; }

  bool isQuantumOp() override { return numTicks() > 0; }

  void liftOpAfter(OpDependencyNode *then_op, OpDependencyNode *else_op,
                   DependencyGraph *parent) {
    auto newDeps = SmallVector<DependencyEdge>();
    auto allocated = then_block->getAllocatedQubits();
    assert(then_op->dependencies.size() == then_op->successors.size());
    for (uint i = 0; i < then_op->dependencies.size(); i++) {
      auto dependency = then_op->dependencies[i];
      assert(dependency.qid && "Lifting operations with classical input after "
                               "blocks is not yet supported.");

      auto then_qid = dependency.qid.value();
      then_op->eraseQID(then_qid);

      // Lift allocated qubit
      if (dependency.qubit && allocated.contains(dependency.qubit.value())) {
        auto else_qid = else_op->dependencies[i].qid.value();
        // Remove virtual allocs from inner blocks
        auto lifted_alloc = dependency.node;
        auto lifted_root = then_block->getRootForQID(then_qid);
        then_block->liftAlloc(then_qid, lifted_alloc);
        else_block->liftAlloc(else_qid, lifted_alloc);

        // Add virtual alloc to current scope
        parent->replaceLeafAndRoot(then_qid, lifted_alloc, lifted_root);
        DependencyEdge newEdge(then_op, then_op->getResultForDependency(i));
        lifted_root->dependencies.push_back(newEdge);
        lifted_alloc->successors.insert(this);
        qids.insert(then_qid);

        newDeps.push_back(DependencyEdge{this, results.size()});
        dependencies.push_back(DependencyEdge{lifted_alloc, 0});
        then_op->successors.insert(lifted_root);
        results.push_back(dependency.getValue().getType());
      } else {
        auto resultidx = then_op->getResultForDependency(i);
        for (auto successor : successors) {
          for (uint i = 0; i < successor->dependencies.size(); i++)
            if (successor->dependencies[i].node == this &&
                successor->dependencies[i].qid == dependency.qid) {
              newDeps.push_back(successor->dependencies[i]);
              successor->dependencies[i] = DependencyEdge{then_op, resultidx};
              break;
            }
        }
      }

      then_op->qids.insert(then_qid);
    }

    successors.insert(then_op);
    then_op->dependencies = newDeps;
    else_op->erase();
  }

  void liftOpBefore(OpDependencyNode *then_op, OpDependencyNode *else_op,
                    DependencyGraph *parent) {
    auto newDeps = SmallVector<DependencyEdge>();

    // Measure ops are a delicate special case because of the classical measure
    // result. When lifting before, we can lift the discriminate op as well.
    if (isa<RAW_MEASURE_OPS>(then_op->associated)) {
      auto then_discriminate = then_op->successors.front()->isQuantumOp()
                                   ? then_op->successors.back()
                                   : then_op->successors.front();
      auto else_discriminate = else_op->successors.front()->isQuantumOp()
                                   ? else_op->successors.back()
                                   : else_op->successors.front();
      else_discriminate->replaceWith(DependencyEdge{then_discriminate, 0});
    }

    // Construct new dependencies
    for (uint i = 0; i < then_op->dependencies.size(); i++) {
      auto dependency = then_op->dependencies[i];

      if (dependency->isAlloc()) {
        auto then_qid = dependency.qid.value();
        auto else_qid = else_op->dependencies[i].qid.value();
        // Remove virtual allocs from inner blocks
        auto lifted_alloc = dependency.node;
        auto lifted_root = then_block->getRootForQID(then_qid);
        then_block->liftAlloc(then_qid, lifted_alloc);
        else_block->liftAlloc(else_qid, lifted_alloc);

        // Add virtual alloc to current scope
        this->successors.insert(lifted_root);
        parent->replaceLeafAndRoot(then_qid, lifted_alloc, lifted_root);
        qids.insert(then_qid);
        newDeps.push_back(dependency);
        DependencyEdge newEdge(then_op, then_op->getResultForDependency(i));
        dependencies.push_back(newEdge);
        lifted_root->dependencies.push_back(
            DependencyEdge{this, results.size()});
        lifted_alloc->successors.insert(then_op);
        results.push_back(dependency.getValue().getType());
      } else if (!dependency->isQuantumOp()) {
        newDeps.push_back(dependency);
      } else if (dependency->isLeaf()) {
        ArgDependencyNode *arg =
            static_cast<ArgDependencyNode *>(dependency.node);
        auto num = arg->getArgNumber();
        auto newDep = dependencies[num + 1];
        newDep->successors.remove(this);
        newDep->successors.insert(then_op);
        newDeps.push_back(newDep);
        arg->successors.remove(then_op);

        dependencies[num + 1] =
            DependencyEdge{then_op, then_op->getResultForDependency(i)};
      }
    }

    then_op->erase();
    else_op->erase();

    // Patch successors
    then_op->successors.insert(this);
    then_op->dependencies = newDeps;
  }

  void combineAllocs(SetVector<PhysicalQID> then_allocs,
                     SetVector<PhysicalQID> else_allocs, LifeTimeAnalysis &set,
                     DependencyGraph *graph) {
    SetVector<PhysicalQID> combined;
    combined.set_union(then_allocs);
    combined.set_union(else_allocs);

    for (auto qubit : combined)
      graph->addPhysicalAllocation(this, qubit);
  }

  void genOp(OpBuilder &builder, LifeTimeAnalysis &set) override {
    cudaq::cc::IfOp oldOp = dyn_cast<cudaq::cc::IfOp>(associated);
    auto operands = gatherOperands(builder, set);

    auto newIf =
        builder.create<cudaq::cc::IfOp>(oldOp->getLoc(), results, operands);
    auto *then_region = &newIf.getThenRegion();
    then_block->codeGen(builder, then_region, set);

    auto *else_region = &newIf.getElseRegion();
    else_block->codeGen(builder, else_region, set);

    associated = newIf;
    builder.setInsertionPointAfter(associated);
  }

  SetVector<PhysicalQID> getQubits() override {
    auto qubits = SetVector<PhysicalQID>();
    qubits.set_union(then_block->getQubits());
    qubits.set_union(else_block->getQubits());
    return qubits;
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    return then_block->getQIDForResult(resultidx);
  }

public:
  IfDependencyNode(cudaq::cc::IfOp op, SmallVector<DependencyEdge> dependencies,
                   DependencyBlock *then_block, DependencyBlock *else_block)
      : OpDependencyNode(op.getOperation(), dependencies),
        then_block(then_block), else_block(else_block) {
    results = SmallVector<mlir::Type>(op.getResultTypes());
    // Unfortunately, some metadata won't be computed properly by
    // OpDependencyNode constructor, so recompute here
    height = 0;
    for (auto edge : dependencies) {
      if (edge->getHeight() > height)
        height = edge->getHeight();
      if (edge.qid.has_value() && isQuantumOp())
        qids.insert(edge.qid.value());
    }
    height += numTicks();
  }

  void contractAllocsPass() override {
    then_block->contractAllocsPass();
    else_block->contractAllocsPass();
  }

  void eraseQID(VirtualQID qid) override {
    for (uint i = 0; i < dependencies.size(); i++)
      if (dependencies[i].qid == qid)
        results.erase(results.begin() + i - 1);

    then_block->removeQID(qid);
    else_block->removeQID(qid);
    this->OpDependencyNode::eraseQID(qid);
    if (results.empty())
      dependencies[0]->successors.remove(this);
  }

  bool tryLiftingBefore(OpDependencyNode *then_use, OpDependencyNode *else_use,
                        DependencyGraph *parent) {
    if (!then_use || !else_use)
      return false;

    if (then_use->prefixEquivalentTo(else_use)) {
      // If two nodes are equivalent, all their dependencies will be too,
      // but we can't lift them until all their dependencies have been lifted,
      // so we skip them for now.
      for (auto dependency : then_use->dependencies)
        if (!dependency->isSkip())
          return false;

      liftOpBefore(then_use, else_use, parent);
      return true;
    }

    return false;
  }

  bool tryLiftingAfter(OpDependencyNode *then_use, OpDependencyNode *else_use,
                       DependencyGraph *parent) {
    // TODO: measure ops are a delicate special case because of the classical
    // measure result. When lifting before, we can lift the discriminate op as
    // well. However, it may have interactions with other classical values, and
    // then be "returned" from the `if`
    if (isa<RAW_MEASURE_OPS>(then_use->associated))
      return false;

    if (!then_use || !else_use)
      return false;

    if (then_use->postfixEquivalentTo(else_use)) {
      // If two nodes are equivalent, all their successors should be too
      // but we can't lift them until all their successors have been lifted,
      // so we skip them for now.
      for (auto successor : then_use->successors)
        if (!successor->isSkip())
          return false;
      // TODO: Classical input from within the if scope poses an issue for
      // lifting for a similar reason as measures
      for (auto dependency : then_use->dependencies)
        if (!dependency->isQuantumOp())
          return false;

      liftOpAfter(then_use, else_use, parent);
      return true;
    }

    return false;
  }

  /// Finds and lifts common operations from the then and else branches
  void performLiftingPass(DependencyGraph *parent) {
    bool lifted = false;

    // First, lift allocated qubits, after which they will be dealt with as QIDs
    auto liftableQubits = SetVector<PhysicalQID>();
    liftableQubits.set_union(then_block->getAllocatedQubits());
    liftableQubits.set_union(else_block->getAllocatedQubits());
    for (auto qubit : liftableQubits) {
      auto then_use = then_block->getFirstUseOfQubit(qubit);
      auto else_use = else_block->getFirstUseOfQubit(qubit);

      if (tryLiftingBefore(then_use, else_use, parent)) {
        lifted = true;
        continue;
      }

      then_use = then_block->getLastUseOfQubit(qubit);
      else_use = else_block->getLastUseOfQubit(qubit);

      if (tryLiftingAfter(then_use, else_use, parent)) {
        lifted = true;
        continue;
      }
    }

    // Now, try lifting all QIDs
    bool run_more = true;
    auto unliftableQIDs = SetVector<VirtualQID>();

    // Lifting operations may reveal more liftable operations!
    while (run_more) {
      run_more = false;
      auto liftableQIDs = SetVector<VirtualQID>(qids);
      liftableQIDs.set_subtract(unliftableQIDs);

      for (auto qid : liftableQIDs) {
        auto then_use = then_block->getFirstUseOfQID(qid);
        auto else_use = else_block->getFirstUseOfQID(qid);

        if (tryLiftingBefore(then_use, else_use, parent)) {
          lifted = true;
          continue;
        }

        then_use = then_block->getLastUseOfQID(qid);
        else_use = else_block->getLastUseOfQID(qid);

        if (tryLiftingAfter(then_use, else_use, parent)) {
          lifted = true;
          continue;
        }
      }
    }

    // Recompute inner block metadata after lifting
    if (lifted) {
      then_block->updateHeight();
      else_block->updateHeight();
      then_block->schedulingPass();
      else_block->schedulingPass();
    }
  }

  /// Performs the analysis and optimizations on this `if` statement inside out:
  /// * First, recurs on the then and else blocks
  /// * Physical allocations from the two blocks are combined
  /// * Common operations are lifted from the blocks
  void performAnalysis(LifeTimeAnalysis &set,
                       DependencyGraph *parent_graph) override {
    // Recur first, as analysis works inside-out
    then_block->performAnalysis(set);
    // Capture allocations from then_block analysis
    auto pqids1 = set.getAllocated();
    else_block->performAnalysis(set);
    // Capture allocations from else_block analysis
    auto pqids2 = set.getAllocated();

    // Combine then and else allocations
    combineAllocs(pqids1, pqids2, set, parent_graph);

    // Lift common operations between then and else blocks
    performLiftingPass(parent_graph);
  }

  bool isContainer() override { return true; }

  /// Move a virtual wire allocated and de-allocated (but not used!) from an
  /// outer scope to be allocated and de-allocated within both the then and else
  /// blocks.
  ///
  /// As a result, removes the dependency on, and result for, \p qid from this
  /// node
  void lowerAlloc(DependencyNode *init, DependencyNode *root,
                  VirtualQID qid) override {
    assert(successors.contains(root) && "Illegal root for contractAlloc");
    assert(init->successors.contains(this) && "Illegal init for contractAlloc");
    root->dependencies.erase(root->dependencies.begin());
    init->successors.clear();
    successors.remove(root);
    auto alloc = static_cast<InitDependencyNode *>(init);
    auto alloc_copy = new InitDependencyNode(*alloc);
    auto sink = static_cast<RootDependencyNode *>(root);
    auto sink_copy = new RootDependencyNode(*sink);
    then_block->lowerAlloc(alloc, root, qid);
    else_block->lowerAlloc(alloc_copy, sink_copy, qid);
    auto iter = std::find_if(dependencies.begin(), dependencies.end(),
                             [init](auto edge) { return edge.node == init; });
    size_t offset = iter - dependencies.begin();
    associated->eraseOperand(offset);
    results.erase(results.begin() + offset);
    dependencies.erase(iter);

    // Since we're removing a result, update the result indices of successors
    for (auto successor : successors)
      for (uint i = 0; i < successor->dependencies.size(); i++)
        if (successor->dependencies[i].node == this &&
            successor->dependencies[i].resultidx >= offset)
          successor->dependencies[i].resultidx--;
  }
};

/// Validates that \p op meets the assumptions:
/// * operations are in linear value semantics
/// * control flow operations (except `if`s) are not allowed
/// * memory stores may be rearranged (this is not a hard error)
bool validateOp(Operation *op) {
  if (isQuakeOperation(op) && !quake::isLinearValueForm(op) &&
      !isa<quake::DiscriminateOp>(op)) {
    op->emitOpError("DependencyAnalysisPass: requires all quake operations to "
                    "be in value form");
    return false;
  }

  if (op->getRegions().size() != 0 && !isa<cudaq::cc::IfOp>(op)) {
    op->emitOpError(
        "DependencyAnalysisPass: loops are not supported");
    return false;
  }

  if (isa<mlir::BranchOpInterface>(op)) {
    op->emitOpError(
        "DependencyAnalysisPass: branching operations are not supported");
    return false;
  }

  if (isa<mlir::CallOpInterface>(op)) {
    op->emitOpError(
        "DependencyAnalysisPass: function calls are not supported ");
    return false;
  }

  if (hasEffect<mlir::MemoryEffects::Write>(op) && !isQuakeOperation(op)) {
    op->emitWarning("DependencyAnalysisPass: memory stores are volatile and "
                    "may be reordered");
  }

  if (isa<quake::NullWireOp>(op)) {
    op->emitWarning("DependencyAnalysisPass: `null_wire` are not supported");
  }

  return true;
}

/// Validates that \p func meets the assumptions:
/// * function bodies contain a single block
[[maybe_unused]] bool validateFunc(func::FuncOp func) {
  if (func.getBlocks().size() != 1) {
    func.emitOpError(
        "DependencyAnalysisPass: multiple blocks are not supported");
    return false;
  }

  return true;
}

class DependencyAnalysisEngine {
private:
  SmallVector<DependencyNode *> perOp;
  DenseMap<BlockArgument, ArgDependencyNode *> argMap;
  SetVector<DependencyNode *> constants;

public:
  DependencyAnalysisEngine() : perOp({}), argMap({}) {}

  /// Creates a new dependency block for \p b by constructing a dependency graph
  /// for the body of \p b starting from the block terminator.
  ///
  /// Any operation not somehow connected to the block terminator (this will
  /// only happen if the entirety of a wire is irrelevant to the block
  /// terminator, in which case it is considered to be dead code)
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
  }

    DenseMap<DependencyNode *, Operation *> roots;
    TerminatorDependencyNode *terminator;
    for (auto &op : b->getOperations()) {
      bool isTerminator = (&op == b->getTerminator());
      auto node = visitOp(&op, isTerminator);

      if (!node)
        return nullptr;

      if (isa<quake::ReturnWireOp>(&op))
        roots[node] = &op;

      if (isTerminator) {
        assert(op.hasTrait<mlir::OpTrait::IsTerminator>() &&
               "Illegal terminator op!");
        terminator = static_cast<TerminatorDependencyNode *>(node);
      }
    }

    DependencyGraph *new_graph = new DependencyGraph(terminator);
    auto included = new_graph->getRoots();

    for (auto [root, op] : roots)
      if (!included.contains(root))
        op->emitWarning(
            "DependencyAnalysisPass: Wire is dead code and its operations will "
            "be deleted (did you forget to return a value?)");

    return new DependencyBlock(argdnodes, new_graph, b, terminator);
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

    if (auto init = dyn_cast<quake::BorrowWireOp>(op))
      newNode = new InitDependencyNode(init);
    else if (auto sink = dyn_cast<quake::ReturnWireOp>(op))
      newNode = new RootDependencyNode(sink, dependencies);
    else if (auto ifop = dyn_cast<cudaq::cc::IfOp>(op)) {
      auto then_block = visitBlock(ifop.getThenEntryBlock(), dependencies);
      auto else_block = visitBlock(ifop.getElseEntryBlock(), dependencies);
      if (!then_block || !else_block)
        return nullptr;
      newNode =
          new IfDependencyNode(ifop, dependencies, then_block, else_block);
    } else if (isTerminator) {
      newNode = new TerminatorDependencyNode(op, dependencies);
    } else {
      newNode = new OpDependencyNode(op, dependencies);
      if (!newNode->isQuantumDependent())
        constants.insert(newNode);
    }

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

  SetVector<DependencyNode *> &getConstants() { return constants; }
};

struct DependencyAnalysisPass
    : public cudaq::opt::impl::DependencyAnalysisBase<DependencyAnalysisPass> {
  using DependencyAnalysisBase::DependencyAnalysisBase;
  void runOnOperation() override {
    auto mod = getOperation();

    for (auto &op : mod) {
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        if (!func->hasAttr("cudaq-kernel"))
          continue;

        if (func.getNumResults() == 0) {
          func->emitRemark("Function marked 'cudaq-kernel' returns no results, "
                           "qubit management skipped.");
          continue;
        }

        validateFunc(func);
        Block *oldBlock = &func.front();

        auto engine = DependencyAnalysisEngine();

        auto body = engine.visitBlock(
            oldBlock, SmallVector<DependencyNode::DependencyEdge>());

        auto constants = engine.getConstants();

        if (!body) {
          signalPassFailure();
          return;
        }

        OpBuilder builder(func);
        LifeTimeAnalysis set(cudaq::opt::topologyAgnosticWiresetName);
        // First, move allocs in as deep as possible. This is outside-in, so it
        // is separated from the rest of the analysis passes.
        body->contractAllocsPass();
        // Next, do the scheduling, lifetime analysis/allocation, and lifting
        // passes inside-out
        body->performAnalysis(set);
        // Finally, perform code generation to move back to quake
        body->codeGen(builder, &func.getRegion(), set);

        // Replace old block
        oldBlock->erase();
      }
    }
  }
};

} // namespace
