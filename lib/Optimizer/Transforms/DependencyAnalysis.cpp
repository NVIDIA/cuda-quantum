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
      delete lifetime;
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

  ~LifeTimeAnalysis() {
    for (auto lifetime : lifetimes)
      if (lifetime)
        delete lifetime;
  }

  PhysicalQID allocatePhysical(VirtualQID qid, LifeTime *lifetime) {
    auto phys = allocatePhysical(lifetime);
    frame.insert(phys);
    return phys;
  }

  SetVector<PhysicalQID> getAllocated() {
    for (uint i = 0; i < lifetimes.size(); i++) {
      if (lifetimes[i])
        delete lifetimes[i];
      lifetimes[i] = nullptr;
    }
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

  virtual std::pair<DependencyNode *, size_t>
  getSuccessorAndEdgeIDXForQID(VirtualQID qid) {
    assert(qids.contains(qid) &&
           "Asking for a qid that doesn't flow through this operation!");
    for (auto successor : successors) {
      // Special case: ignore patch discriminate for a measure
      if (!successor->isQuantumOp())
        continue;

      for (uint j = 0; j < successor->dependencies.size(); j++) {
        auto edge = successor->dependencies[j];
        if (edge.node == this && edge.qid == qid)
          return std::make_pair(successor, j);
      }
    }

    assert(false && "Couldn't find successor for linear type!");
  }

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
      for (auto &dependency : successor->dependencies) {
        if (dependency.node == this) {
          dependency = other;
          other->successors.remove(this);
          other->successors.insert(successor);
        }
      }
    }
  }

  virtual void updateWithPhysical(VirtualQID qid, PhysicalQID qubit) {
    for (auto &dependency : dependencies) {
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
    for (uint i = 0; i < dependencies.size(); i++) {
      if (dependencies[i].qid == old_qid) {
        dependencies[i].qid = new_qid;
        break;
      }
    }

    for (auto successor : successors)
      if (successor->qids.contains(old_qid))
        successor->updateQID(old_qid, new_qid);
  }

public:
  DependencyNode() : successors(), dependencies({}), qids({}), height(0) {}

  virtual ~DependencyNode(){};

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
  virtual void eraseEdgeForQID(VirtualQID qid) = 0;

  virtual std::optional<VirtualQID> getQIDForResult(size_t resultidx) = 0;
};

class InitDependencyNode : public DependencyNode {
  friend class DependencyGraph;
  friend class IfDependencyNode;

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

public:
  InitDependencyNode(quake::BorrowWireOp op) : wire(op.getResult()) {
    // Should be ensured by assign-ids pass

    // Lookup qid
    auto qid = op.getIdentity();
    qids.insert(qid);
  };

  VirtualQID getQID() { return qids.front(); }

  ~InitDependencyNode() override {}

  bool isAlloc() override { return true; }

  std::string getOpName() override { return "init"; };

  bool prefixEquivalentTo(DependencyNode *other) override {
    if (!other->isAlloc())
      return false;

    auto other_init = static_cast<InitDependencyNode *>(other);

    return qubit && other_init->qubit && qubit == other_init->qubit;
  }

  void eraseEdgeForQID(VirtualQID qid) override {
    assert(false && "Can't call eraseEdgeForQID with an InitDependencyNode");
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
  friend class ShadowDependencyNode;

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

  virtual ~OpDependencyNode() override {}

  void print() { printSubGraph(0); }

  uint getHeight() { return height; }

  virtual DependencyEdge getDependencyForResult(size_t resultidx) {
    return dependencies[getOperandIDXFromResultIDX(resultidx, associated)];
  }

  size_t getResultForDependency(size_t operandidx) {
    return getResultIDXFromOperandIDX(operandidx, associated);
  }

  virtual void eraseEdgeForQID(VirtualQID qid) override {
    DependencyNode *dependency = nullptr;

    for (auto successor : successors) {
      // Special case: don't patch discriminate for a measure
      if (!successor->isQuantumOp())
        continue;

      bool remove = true;
      for (uint j = 0; j < successor->dependencies.size(); j++) {
        auto edge = successor->dependencies[j];
        if (edge.node == this) {
          if (edge.qid == qid) {
            auto dep = getDependencyForResult(edge.resultidx);
            successor->dependencies[j] = dep;
            dependency = dep.node;
            dependency->successors.insert(successor);
          } else {
            remove = false;
          }
        }
      }

      if (remove) {
        successors.remove(successor);
        successor->updateHeight();
      }
    }

    if (dependency) {
      bool remove = true;
      for (uint i = 0; i < dependencies.size(); i++)
        if (dependencies[i].node == dependency) {
          // Remove index
          if (dependencies[i].qid == qid)
            dependencies.erase(dependencies.begin() + i);
          // We still depend on dependency for other QIDs
          else
            remove = false;
        }

      // Only remove this as a successor from dependency if this was the last
      // QID from dependency we depended on
      if (remove)
        dependency->successors.remove(this);
    }
  }

  /// Removes this dependency node from the graph by replacing all successor
  /// dependencies with the relevant dependency from this node. Also deletes
  /// this node and any classical values that only this node depends on.
  void erase() {
    for (auto successor : successors) {
      bool remove = true;
      for (auto &edge : successor->dependencies) {
        if (edge.node == this) {
          // If the output isn't a linear type, then don't worry about moving it
          if (quake::isQuantumType(edge.getValue().getType())) {
            auto dep = getDependencyForResult(edge.resultidx);
            edge = dep;
            dep->successors.insert(successor);
          } else {
            remove = false;
          }
        }
      }

      if (remove) {
        successors.remove(successor);
        successor->updateHeight();
      }
    }

    // Clean up any unused constants
    for (auto dependency : dependencies) {
      dependency->successors.remove(this);
      if (dependency->successors.empty() && !dependency->isQuantumDependent()) {
        static_cast<OpDependencyNode *>(dependency.node)->erase();
        delete dependency.node;
      }
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
      if (next->height > total_height)
        total_height = next->height;
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
      auto old_leaf = leafs[old_qid];

      auto [first_use, idx] = old_leaf->getSuccessorAndEdgeIDXForQID(old_qid);

      // TODO: use replaceWith
      first_use->dependencies[idx] =
          DependencyNode::DependencyEdge(new_leaf, 0);
      old_leaf->successors.remove(first_use);
      new_leaf->successors.insert(first_use);
      if (old_leaf->isAlloc()) {
        allocs.erase(allocs.find(old_qid));
        auto alloc = static_cast<InitDependencyNode *>(old_leaf);
        if (alloc->qubit)
          qubits.erase(qubits.find(alloc->qubit.value()));
      }
    }

    leafs[new_qid] = new_leaf;
    if (new_leaf->isAlloc()) {
      auto alloc = static_cast<InitDependencyNode *>(new_leaf);
      allocs[new_qid] = alloc;
      if (alloc->qubit)
        qubits[alloc->qubit.value()] = alloc;
    }
  }

  void replaceRoot(VirtualQID old_qid, VirtualQID new_qid,
                   DependencyNode *new_root) {
    assert(new_root->isRoot() && "Invalid root!");

    if (qids.contains(old_qid)) {
      auto last_use = getLastUseOfQID(old_qid);

      auto [old_root, idx] = last_use->getSuccessorAndEdgeIDXForQID(old_qid);

      new_root->dependencies.push_back(old_root->dependencies[idx]);
      old_root->dependencies.erase(old_root->dependencies.begin() + idx);

      last_use->successors.remove(old_root);
      last_use->successors.insert(new_root);

      // If the terminator is somehow getting deleted, then the entire block
      // must be empty, and then it will never be used
      if (old_root->dependencies.size() == 0)
        roots.remove(old_root);

      old_root->qids.remove(old_qid);
    }

    new_root->qids.insert(new_qid);
    // new_root->qids.insert(old_qid);
    roots.insert(new_root);
  }

  /// Gathers all the nodes in the graph into seen, starting from next
  void gatherNodes(SetVector<DependencyNode *> &seen, DependencyNode *next) {
    if (seen.contains(next) || !next->isQuantumDependent())
      return;

    seen.insert(next);

    for (auto successor : next->successors)
      gatherNodes(seen, successor);
    for (auto dependency : next->dependencies)
      gatherNodes(seen, dependency.node);
  }

public:
  DependencyGraph(DependencyNode *root) {
    total_height = 0;
    SetVector<DependencyNode *> seen;
    qids = SetVector<VirtualQID>();
    gatherRoots(seen, root);
  }

  ~DependencyGraph() {
    SetVector<DependencyNode *> nodes;
    for (auto root : roots)
      gatherNodes(nodes, root);

    for (auto node : nodes)
      // ArgDependencyNodes are handled by the block and skipped here
      if (!node->isLeaf() || !node->isQuantumOp() || node->isAlloc())
        delete node;
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
    DependencyNode *lastUse = nullptr;
    for (auto dependency : root->dependencies) {
      if (dependency.qid == qid) {
        lastUse = dependency.node;
        break;
      }
    }
    if (lastUse && lastUse->isLeaf())
      return nullptr;
    return static_cast<OpDependencyNode *>(lastUse);
  }

  OpDependencyNode *getFirstUseOfQubit(PhysicalQID qubit) {
    assert(qubits.count(qubit) == 1 && "Given qubit not in dependency graph");
    auto defining = qubits[qubit];
    // Qubit is defined here, return the first use
    if (defining->isAlloc()) {
      auto first_use = defining->successors.front();
      if (first_use->isRoot())
        return nullptr;
      return static_cast<OpDependencyNode *>(first_use);
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

  InitDependencyNode *getAllocForQubit(PhysicalQID qubit) {
    if (qubits.count(qubit) != 1 || !qubits[qubit]->isAlloc())
      return nullptr;
    return static_cast<InitDependencyNode *>(qubits[qubit]);
  }

  DependencyNode *getRootForQubit(PhysicalQID qubit) {
    for (auto root : roots)
      if (root->getQubits().contains(qubit))
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
    if (qubits.count(phys) != 1) {
      qubits[phys] = allocs[qid];
      allocs[qid]->assignToPhysical(phys);
      return;
    }

    assert(allocs.count(qid) == 1 && "Assigning a qid not in DependencyGraph!");
    auto new_lifetime = getLifeTimeForQID(qid);
    auto old_lifetime = getLifeTimeForQubit(phys);

    // TODO: can probably clean up a bit
    if (new_lifetime->isAfter(old_lifetime)) {
      auto new_alloc = getAllocForQID(qid);
      auto old_root = getRootForQubit(phys);

      auto [successor, idx] = new_alloc->getSuccessorAndEdgeIDXForQID(qid);

      // Replace new allocation with result value for old wire
      auto dep = old_root->dependencies[0];
      successor->dependencies[idx] = dep;
      dep->successors.insert(successor);
      dep->successors.remove(old_root);

      dep->updateQID(dep.qid.value(), new_alloc->getQID());

      roots.remove(old_root);
      delete old_root;
      allocs.erase(allocs.find(new_alloc->getQID()));
      delete new_alloc;

      successor->updateWithPhysical(qid, phys);
    } else {
      auto old_alloc = getAllocForQubit(phys);
      auto new_root = getRootForQID(qid);

      auto [successor, idx] =
          old_alloc->getSuccessorAndEdgeIDXForQID(old_alloc->getQID());

      auto dep = new_root->dependencies[0];
      successor->dependencies[idx] = dep;
      dep->successors.insert(successor);
      dep->successors.remove(new_root);

      dep->updateQID(old_alloc->getQID(), dep.qid.value());

      roots.remove(new_root);
      allocs.erase(allocs.find(old_alloc->getQID()));
      delete old_alloc;
      delete new_root;

      auto new_alloc = getAllocForQID(qid);
      new_alloc->assignToPhysical(phys);
      qubits[phys] = new_alloc;
    }
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
  /// Every node must be assigned a schedule greater than or equal to the height
  /// of each of its dependencies
  ///
  /// The current implementation of the scheduling algorithm can be found in
  /// DependencyGraph::schedule
  void schedulingPass() {
    SetVector<DependencyNode *> seen;
    // Schedule from the roots in order of height (starting from the tallest
    // root)
    auto sorted = SmallVector<DependencyNode *>({roots.begin(), roots.end()});
    std::sort(sorted.begin(), sorted.end(),
              [](auto x, auto y) { return x->getHeight() > y->getHeight(); });

    // Every node visiting during scheduling will be in seen, so
    // if the scheduling function has already visited the root then it will be
    // skipped
    for (auto root : sorted) {
      // Can either schedule starting with a level of `root->getHeight()`, which
      // will result in more operations at earlier cycles, or `total_height`,
      // which will result in more operations at later cycles
      schedule(seen, root, total_height);
    }
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
    SetVector<DependencyNode *> seen;
    for (auto root : roots) {
      updateHeight(seen, root);
      if (root->height > total_height)
        total_height = root->height;
    }
  }
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

  ~RootDependencyNode() override {}

  void eraseEdgeForQID(VirtualQID qid) override {
    if (qids.contains(qid))
      dependencies.clear();
  }

  SetVector<PhysicalQID> getQubits() override {
    SetVector<PhysicalQID> qubits;
    for (auto dependency : dependencies)
      if (dependency.qubit.has_value())
        qubits.insert(dependency.qubit.value());
    return qubits;
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

  ~ArgDependencyNode() override {}

  virtual std::string getOpName() override {
    return std::to_string(barg.getArgNumber()).append("arg");
  };

  void eraseEdgeForQID(VirtualQID qid) override {
    assert(false && "Can't call eraseEdgeForQID with an ArgDependencyNode");
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    assert(resultidx == 0 && "Invalid resultidx");
    if (qids.size() == 1)
      return std::optional(qids.front());
    return std::nullopt;
  }

  uint getArgNumber() { return argNum; }
};

class ShadowDependencyNode : public DependencyNode {
  friend class DependencyBlock;
  friend class IfDependencyNode;

protected:
  OpDependencyNode *shadowed;
  DependencyEdge shadow_edge;

  void printNode() override {
    llvm::outs() << "Shadow dependency on: ";
    shadowed->printNode();
  }

  bool isRoot() override { return false; }
  bool isLeaf() override { return true; }
  bool isQuantumOp() override { return false; }
  uint numTicks() override { return 0; }

  Value getResult(uint resultidx) override {
    return shadowed->getResult(resultidx);
  }

  ValueRange getResults() override { return shadowed->getResults(); }

  void codeGen(OpBuilder &builder, LifeTimeAnalysis &set) override {
    if (shadowed->hasCodeGen)
      hasCodeGen = true;
  };

public:
  ShadowDependencyNode(OpDependencyNode *shadowed, size_t resultidx)
      : shadowed(shadowed), shadow_edge(shadowed, resultidx) {}

  ~ShadowDependencyNode() override {}

  virtual std::string getOpName() override {
    return shadowed->getOpName().append("shadow");
  };

  void eraseEdgeForQID(VirtualQID qid) override {
    assert(false && "Can't call eraseEdgeForQID with an ShadowDependencyNode");
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    return std::nullopt;
  }

  DependencyEdge getShadowedEdge() { return shadow_edge; }
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

  ~TerminatorDependencyNode() override {}

  void genTerminator(OpBuilder &builder, LifeTimeAnalysis &set) {
    OpDependencyNode::codeGen(builder, set);
  }

  void eraseEdgeForQID(VirtualQID qid) override {
    for (uint i = 0; i < dependencies.size(); i++)
      if (dependencies[i].qid == qid)
        dependencies.erase(dependencies.begin() + i);
  }

  std::optional<VirtualQID> getQIDForResult(size_t resultidx) override {
    if (resultidx >= dependencies.size())
      return std::nullopt;
    return dependencies[resultidx].qid;
  }

  SetVector<PhysicalQID> getQubits() override {
    SetVector<PhysicalQID> qubits;
    for (auto dependency : dependencies)
      if (dependency.qubit.has_value())
        qubits.insert(dependency.qubit.value());
    return qubits;
  }
};

class DependencyBlock {
  friend class IfDependencyNode;

private:
  SmallVector<ArgDependencyNode *> argdnodes;
  DependencyGraph *graph;
  Block *block;
  TerminatorDependencyNode *terminator;
  SetVector<size_t> pqids;

public:
  DependencyBlock(SmallVector<ArgDependencyNode *> argdnodes,
                  DependencyGraph *graph, Block *block,
                  TerminatorDependencyNode *terminator)
      : argdnodes(argdnodes), graph(graph), block(block),
        terminator(terminator), pqids() {}

  ~DependencyBlock() {
    // Terminator is cleaned up by graph since it must be a root
    delete graph;
    // Arguments are not handled by the graph since they may not show up in the
    // graph
    for (auto argdnode : argdnodes)
      delete argdnode;
  }

  uint getHeight() { return graph->getHeight(); }

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

  InitDependencyNode *getAllocForQID(VirtualQID qid) {
    return graph->getAllocForQID(qid);
  }

  DependencyNode *getRootForQID(VirtualQID qid) {
    return graph->getRootForQID(qid);
  }

  InitDependencyNode *getAllocForQubit(PhysicalQID qubit) {
    return graph->getAllocForQubit(qubit);
  }

  DependencyNode *getRootForQubit(PhysicalQID qubit) {
    return graph->getRootForQubit(qubit);
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

    for (uint cycle = 0; cycle < graph->getHeight(); cycle++)
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

  void updateHeight() { graph->updateHeight(); }

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
      assert(first_use && "Unused virtual qubit in block!");
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
    graph->replaceLeafAndRoot(qid, init, root);
    removeArgument(qid);
    // If the qid isn't used in the block, remove it
    if (!graph->getFirstUseOfQID(qid)) {
      graph->removeVirtualAlloc(qid);
      graph->removeQID(qid);
    }
  }

  void liftAlloc(VirtualQID qid, DependencyNode *lifted_alloc) {
    auto new_edge = DependencyNode::DependencyEdge{lifted_alloc, 0};
    auto new_argdnode = addArgument(new_edge);

    graph->replaceLeafAndRoot(qid, new_argdnode, terminator);
  }

  void schedulingPass() { graph->schedulingPass(); }

  void removeQID(VirtualQID qid) {
    removeArgument(qid);

    terminator->eraseEdgeForQID(qid);
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
        delete argdnodes[i];
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
  SetVector<DependencyNode *> freevars;

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
      // Remove old edge from in this `if`
      then_op->eraseEdgeForQID(then_qid);

      // Add new edge from after this `if`
      auto resultidx = then_op->getResultForDependency(i);
      auto [successor, idx] = getSuccessorAndEdgeIDXForQID(then_qid);
      newDeps.push_back(successor->dependencies[idx]);
      successor->dependencies[idx] = DependencyEdge{then_op, resultidx};

      // Readd QID
      then_op->qids.insert(then_qid);
    }

    successors.insert(then_op);
    then_op->dependencies = newDeps;
    else_op->erase();
    delete else_op;
  }

  void liftAlloc(PhysicalQID qubit, DependencyGraph *parent) {
    InitDependencyNode *lifted_alloc = nullptr;
    DependencyNode *lifted_root = nullptr;

    // Remove virtual allocs from inner blocks
    if (then_block->getAllocatedQubits().contains(qubit)) {
      lifted_alloc = then_block->getAllocForQubit(qubit);
      lifted_root = then_block->getRootForQubit(qubit);
      then_block->liftAlloc(lifted_alloc->getQID(), lifted_alloc);
    }

    if (else_block->getAllocatedQubits().contains(qubit)) {
      lifted_alloc = else_block->getAllocForQubit(qubit);
      lifted_root = else_block->getRootForQubit(qubit);
      else_block->liftAlloc(lifted_alloc->getQID(), lifted_alloc);
    }

    assert(lifted_alloc && lifted_root && "Illegal qubit to lift!");

    if (!then_block->getQIDs().contains(lifted_alloc->getQID())) {
      auto new_arg = then_block->addArgument(DependencyEdge{lifted_alloc, 0});
      then_block->terminator->dependencies.push_back(
          DependencyEdge{new_arg, 0});
    }
    if (!else_block->getQIDs().contains(lifted_alloc->getQID())) {
      auto new_arg = else_block->addArgument(DependencyEdge{lifted_alloc, 0});
      else_block->terminator->dependencies.push_back(
          DependencyEdge{new_arg, 0});
    }

    // Add virtual alloc to current scope
    parent->replaceLeafAndRoot(lifted_alloc->getQID(), lifted_alloc,
                               lifted_root);
    qids.insert(lifted_alloc->getQID());
    // Hook lifted_root to the relevant result wire from this
    this->successors.insert(lifted_root);
    lifted_root->dependencies.push_back(DependencyEdge{this, results.size()});
    // Add a new result wire for the lifted wire which will flow to
    // lifted_root
    results.push_back(lifted_alloc->getResult(0).getType());
    // Hook lifted_alloc to then_op
    lifted_alloc->successors.insert(this);
    // Hook this to then_op by adding a new dependency for the lifted wire
    DependencyEdge newEdge(lifted_alloc, 0);
    dependencies.push_back(newEdge);
  }

  void liftOpBefore(OpDependencyNode *then_op, OpDependencyNode *else_op,
                    DependencyGraph *parent) {
    auto newDeps = SmallVector<DependencyEdge>();

    // Measure ops are a delicate special case because of the classical measure
    // result. When lifting before, we can lift the discriminate op as well,
    // but, the classical result is now free in the body of the if (assuming it
    // was used) so we must
    if (isa<RAW_MEASURE_OPS>(then_op->associated)) {
      auto then_discriminate = then_op->successors.front()->isQuantumOp()
                                   ? then_op->successors.back()
                                   : then_op->successors.front();
      auto else_discriminate = else_op->successors.front()->isQuantumOp()
                                   ? else_op->successors.back()
                                   : else_op->successors.front();
      auto casted = static_cast<OpDependencyNode *>(then_discriminate);
      auto newfreevar = new ShadowDependencyNode(casted, 0);
      auto newEdge = DependencyEdge{newfreevar, 0};
      then_discriminate->replaceWith(newEdge);
      else_discriminate->replaceWith(newEdge);
      dependencies.push_back(newEdge);
      freevars.insert(newfreevar);

      delete else_discriminate;
    }

    // Construct new dependencies for then_op based on the dependencies for this
    // `if`
    for (uint i = 0; i < then_op->dependencies.size(); i++) {
      auto dependency = then_op->dependencies[i];

      if (freevars.contains(dependency.node)) {
        // If the dependency is a free variable with this `if` as the frontier,
        // then we can just use the value directly, instead of the shadowed
        // value
        auto shadowNode = static_cast<ShadowDependencyNode *>(dependency.node);
        auto edge = shadowNode->getShadowedEdge();
        newDeps.push_back(edge);
        shadowNode->successors.remove(then_op);
        // Remove shadowNode if it is no longer needed
        if (shadowNode->successors.empty()) {
          for (uint i = 0; i < dependencies.size(); i++)
            if (dependencies[i].node == edge.node &&
                dependencies[i].resultidx == edge.resultidx)
              dependencies.erase(dependencies.begin() + i);
          freevars.remove(shadowNode);
          delete shadowNode;
        }
      } else if (dependency->isLeaf()) {
        // The dependency is a block argument, and therefore reflects a
        // dependency for this `if` First, find the relevant argument
        ArgDependencyNode *arg =
            static_cast<ArgDependencyNode *>(dependency.node);
        auto num = arg->getArgNumber();
        // Then, get the dependency from this `if` for the relevant argument,
        // this will be the new dependency for `then_op`
        auto newDep = dependencies[num + 1];
        newDep->successors.remove(this);
        newDep->successors.insert(then_op);
        newDeps.push_back(newDep);
        arg->successors.remove(then_op);

        // Replace the dependency with the relevant result from the lifted node
        dependencies[num + 1] =
            DependencyEdge{then_op, then_op->getResultForDependency(i)};

        // Remove then_op from the route for then_qid inside the block
        then_op->eraseEdgeForQID(dependency.qid.value());
      } else if (!dependency->isQuantumOp()) {
        newDeps.push_back(dependency);
      } else {
        assert(
            false &&
            "Trying to lift a quantum operation before dependency was lifted");
      }
    }

    else_op->erase();
    delete else_op;

    // Patch successors
    then_op->successors.insert(this);
    then_op->dependencies = newDeps;
  }

  void combineAllocs(SetVector<PhysicalQID> then_allocs,
                     SetVector<PhysicalQID> else_allocs, LifeTimeAnalysis &set,
                     DependencyGraph *parent) {
    SetVector<PhysicalQID> combined;
    combined.set_union(then_allocs);
    combined.set_union(else_allocs);

    for (auto qubit : combined)
      parent->addPhysicalAllocation(this, qubit);
  }

  void genOp(OpBuilder &builder, LifeTimeAnalysis &set) override {
    cudaq::cc::IfOp oldOp = dyn_cast<cudaq::cc::IfOp>(associated);

    auto operands = gatherOperands(builder, set);

    // Remove operands from shadow dependencies
    // First operand must be conditional, skip it
    for (uint i = 1; i < operands.size(); i++) {
      if (!quake::isQuantumType(operands[i].getType())) {
        operands.erase(operands.begin() + i);
        i--;
      }
    }

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

  DependencyEdge getDependencyForResult(size_t resultidx) override {
    auto qid = getQIDForResult(resultidx);
    assert(qid.has_value() &&
           "Cannot get dependency for classical result of if");
    for (auto dependency : dependencies)
      if (dependency.qid == qid.value())
        return dependency;

    assert(false && "Cannot find dependency for linear type result of if");
  }

public:
  IfDependencyNode(cudaq::cc::IfOp op,
                   SmallVector<DependencyEdge> _dependencies,
                   DependencyBlock *then_block, DependencyBlock *else_block,
                   SetVector<ShadowDependencyNode *> _freevars)
      : OpDependencyNode(op.getOperation(), _dependencies),
        then_block(then_block), else_block(else_block) {
    for (auto freevar : _freevars) {
      dependencies.push_back(freevar->getShadowedEdge());
      freevars.insert(freevar);
    }

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

  ~IfDependencyNode() override {
    delete then_block;
    delete else_block;
  }

  void contractAllocsPass() override {
    then_block->contractAllocsPass();
    else_block->contractAllocsPass();
  }

  void eraseEdgeForQID(VirtualQID qid) override {
    // First, calculate which result to remove, but don't remove it yet
    uint i = 0;
    for (; i < results.size(); i++)
      if (getQIDForResult(i) == qid)
        break;

    // Erase the actual edge with the blocks now set up properly
    this->OpDependencyNode::eraseEdgeForQID(qid);

    // Now, remove the QID from the blocks so that the blocks are set up
    // properly
    then_block->removeQID(qid);
    else_block->removeQID(qid);

    // Finally, remove the calculated result, which can no longer be calculated
    // because it was removed from the blocks
    results.erase(results.begin() + i);
  }

  bool tryLiftingBefore(OpDependencyNode *then_use, OpDependencyNode *else_use,
                        DependencyGraph *parent) {
    if (!then_use || !else_use)
      return false;

    if (then_use->prefixEquivalentTo(else_use)) {
      // If two nodes are equivalent, all their dependencies will be too,
      // but we can't lift them until all their dependencies have been lifted,
      // so we skip them for now.
      if (then_use->height > then_use->numTicks())
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
    for (auto qubit : getQubits()) {
      if (!then_block->getAllocatedQubits().contains(qubit) ||
          !else_block->getAllocatedQubits().contains(qubit))
        continue;
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
        if (!then_block->getQIDs().contains(qid) ||
            !else_block->getQIDs().contains(qid)) {
          unliftableQIDs.insert(qid);
          continue;
        }

        auto then_use = then_block->getFirstUseOfQID(qid);
        auto else_use = else_block->getFirstUseOfQID(qid);

        // QID is no longer reference in the if, erase it
        if (!then_use || !else_use) {
          if (!then_use && !else_use)
            eraseEdgeForQID(qid);
          unliftableQIDs.insert(qid);
          continue;
        }

        if (tryLiftingBefore(then_use, else_use, parent)) {
          lifted = true;
          run_more = true;
          continue;
        }

        then_use = then_block->getLastUseOfQID(qid);
        else_use = else_block->getLastUseOfQID(qid);

        if (tryLiftingAfter(then_use, else_use, parent)) {
          lifted = true;
          run_more = true;
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

    // Lift all physical allocations out of the if to respect the if
    auto allocs = then_block->getAllocatedQubits();
    allocs.set_union(else_block->getAllocatedQubits());
    for (auto qubit : allocs)
      liftAlloc(qubit, parent_graph);

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
    op->emitOpError("DependencyAnalysisPass: loops are not supported");
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

  if (hasEffect<mlir::MemoryEffects::Allocate>(op) && isQuakeOperation(op) &&
      !isa<quake::BorrowWireOp>(op)) {
    op->emitOpError("DependencyAnalysisPass: `quake.borrow_wire` is only "
                    "supported qubit allocation operation");
    return false;
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
  SmallVector<Operation *> ifStack;
  DenseMap<Operation *, SetVector<ShadowDependencyNode *>> freeClassicals;

public:
  DependencyAnalysisEngine()
      : perOp({}), argMap({}), ifStack({}), freeClassicals({}) {}

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

    DenseMap<DependencyNode *, Operation *> roots;
    TerminatorDependencyNode *terminator = nullptr;
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

    // In debug mode, alert about dead code wires
    // TODO: If an unused wire flows through an `if` with a useful wire,
    //       then the unused wire is considered useful as the parent context
    //       doesn't know that it doesn't interact with anything inside the if,
    //       it would be nice to have a "hasInteraction" predicate inside `if`s
    //       to be able to detect this case and do a better job of removing
    //       unused wires.
    //
    //       In fact, it may be possible to completely split `if`s into various
    //       non-interacting sub-graphs, which may make solving this problem easier,
    //       and may or may not present more optimization opportunities.
    LLVM_DEBUG(for (auto [root, op]
                    : roots) {
      if (!included.contains(root)) {
        llvm::dbgs()
            << "DependencyAnalysisPass: Wire is dead code and its "
            << "operations will be deleted (did you forget to return a value?)"
            << root << "\n";
      }
    });

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
      freeClassicals[op] = SetVector<ShadowDependencyNode *>();
      ifStack.push_back(op);
      auto then_block = visitBlock(ifop.getThenEntryBlock(), dependencies);
      auto else_block = visitBlock(ifop.getElseEntryBlock(), dependencies);
      if (!then_block || !else_block)
        return nullptr;
      ifStack.pop_back();

      SetVector<ShadowDependencyNode *> freeIn = freeClassicals[op];
      freeClassicals.erase(freeClassicals.find(op));

      newNode = new IfDependencyNode(ifop, dependencies, then_block, else_block,
                                     freeIn);
    } else if (isTerminator) {
      newNode = new TerminatorDependencyNode(op, dependencies);
    } else {
      newNode = new OpDependencyNode(op, dependencies);
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
    assert(defOp &&
           "Cannot handle value that is neither a BlockArgument nor OpResult");

    auto resultidx = dyn_cast<OpResult>(v).getResultNumber();

    // Since we walk forward through the ast, every value should be defined
    // before it is used, so we should have already visited defOp,
    // and thus should have a memoized dnode for defOp, fail if not
    assert(defOp->hasAttr("dnodeid") && "No dnodeid found for operation");

    auto id = defOp->getAttr("dnodeid").cast<IntegerAttr>().getUInt();
    auto dnode = perOp[id];

    if (!ifStack.empty() && defOp->getParentOp() != ifStack.back() &&
        dnode->isQuantumDependent()) {
      auto opdnode = static_cast<OpDependencyNode *>(dnode);
      auto shadow_node = new ShadowDependencyNode{opdnode, resultidx};

      auto parent = ifStack.back();

      while (parent->getParentOp() != defOp->getParentOp())
        parent = parent->getParentOp();

      freeClassicals[parent].insert(shadow_node);

      return DependencyNode::DependencyEdge{shadow_node, resultidx};
    }

    return DependencyNode::DependencyEdge{dnode, resultidx};
  }
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

        delete body;
        // Replace old block
        oldBlock->erase();
      }
    }
  }
};

} // namespace
