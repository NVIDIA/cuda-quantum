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
bool isBeginOp(Operation *op) {
  return dyn_cast<quake::UnwrapOp>(*op) || dyn_cast<quake::ExtractRefOp>(*op) ||
         dyn_cast<quake::NullWireOp>(*op);
}

bool isEndOp(Operation *op) {
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
  SetVector<DependencyNode *> successors;
  // Dependencies are in the order of operands of associated
  SmallVector<DependencyNode *> dependencies;
  // If a given dependency appears multiple times,
  // (e.g., multiple results of the dependency are used by associated),
  // it is important to know which result from the dependency
  // corresponds to which operand of associated.
  // Otherwise, the dependency will be code gen'ed first, and it will
  // be impossible to know (e.g., which result is a control and which is a
  // target). Result_idxs tracks this information, and therefore should be
  // exactly the same size/order as dependencies.
  SmallVector<size_t> result_idxs;
  SetVector<size_t> qids;
  Operation *associated;
  uint cycle = INT_MAX;
  bool hasCodeGen = false;
  uint height;
  bool quantumOp;
  bool isScheduled;

  void printNode() {
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
      dependency->printSubGraph(tabIndex + 1);
  }

  bool isRoot() { return isEndOp(associated); }

  bool isLeaf() { return isBeginOp(associated); }

  bool isSkip() { return isRoot() || isLeaf() || !quantumOp; }

  bool isQuantumDependent() { return qids.size() > 0; }

  uint numTicks() { return isSkip() ? 0 : 1; }

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
    if (!quantumOp && !isQuantumDependent())
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

  /// Returns the index of the result of the dependency corresponding to the
  /// \p operand_idx'th operand of this node
  size_t getResultIdx(size_t operand_idx) { return result_idxs[operand_idx]; }

  /// Recursively find nodes scheduled at a given cycle
  SetVector<DependencyNode *> getNodesAtCycle(uint _cycle, SetVector<DependencyNode *> &seen) {
    SetVector<DependencyNode *> nodes;

    if (cycle < _cycle || seen.contains(this))
      return nodes;
    else if (cycle == _cycle && !isSkip()) {
      nodes.insert(this);
      return nodes;
    }

    seen.insert(this);

    for (auto dependency : dependencies)
      nodes.set_union(dependency->getNodesAtCycle(_cycle, seen));

    return nodes;
  }

  /// Generates a new operation for this node in the dependency graph
  /// using the dependencies of the node as operands.
  void codeGen(OpBuilder &builder) {
    if (hasCodeGen || isRoot() || isLeaf())
      return;

    if (!quantumOp)
      for (auto dependency : dependencies)
        if (!dependency->hasCodeGen)
          return;

    auto oldOp = associated;
    SmallVector<mlir::Value> operands(oldOp->getNumOperands());

    for (size_t i = 0; i < dependencies.size(); i++) {
      auto dependency = dependencies[i];

      // Ensure classical values are available
      if (!dependency->quantumOp)
        dependency->codeGen(builder);

      assert(dependency->hasCodeGen &&
             "Generating code for successor before dependency");

      // Get relevant result from dependency's updated op
      // to use as the relevant operand
      auto result_idx = getResultIdx(i);
      operands[i] = dependency->associated->getResult(result_idx);
    }

    associated =
        Operation::create(oldOp->getLoc(), oldOp->getName(),
                          oldOp->getResultTypes(), operands, oldOp->getAttrs());
    associated->removeAttr("dnodeid");
    builder.insert(associated);
    hasCodeGen = true;

    for (auto successor : successors)
      // Ensure classical values are generated
      if (!successor->quantumOp)
        successor->codeGen(builder);
  }

  /// Replaces the null_wire op for \p qid with \p init
  void initializeWire(size_t qid, Operation *init, uint result_idx) {
    if (!qids.contains(qid))
      return;

    if (isLeaf()) {
      associated = init;
      hasCodeGen = true;

      // Update result_idxs of successors
      for (auto successor : successors)
        for (uint i = 0; i < successor->dependencies.size(); i++)
          if (successor->dependencies[i] == this)
            successor->result_idxs[i] = result_idx;

      return;
    }

    for (auto dependency : dependencies)
      dependency->initializeWire(qid, init, result_idx);
  }

public:
  DependencyNode(Operation *op, SmallVector<DependencyNode *> _dependencies)
      : dependencies(_dependencies), qids(), associated(op) {
    assert(op && "Cannot make dependency node for null op");
    assert(_dependencies.size() == op->getNumOperands() &&
           "Wrong # of dependencies to construct node");
    successors = SetVector<DependencyNode *>();
    result_idxs = SmallVector<size_t>(dependencies.size(), INT_MAX);

    if (isBeginOp(op) || isEndOp(op)) {
      // Should be ensured by assign-ids pass
      assert(op->hasAttr("qid") && "quake.null_wire or quake.sink missing qid");

      // Lookup qid
      auto qid = op->getAttrOfType<IntegerAttr>("qid").getUInt();
      qids.insert(qid);
    }

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
      auto results = dependency->associated->getResults();
      for (; result_idx < results.size(); result_idx++)
        if (results[result_idx] == operand)
          break;

      assert(result_idx < results.size() &&
             "Node passed as dependency isn't actually a dependency!");

      result_idxs[i] = result_idx;
      // Set relevant successor of dependency to this
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

  /// Assuming this is a root, replaces the old sink operation
  /// with a new one for the "physical" wire we replaced the virtual wire with
  void addCleanUp(OpBuilder &builder) {
    assert(isRoot() && "Can only call addCleanUp on a root node");
    auto last_use = dependencies[0];
    auto result_idx = getResultIdx(0);
    auto wire = last_use->associated->getResult(result_idx);
    auto newOp = builder.create<quake::SinkOp>(builder.getUnknownLoc(), wire);
    newOp->setAttrs(associated->getAttrs());
    associated->removeAttr("dnodeid");
    associated = newOp;
    hasCodeGen = true;
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

    if (isBeginOp(next->associated))
      firstUses.insert({next->qids.front(), next->successors.front()});

    for (auto successor : next->successors)
      gatherRoots(seen, successor);
    for (auto dependency : next->dependencies)
      gatherRoots(seen, dependency);
  }

  void scheduleNodes() { tallest->schedule(total_height); }

  SetVector<DependencyNode *> getNodesAtCycle(uint cycle) {
    SetVector<DependencyNode *> nodes;
    SetVector<DependencyNode *> seen;
    for (auto root : roots)
      nodes.set_union(root->getNodesAtCycle(cycle, seen));
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
    for (auto dependency : node->dependencies)
      validateNode(dependency, parent_cycle);
  }

public:
  DependencyGraph(DependencyNode *root) {
    total_height = 0;
    SetVector<DependencyNode *> seen;
    gatherRoots(seen, root);
    scheduleNodes();
  }

  SetVector<DependencyNode *> &getRoots() { return roots; }

  SetVector<size_t> &getQIDs() { return qids; }

  size_t getNumQIDs() { return qids.size(); }

  LifeTime *getLifeTimeForQID(size_t qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    uint first = getFirstUseOf(qid)->cycle;
    auto last = getLastUseOf(qid)->cycle;

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

  /// Creates a new physical null wire to replace the
  /// "virtual" qubit represented by \p qid
  void initializeWire(size_t qid, OpBuilder &builder) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    auto ctx = builder.getContext();
    auto wireTy = quake::WireType::get(ctx);
    auto initOp =
        builder.create<quake::NullWireOp>(builder.getUnknownLoc(), wireTy);
    getFirstUseOf(qid)->initializeWire(qid, initOp, 0);
  }

  /// Replaces the "virtual" qubit represented by \p qid with the same
  /// physical qubit as \p init, which is assumed to be the last use.
  void initializeWireFromRoot(size_t qid, DependencyNode *init) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
    assert(init && init->isRoot() &&
           "Can only initialize wire from a valid root");
    auto lastOp = init->dependencies[0]->associated;
    getFirstUseOf(qid)->initializeWire(qid, lastOp, init->result_idxs[0]);
  }

  void print() {
    llvm::outs() << "Graph Start\n";
    for (auto root : roots)
      root->print();
    llvm::outs() << "Graph End\n";
  }

  Location getIntroductionLoc(size_t qid) {
    assert(qids.contains(qid) && "Given qid not in dependency graph");
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
  size_t qubits;

  /// Validates that \p op meets the assumptions:
  /// * control flow operations are not allowed
  bool validateOp(Operation *op) {
    assert((!isQuakeOperation(op) || quake::isLinearValueForm(op) ||
      dyn_cast<quake::DiscriminateOp>(op)) && "DependencyAnalysisPass requires all quake operations to be in value form");

    if (op->getRegions().size() != 0) {
      op->emitOpError(
        "DependencyAnalysisPass cannot handle non-function operations with regions."
        " Do you have if statements in a Base Profile QIR program?");
      signalPassFailure();
      return false;
    }

    if (auto br = dyn_cast<mlir::BranchOpInterface>(op)) {
      br.emitOpError(
        "DependencyAnalysisPass cannot handle branching operations."
        " Do you have if statements in a Base Profile QIR program?");
      signalPassFailure();
      return false;
    }

    if (auto call = dyn_cast<mlir::CallOpInterface>(op)) {
      call.emitOpError("DependencyAnalysisPass does not support function calls");
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
    if (!func.getFunctionBody().hasOneBlock()) {
      func.emitError("DependencyAnalysisPass cannot handle multiple blocks. Do "
                     "you have if statements in a Base Profile QIR program?");
      signalPassFailure();
      return false;
    }

    // TODO: I think synthesis and inlining should cover this
    //       so it may make sense to turn into an assert
    if (func.getArguments().size() != 0) {
      func.emitError(
          "DependencyAnalysisPass cannot handle kernel arguments. "
          "Was quake synthesis run before this pass?");
      signalPassFailure();
      return false;
    }

    if (func.getNumResults() != 0) {
      func.emitError(
          "DependencyAnalysisPass cannot handle non-void return types for kernels");
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

    if (dumpNumQubits)
      llvm::dbgs() << "DependencyAnalysis used " << lifetimes.size() << " physical qubits\n";

    // Add teardown instructions
    for (auto sink : sinks)
      sink->addCleanUp(builder);
  }

  void runOnOperation() override {
    auto func = getOperation();
    // Ignore non-quantum functions
    if (!func->hasAttr("cudaq-kernel") || func.getBlocks().empty())
      return;

    if (!validateFunc(func))
      return;

    SetVector<DependencyNode *> roots;

    for (auto &op : func.front().getOperations()) {
      if (dyn_cast<func::ReturnOp>(op))
        continue;

      auto node = visitOp(&op);

      if (!node) {
        signalPassFailure();
        return;
      }

      if (isBeginOp(&op))
        qubits++;
      if (isEndOp(&op))
        roots.insert(node);
    }

    assert(qubits == roots.size() && "Too few sinks for qubits -- was add-dealloc run?");

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

struct ManageQubitsPipelineOptions
    : public PassPipelineOptions<ManageQubitsPipelineOptions> {
  PassOptions::Option<bool> runQubitManagement{
      *this, "run-qubit-management",
      llvm::cl::desc(
          "Runs qubit management pipeline. (default: true)"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> dumpNumQubits{
      *this, "dump-num-qubits",
      llvm::cl::desc(
          "Dumps the number of physical qubits used to STDERR. (default: false)"),
      llvm::cl::init(false)};
};
} // namespace


// TODO: ensure this is run only with BASE profile
static void createQubitManagementPipeline(OpPassManager &pm, bool runQubitManagement,
                                          bool dumpNumQubits) {
  if (!runQubitManagement)
    return;

  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandControlVeqs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createFactorQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuantumMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createAssignIDs());
  cudaq::opt::DependencyAnalysisOptions dao{dumpNumQubits};
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createDependencyAnalysis(dao));
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createRegToMem());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createDelayMeasurementsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void cudaq::opt::registerQubitManagementPipeline() {
  PassPipelineRegistration<ManageQubitsPipelineOptions>(
      "qubit-management-pipeline",
      "Map virtual qubits to physical qubits, minimizing the # of physical qubits.",
      [](OpPassManager &pm, const ManageQubitsPipelineOptions &upo) {
        createQubitManagementPipeline(pm, upo.runQubitManagement, upo.dumpNumQubits);
      });
}
