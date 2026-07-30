/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_PHASEFOLDING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "phase-folding"

using namespace mlir;

// AXIS-SPECIFIC: Defines which operations break a circuit into subcircuits
#define CIRCUIT_BREAKERS(MACRO)                                                \
  MACRO(YOp), MACRO(ZOp), MACRO(HOp), MACRO(R1Op), MACRO(RxOp),                \
      MACRO(PhasedRxOp), MACRO(RyOp), MACRO(U2Op), MACRO(U3Op)
#define RAW(X) cudaq::quake::X
#define RAW_CIRCUIT_BREAKERS CIRCUIT_BREAKERS(RAW)

namespace {

// ============================================================================
// Phase algebra
// ============================================================================

struct PhaseVariable {
  size_t idx;
  PhaseVariable(size_t index) : idx(index) {}
  bool operator==(PhaseVariable other) { return idx == other.idx; }
};

using PhaseKey = std::pair<SmallVector<unsigned, 4>, bool>;

struct PhaseKeyInfo {
  static PhaseKey getEmptyKey() { return {{unsigned(-1)}, false}; }
  static PhaseKey getTombstoneKey() { return {{unsigned(-1)}, true}; }
  static unsigned getHashValue(const PhaseKey &k) {
    auto h = llvm::hash_value(k.second);
    for (auto idx : k.first)
      h = llvm::hash_combine(h, idx);
    return (unsigned)h;
  }
  static bool isEqual(const PhaseKey &a, const PhaseKey &b) { return a == b; }
};

/// A `Phase` is an exclusive sum of `PhaseVariable`s plus an inversion flag.
/// Two rotations on qubits with equal Phases can be merged into one rotation.
class Phase {
  SetVector<PhaseVariable *> vars;
  bool isInverted;

public:
  Phase() : isInverted(false) {}
  Phase(PhaseVariable *var) : isInverted(false) { vars.insert(var); }

  bool operator==(Phase other) {
    for (auto var : vars)
      if (!other.vars.contains(var))
        return false;
    for (auto var : other.vars)
      if (!vars.contains(var))
        return false;
    return isInverted == other.isInverted;
  }

  static Phase sum(const Phase &p1, const Phase &p2) {
    Phase p;
    for (auto var : p1.vars)
      p.vars.insert(var);
    for (auto var : p2.vars)
      if (p.vars.contains(var))
        p.vars.remove(var);
      else
        p.vars.insert(var);
    p.isInverted = (p1.isInverted != p2.isInverted);
    return p;
  }

  static Phase invert(const Phase &p1) {
    Phase p;
    p.vars.insert(p1.vars.begin(), p1.vars.end());
    p.isInverted = !p1.isInverted;
    return p;
  }

  PhaseKey toKey() const {
    SmallVector<unsigned, 4> indices;
    for (auto *var : vars)
      indices.push_back(var->idx);
    sort(indices);
    return {indices, isInverted};
  }
};

class PhaseStorage {
  DenseMap<PhaseKey, cudaq::quake::RzOp, PhaseKeyInfo> phaseToRot;
  size_t numCombined = 0;

  void combineRotations(cudaq::quake::RzOp old_rzop, cudaq::quake::RzOp rzop) {
    auto builder = OpBuilder(rzop);
    auto new_rot_arg = arith::AddFOp::create(
        builder, rzop.getLoc(), old_rzop.getOperand(0), rzop.getOperand(0));
    rzop->setOperand(0, new_rot_arg.getResult());
    // In wire semantics the erased rotation's result must be bypassed first.
    // Ref-form rotations have no results so this is skipped for them.
    if (old_rzop->getNumResults() > 0 &&
        isa<cudaq::quake::WireType>(old_rzop.getResult(0).getType()))
      old_rzop.getResult(0).replaceAllUsesWith(old_rzop.getOperand(1));
    old_rzop.erase();
    numCombined++;
  }

public:
  bool addOrCombineRotationForPhase(cudaq::quake::RzOp op, Phase phase) {
    auto key = phase.toKey();
    auto it = phaseToRot.find(key);
    if (it != phaseToRot.end()) {
      combineRotations(it->second, op);
      it->second = op;
      return true;
    }
    phaseToRot[key] = op;
    return false;
  }

  size_t getNumCombined() { return numCombined; }
};

// ============================================================================
// Reference semantics implementation
// ============================================================================

namespace ref {

// AXIS-SPECIFIC: could allow controlled y and z here
static bool isCNOT(Operation *op) {
  if (auto xop = dyn_cast<cudaq::quake::XOp>(op))
    return xop.getControls().size() == 1;
  return false;
}

/// Currently, only `!quake.ref`s generated directly from
/// `quake.alloca`s are supported. This is with the assumption that
/// the `factor-quantum-alloc` pass was run before, so any veqs, etc...
/// with variable indices are excluded to prevent side effects from
/// breaking a circuit without it being noticed. This does unfortunately
/// restrict the possible optimizations, so future work to recognize
/// these possible side effects could be beneficial.
static bool isSupportedValue(Value ref) {
  if (!isa<cudaq::quake::RefType>(ref.getType()))
    return false;

  if (!ref.getDefiningOp())
    return false;

  if (!ref.getDefiningOp<cudaq::quake::AllocaOp>())
    return false;

  // TODO: Concat op allows the pointer to be loaded again in a separate
  // ref. This aliasing means that we cannot reason about the operations
  // on ref just by looking at ref.getUsers(), which is problematic for
  // the phase folding algorithm. Currently, we handle this by simply
  // disregarding any refs that get concatenated (and possibly aliased).
  // We could eventually be a little smarter: we can probably reason
  // about a wire until it is aliased. We may even be able to trace the
  // aliases to resume reasoning after the alias is definitely no longer
  // used if it is relatively isolated.
  for (auto user : ref.getUsers())
    if (isa<cudaq::quake::ConcatOp>(user))
      return false;

  return true;
}

static bool isCircuitBreaker(Operation *op) {
  // TODO: it may be cleaner to only accept non-null input to
  // ensure the null case is explicitly handled by users
  if (!op)
    return true;

  if (!isQuakeOperation(op))
    return true;

  if (isa<RAW_CIRCUIT_BREAKERS, cudaq::quake::NullWireOp>(op))
    return true;

  auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);

  if (!opi)
    return true;

  // Only allow control in the case of CNOT
  if (opi.getControls().size() > 0 && !isCNOT(op))
    return true;

  // If any values are unsupported, the operation is also unsupported
  for (auto operand : cudaq::quake::getQuantumOperands(op))
    if (!isSupportedValue(operand))
      return true;

  return false;
}

inline bool isTwoQubitOp(Operation *op) {
  return cudaq::quake::getQuantumOperands(op).size() == 2;
}

/// A netlist representation of a circuit is a list of lists,
/// with each sublist holding the operations on a particular
/// qubit in order. Multi-qubit operations will appear in the
/// lists of each of their operands.
class Netlist {
  SmallVector<SmallVector<Operation *>> netlists;
  SmallPtrSet<Operation *, 8> processed;

public:
  Netlist(func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (auto allocaop = dyn_cast<cudaq::quake::AllocaOp>(op)) {
        if (isa<cudaq::quake::RefType>(allocaop.getType()))
          allocNetlist(allocaop);
        return;
      }

      for (auto operand : cudaq::quake::getQuantumOperands(op))
        if (isSupportedValue(operand))
          netlists[getIndexOf(operand)].push_back(op);
    });
  }

  void allocNetlist(Operation *refop) {
    auto nlindex = netlists.size();
    refop->setAttr(
        "nlindex",
        IntegerAttr::get(IntegerType::get(refop->getContext(), 64), nlindex));
    auto nl = SmallVector<Operation *>();
    netlists.push_back(nl);
  }

  size_t getIndexOf(Value ref) {
    assert(isSupportedValue(ref));
    auto refop = ref.getDefiningOp();
    if (!refop->hasAttr("nlindex"))
      allocNetlist(refop);
    auto nlindex = refop->getAttrOfType<IntegerAttr>("nlindex").getInt();
    return nlindex;
  }

  size_t size() { return netlists.size(); }

  SmallVector<Operation *> *getNetlist(size_t index) {
    return &netlists[index];
  }

  void markProcessed(Operation *op) { processed.insert(op); }

  bool wasProcessed(Operation *op) { return processed.contains(op); }
};

/// A subcircuit is a connected portion of the netlist containing
/// only RZ, NOT, CNOT, and Swap gates. Currently it only accepts
/// `quake.ref` types produced directly by `quake.alloca`, to avoid
/// possible issues with aliasing of `quake.veq`s.
class Subcircuit {
protected:
  SmallVector<std::pair<Value, Operation *>> anchor_points;
  Netlist *container = nullptr;

  void addAnchorPoint(Value qubit, Operation *op) {
    anchor_points.push_back({qubit, op});
  }

  bool isTerminationPoint(Operation *op) {
    // Currently, each operation can only be part of one subcircuit (hence the
    // check for the processed flag)
    return (op->getBlock() != start->getBlock()) || isCircuitBreaker(op) ||
           container->wasProcessed(op);
  }

  class NetlistWrapper {
    Subcircuit *subcircuit = nullptr;
    SmallVector<Operation *> *nl = nullptr;
    Value def;
    // Inclusive
    size_t start_point;
    // Exclusive
    size_t end_point;

    size_t getIndexOf(Operation *op) {
      auto iter = std::find(nl->begin(), nl->end(), op);
      assert(iter != nl->end());
      return std::distance(nl->begin(), iter);
    }

    bool processOp(size_t op_idx) {
      auto op = (*nl)[op_idx];
      if (subcircuit->isTerminationPoint(op))
        return false;
      subcircuit->ops.insert(op);
      if (isTwoQubitOp(op)) {
        if (op->getOperand(0) == def)
          subcircuit->addAnchorPoint(op->getOperand(1), op);
        else
          subcircuit->addAnchorPoint(op->getOperand(0), op);
      } else if (!isa<cudaq::quake::XOp>(op)) {
        // AXIS-SPECIFIC
        subcircuit->num_rot_gates++;
      }
      return true;
    }

    void processFrom(size_t index) {
      assert(index < nl->size());
      for (end_point = index + 1; end_point < nl->size(); end_point++)
        if (!processOp(end_point))
          break;
      for (start_point = index; start_point > 0; start_point--)
        if (!processOp(start_point))
          break;
      // Handle 0th element separately to prevent underflow
      if (!processOp(start_point))
        start_point++;
    }

    void pruneFrom(size_t idx) {
      for (; idx < nl->size(); idx++) {
        auto op = (*nl)[idx];
        if (isTwoQubitOp(op)) {
          auto control = op->getOperand(0);
          auto target = op->getOperand(1);
          NetlistWrapper *otherWrapper = nullptr;
          if (def == control)
            otherWrapper = subcircuit->getWrapper(target);
          // If pruning along the target of a CNOT, no need to prune control
          else if (!isCNOT(op))
            otherWrapper = subcircuit->getWrapper(control);
          if (otherWrapper)
            otherWrapper->pruneFrom(op);
        } else if (isa<cudaq::quake::RzOp>(op) &&
                   subcircuit->ops.contains(op)) {
          // AXIS-SPECIFIC
          subcircuit->num_rot_gates--;
        }
        subcircuit->ops.remove(op);
      }
    }

    void pruneFrom(Operation *op) {
      auto index = getIndexOf(op);
      if (index >= end_point)
        return;
      end_point = index;
      pruneFrom(index);
    }

  public:
    NetlistWrapper(Subcircuit *subcircuit, SmallVector<Operation *> *nl,
                   Operation *anchor_point, Value def)
        : subcircuit(subcircuit), nl(nl), def(def) {
      processFrom(getIndexOf(anchor_point));
    }

    void addNewAnchorPoint(Operation *op) {
      auto index = getIndexOf(op);
      if (index >= start_point)
        return;
      processFrom(index);
    }

    bool hasOps() { return end_point > start_point; }
    void prune() { pruneFrom(end_point); }
    Value getDef() { return def; }
  };

  SmallVector<NetlistWrapper *> qubits = {};
  SetVector<Operation *> ops = {};
  SmallVector<Operation *> ordered_ops = {};
  Operation *start = nullptr;
  size_t num_rot_gates = 0;

  void allocWrapper(Value ref, Operation *anchor_point) {
    auto nlindex = container->getIndexOf(ref);
    if (nlindex >= qubits.size())
      for (auto i = qubits.size(); i < container->size(); i++)
        qubits.push_back(nullptr);
    qubits[nlindex] = new NetlistWrapper(this, container->getNetlist(nlindex),
                                         anchor_point, ref);
  }

  NetlistWrapper *getWrapper(Value ref) {
    if (!isSupportedValue(ref))
      return nullptr;
    auto nlindex = container->getIndexOf(ref);
    return qubits[nlindex];
  }

  void processNextAnchorPoint() {
    auto next = anchor_points.back();
    anchor_points.pop_back();
    auto nl = getWrapper(next.first);
    if (nl)
      nl->addNewAnchorPoint(next.second);
    else
      allocWrapper(next.first, next.second);
  }

  void calculateInitialSubcircuit() {
    auto control = start->getOperand(0);
    auto target = start->getOperand(1);
    addAnchorPoint(control, start);
    addAnchorPoint(target, start);
    while (!anchor_points.empty())
      processNextAnchorPoint();
  }

  void pruneSubcircuit() {
    for (auto *netlist : qubits)
      if (netlist)
        netlist->prune();
    for (size_t i = 0; i < qubits.size(); i++) {
      if (qubits[i] && !qubits[i]->hasOps()) {
        delete qubits[i];
        qubits[i] = nullptr;
      }
    }
  }

public:
  Subcircuit(Operation *cnot, Netlist *netlist)
      : container(netlist), start(cnot) {
    assert(isCNOT(cnot));
    qubits = SmallVector<NetlistWrapper *>(netlist->size(), nullptr);
    calculateInitialSubcircuit();
    pruneSubcircuit();
    for (auto op : ops)
      netlist->markProcessed(op);
  }

  ~Subcircuit() {
    for (auto wrapper : qubits)
      if (wrapper)
        delete wrapper;
  }

  SmallVector<Value> getRefs() {
    SmallVector<Value> refs;
    for (auto wrapper : qubits)
      if (wrapper)
        refs.push_back(wrapper->getDef());
    return refs;
  }

  SmallVector<Operation *> getOrderedOps() {
    if (ordered_ops.size() == 0 && ops.size() > 0) {
      ordered_ops = SmallVector<Operation *>(ops.begin(), ops.end());
      auto less = [&](Operation *a, Operation *b) {
        return a->isBeforeInBlock(b);
      };
      std::sort(ordered_ops.begin(), ordered_ops.end(), less);
    }

    return ordered_ops;
  }

  size_t getNumRotations() { return num_rot_gates; }

  float getRotationWeight() {
    return (float)getNumRotations() / (float)getNumOps();
  }

  size_t getNumOps() { return ops.size(); }
};

} // namespace ref

// ============================================================================
// Wire semantics implementation
// ============================================================================

namespace wire {

static unsigned calculateSkip(Operation *op) {
  unsigned i = 0;
  for (auto type : op->getOperandTypes()) {
    if (isa<cudaq::quake::WireType>(type))
      return i;
    i++;
  }
  return i;
}

static Value getNextOperand(Value v) {
  auto result = dyn_cast<OpResult>(v);
  auto op = result.getDefiningOp();
  auto skip = calculateSkip(op);
  return op->getOperand(result.getResultNumber() + skip);
}

static OpResult getNextResult(Value v) {
  assert(v.hasOneUse());
  auto correspondingOperand = v.getUses().begin();
  auto op = correspondingOperand.getUser();
  auto skip = calculateSkip(op);
  return op->getResult(correspondingOperand.getOperand()->getOperandNumber() -
                       skip);
}

// AXIS-SPECIFIC: could allow controlled y and z here
static bool isControlledOp(Operation *op) {
  return isa<cudaq::quake::XOp>(op) && op->getNumOperands() == 2;
}

static bool isTerminationPoint(Operation *op) {
  if (!op)
    return true;
  if (!isQuakeOperation(op))
    return true;
  if (isa<RAW_CIRCUIT_BREAKERS>(op))
    return true;
  if (isa<cudaq::quake::NullWireOp>(op))
    return true;
  auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
  if (!opi)
    return true;
  // Only allow single control
  if (opi.getControls().size() > 1)
    return true;
  return false;
}

class Subcircuit {
protected:
  SetVector<Operation *> ops;
  SetVector<Value> initial_wires;
  SetVector<Value> terminal_wires;
  Operation *start;
  // TODO: these three are really intermediate state for constructing the
  // subcircuit; would be nice to turn them into local arguments instead
  SetVector<Value> termination_points;
  SetVector<Value> anchor_points;
  SetVector<Value> seen;

  bool isAfterTerminationPoint(Value wire) {
    return isTerminationPoint(wire.getDefiningOp());
  }

  void addAnchorPoint(Value v) { anchor_points.insert(v); }
  void addTerminationPoint(Value v) { termination_points.insert(v); }

  void calculateSubcircuitForQubitForward(OpResult v) {
    if (seen.contains(v))
      return;
    seen.insert(v);
    if (!v.hasOneUse()) {
      addTerminationPoint(v);
      return;
    }
    Operation *op = v.getUses().begin().getUser();
    if (isTerminationPoint(op)) {
      addTerminationPoint(v);
      return;
    }
    ops.insert(op);
    auto nextResult = getNextResult(v);
    // Controlled not: add an anchor point for the other qubit
    if (op->getResults().size() > 1) {
      auto control = op->getResult(0);
      auto target = op->getResult(1);
      if (nextResult == control)
        addAnchorPoint(target);
      else
        addAnchorPoint(control);
    }
    calculateSubcircuitForQubitForward(nextResult);
  }

  void calculateSubcircuitForQubitBackward(Value v) {
    if (seen.contains(v))
      return;
    seen.insert(v);
    Operation *op = v.getDefiningOp();
    if (isTerminationPoint(op)) {
      addTerminationPoint(v);
      return;
    }
    ops.insert(op);
    auto nextOperand = getNextOperand(v);
    // Controlled not: add an anchor point for the other qubit
    // Use getResults() as Rz has two operands but only one result
    if (op->getResults().size() > 1) {
      auto control = op->getOperand(0);
      auto target = op->getOperand(1);
      if (nextOperand == control)
        addAnchorPoint(target);
      else
        addAnchorPoint(control);
    }
    calculateSubcircuitForQubitBackward(nextOperand);
  }

  void calculateInitialSubcircuit(Operation *op) {
    // AXIS-SPECIFIC: This could be any controlled operation
    auto cnot = dyn_cast<cudaq::quake::XOp>(op);
    assert(cnot && cnot.getWires().size() == 2);
    ops.insert(cnot);
    anchor_points.insert(cnot->getResult(1));
    calculateSubcircuitForQubitForward(cnot->getResult(0));
    calculateSubcircuitForQubitBackward(cnot->getOperand(0));
    while (!anchor_points.empty()) {
      auto next = anchor_points.back();
      anchor_points.pop_back();
      if (seen.contains(next))
        continue;
      calculateSubcircuitForQubitForward(dyn_cast<OpResult>(next));
      // Remove next from seen for working backwards
      seen.remove(next);
      calculateSubcircuitForQubitBackward(next);
    }
  }

public:
  Subcircuit(Operation *cnot, DenseSet<Operation *> &processedOps) {
    calculateInitialSubcircuit(cnot);
    // TODO: there is a performance issue that the current pruning definition
    // will always preference earlier operations, so a large interconnected
    // circuit will always tend towards the same subcircuit early in the circuit
    // and will not process later CNOTs, so the same early subcircuit will be
    // processed repeatedly. For value semantics, we don't actually need to
    // prune because we can just assign fresh phase variables for wires after
    // circuit breaking ops. Another possible option is to stop at ops in
    // `processedOps`. This is likely also an issue in the ref semantics form.
    for (auto *op : ops)
      processedOps.insert(op);
    start = cnot;
    for (auto w : termination_points)
      if (isAfterTerminationPoint(w))
        initial_wires.insert(w);
      else
        terminal_wires.insert(w);
  }

  SetVector<Value> getInitialWires() { return initial_wires; }
  bool isInSubcircuit(Operation *op) { return ops.contains(op); }
  size_t getNumOps() { return ops.size(); }

  float getRotationWeight() {
    if (ops.empty())
      return 0.0f;
    size_t rzCount = 0;
    for (auto *op : ops)
      if (isa<cudaq::quake::RzOp>(op))
        rzCount++;
    return (float)rzCount / (float)ops.size();
  }

  SmallVector<Operation *> getOrderedOps() {
    SmallVector<Operation *> ordered(ops.begin(), ops.end());
    sort(ordered,
         [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    return ordered;
  }
};

} // namespace wire

// ============================================================================
// Pass
// ============================================================================

class PhaseFoldingPass
    : public cudaq::opt::impl::PhaseFoldingBase<PhaseFoldingPass> {
  using PhaseFoldingBase::PhaseFoldingBase;

  void doPhaseFolding(ref::Subcircuit *subcircuit) {
    SmallVector<PhaseVariable *> phase_vars;
    SmallVector<Phase> current_phases;
    PhaseStorage store;
    size_t i = 0;
    for (auto refVal : subcircuit->getRefs()) {
      auto phase_idx = i++;
      auto *new_phase_var = new PhaseVariable(phase_idx);
      auto *defop = refVal.getDefiningOp();
      assert(defop);
      defop->setAttr("phaseidx",
                     OpBuilder(defop).getUI32IntegerAttr(phase_idx));
      phase_vars.push_back(new_phase_var);
      current_phases.push_back(Phase(new_phase_var));
    }

    auto getPhase = [&](Value ref) {
      auto idx =
          ref.getDefiningOp()->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      return current_phases[idx];
    };
    auto setPhase = [&](Value ref, Phase phase) {
      auto idx =
          ref.getDefiningOp()->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      current_phases[idx] = phase;
    };

    for (auto op : subcircuit->getOrderedOps()) {
      if (ref::isCNOT(op)) {
        auto control = op->getOperand(0);
        auto target = op->getOperand(1);
        setPhase(target, Phase::sum(getPhase(target), getPhase(control)));
      } else if (isa<cudaq::quake::XOp>(op)) {
        // AXIS-SPECIFIC: Would want to handle y and z gates here too
        auto target = op->getOperand(0);
        setPhase(target, Phase::invert(getPhase(target)));
      } else if (auto rzop = dyn_cast<cudaq::quake::RzOp>(op)) {
        store.addOrCombineRotationForPhase(rzop, getPhase(op->getOperand(1)));
      } else if (auto swap = dyn_cast<cudaq::quake::SwapOp>(op)) {
        auto t1 = op->getOperand(0);
        auto t2 = op->getOperand(1);
        auto p1 = getPhase(t1);
        setPhase(t1, getPhase(t2));
        setPhase(t2, p1);
      }
    }

    for (auto *pv : phase_vars)
      delete pv;
  }

  void runRefSemantics() {
    auto func = getOperation();
    ref::Netlist nl(func);
    SmallVector<ref::Subcircuit *> subcircuits;

    func.walk([&](cudaq::quake::XOp op) {
      // AXIS-SPECIFIC: controlled not only
      if (!ref::isCNOT(op) || nl.wasProcessed(op))
        return;
      if (!ref::isSupportedValue(op.getOperand(0)) ||
          !ref::isSupportedValue(op.getOperand(1)))
        return;
      auto *subcircuit = new ref::Subcircuit(op, &nl);
      if (subcircuit->getNumOps() < minimumBlockLength ||
          subcircuit->getRotationWeight() < minimumrzWeight) {
        LLVM_DEBUG(llvm::dbgs() << "Subcircuit below threshold, skipping!\n");
        delete subcircuit;
        return;
      }
      subcircuits.push_back(subcircuit);
    });

    // Collect then optimize to avoid rewriting the IR during the walk
    for (auto *subcircuit : subcircuits) {
      doPhaseFolding(subcircuit);
      delete subcircuit;
    }
  }

  void doWirePhaseFolding(wire::Subcircuit *subcircuit) {
    DenseMap<Value, Phase> wirePhase;
    SmallVector<std::unique_ptr<PhaseVariable>> vars;
    PhaseStorage store;
    size_t i = 0;

    for (auto w : subcircuit->getInitialWires()) {
      auto &var = vars.emplace_back(std::make_unique<PhaseVariable>(i++));
      wirePhase[w] = Phase(var.get());
    }

    for (auto *op : subcircuit->getOrderedOps()) {
      if (wire::isControlledOp(op)) {
        auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
        Phase ctrlPhase = wirePhase[opi.getControls().front()];
        Phase tgtPhase = wirePhase[opi.getTarget(0)];
        wirePhase[op->getResult(0)] = ctrlPhase;
        wirePhase[op->getResult(1)] = Phase::sum(ctrlPhase, tgtPhase);
      } else if (isa<cudaq::quake::XOp>(op)) {
        // AXIS-SPECIFIC: Would want to handle y and z gates here too
        auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
        wirePhase[op->getResult(0)] =
            Phase::invert(wirePhase[opi.getTarget(0)]);
      } else if (auto rzop = dyn_cast<cudaq::quake::RzOp>(op)) {
        Phase p = wirePhase[op->getOperand(1)]; // operand 0 is the angle
        store.addOrCombineRotationForPhase(rzop, p);
        wirePhase[op->getResult(0)] = p;
      } else if (auto swap = dyn_cast<cudaq::quake::SwapOp>(op)) {
        Phase p0 = wirePhase[swap.getTarget(0)];
        Phase p1 = wirePhase[swap.getTarget(1)];
        wirePhase[op->getResult(0)] = p1;
        wirePhase[op->getResult(1)] = p0;
      }
    }
  }

  void runWireSemantics() {
    auto func = dyn_cast<func::FuncOp>(getOperation());
    if (!func)
      return;

    // Subcircuits are built and folded one at a time so that Rz ops erased
    // during folding are gone from the IR before the next subcircuit is built.
    // TODO: Parallel folding would require tracking which Rz ops have been
    // erased so that subcircuits built concurrently can skip stale references.
    DenseSet<Operation *> processedOps;
    func.walk([&](cudaq::quake::XOp xop) {
      if (!wire::isControlledOp(xop) || processedOps.count(xop))
        return;
      wire::Subcircuit subcircuit(xop, processedOps);
      if (subcircuit.getNumOps() < minimumBlockLength ||
          subcircuit.getRotationWeight() < minimumrzWeight) {
        LLVM_DEBUG(llvm::dbgs() << "Subcircuit below threshold, skipping!\n");
        return;
      }
      doWirePhaseFolding(&subcircuit);
    });
  }

public:
  void runOnOperation() override {
    if (useWireSemantics)
      runWireSemantics();
    else
      runRefSemantics();
  }
};

/// Phase folding pass pipeline command-line options.
struct PhaseFoldingPipelineOptions
    : public PassPipelineOptions<PhaseFoldingPipelineOptions> {
  PassOptions::Option<unsigned> minimumBlockLength{
      *this, "min-length",
      llvm::cl::desc(
          "Minimum subcircuit length to run phase folding (ref mode only)."),
      llvm::cl::init(20)};
  PassOptions::Option<double> minimumrzWeight{
      *this, "min-rz-weight",
      llvm::cl::desc("Minimum rz percentage to run phase folding "
                     "(ref mode only)."),
      llvm::cl::init(0.2)};
  PassOptions::Option<bool> useWireSemantics{
      *this, "use-wire-semantics",
      llvm::cl::desc("Use wire (value) semantics instead of reference "
                     "semantics."),
      llvm::cl::init(false)};
};
} // namespace

static void createPhaseFoldingPipeline(OpPassManager &pm, bool wireSemantics,
                                       unsigned min_length,
                                       double min_rz_weight) {
  pm.addNestedPass<func::FuncOp>(
      cudaq::opt::createFactorQuantumAllocations({.enableFailures = true}));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createDeadQuantumElimination());
  cudaq::opt::PhaseFoldingOptions pfo{min_length, min_rz_weight, wireSemantics};
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createPhaseFolding(pfo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
}

void cudaq::opt::registerPhaseFoldingPipeline() {
  PassPipelineRegistration<PhaseFoldingPipelineOptions>(
      "phase-folding-pipeline",
      "Performs the phase-polynomial based rotation merging optimization.",
      [](OpPassManager &pm, const PhaseFoldingPipelineOptions &pfpo) {
        createPhaseFoldingPipeline(pm, pfpo.useWireSemantics,
                                   pfpo.minimumBlockLength,
                                   pfpo.minimumrzWeight);
      });
}
