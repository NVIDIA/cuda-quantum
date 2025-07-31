#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

using namespace mlir;

#define RAW(X) quake::X
#define RAW_MEASURE_OPS MEASURE_OPS(RAW)
#define RAW_GATE_OPS GATE_OPS(RAW)
#define RAW_QUANTUM_OPS QUANTUM_OPS(RAW)
// AXIS-SPECIFIC: Defines which operations break a circuit into subcircuits
#define CIRCUIT_BREAKERS(MACRO)                                                \
  MACRO(YOp), MACRO(ZOp), MACRO(HOp), MACRO(R1Op), MACRO(RxOp),                \
      MACRO(PhasedRxOp), MACRO(RyOp), MACRO(U2Op), MACRO(U3Op)
#define RAW_CIRCUIT_BREAKERS CIRCUIT_BREAKERS(RAW)

bool processed(Operation *op) { return op->hasAttr("processed"); }

void markProcessed(Operation *op) {
  op->setAttr("processed", OpBuilder(op).getUnitAttr());
}

// AXIS-SPECIFIC: could allow controlled y and z here
bool isCNOT(Operation *op) {
  return isa<quake::XOp>(op) && op->getNumOperands() == 2;
}

/// Currently, only `!quake.ref`s that are not block arguments are supported
bool isSupportedValue(Value ref) {
  return isa<quake::RefType>(ref.getType()) && ref.getDefiningOp();
}

bool isCircuitBreaker(Operation *op) {
  // TODO: it may be cleaner to only accept non-null input to
  // ensure the null case is explicitly handled by users
  if (!op)
    return true;

  if (!isQuakeOperation(op))
    return true;

  if (isa<RAW_CIRCUIT_BREAKERS>(op))
    return true;

  if (isa<quake::NullWireOp>(op))
    return true;

  auto opi = dyn_cast<quake::OperatorInterface>(op);

  if (!opi)
    return true;

  // Only allow control in the case of CNOT
  if (opi.getControls().size() > 0 && !isCNOT(op))
    return true;

  // If any values are unsupported, the operation is also unsupported
  for (auto operand : quake::getQuantumOperands(op))
    if (!isSupportedValue(operand))
      return true;

  return false;
}

inline bool isTwoQubitOp(Operation *op) {
  return quake::getQuantumOperands(op).size() == 2;
}

class Netlist {
  SmallVector<SmallVector<Operation *>> netlists;

public:
  Netlist(mlir::func::FuncOp func) {
    func.walk([&](Operation *op) {
      if (auto refop = dyn_cast<quake::ExtractRefOp>(op)) {
        allocNetlist(refop);
        return;
      } else if (auto allocaop = dyn_cast<quake::AllocaOp>(op)) {
        if (isa<quake::RefType>(allocaop.getType()))
          allocNetlist(allocaop);
        return;
      }

      if (isa<quake::OperatorInterface>(op))
        for (auto operand : quake::getQuantumOperands(op))
          if (isSupportedValue(operand))
            netlists[getIndexOf(operand)].push_back(op);
    });
  }

  void allocNetlist(Operation *refop) {
    auto nlindex = netlists.size();
    refop->setAttr(
        "nlindex",
        mlir::IntegerAttr::get(mlir::IntegerType::get(refop->getContext(), 64),
                               nlindex));
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
};

class Subcircuit {
protected:
  SmallVector<std::pair<Value, Operation *>> anchor_points;

  void addAnchorPoint(Value qubit, Operation *op) {
    anchor_points.push_back({qubit, op});
  }

  bool isTerminationPoint(Operation *op) {
    // Currently, each operation can only be part of one subcircuit (hence the
    // check for the processed flag)
    return (op->getBlock() != start->getBlock()) || processed(op) ||
           isCircuitBreaker(op);
  }

  class NetlistWrapper {
    Subcircuit *subcircuit;
    SmallVector<Operation *> *nl;
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

    /// Returns `true` if processing should continue, `false` otherwise
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
      } else if (!isa<quake::XOp>(op)) {
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

      // Handle possible 0th element separately to prevent overflow
      // This is why start_point must be inclusive
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
          // If we are pruning along the target of a CNot, we do not
          // need to prune along the control, as it will be unaffected
          else if (!isCNOT(op))
            otherWrapper = subcircuit->getWrapper(control);

          if (otherWrapper)
            otherWrapper->pruneFrom(op);
        } else if (isa<quake::RzOp>(op) && subcircuit->ops.contains(op)) {
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
                   Operation *anchor_point, Value def) {
      this->nl = nl;
      this->subcircuit = subcircuit;
      this->def = def;
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

  Netlist *container;
  SmallVector<NetlistWrapper *> qubits;
  SetVector<Operation *> ops;
  SmallVector<Operation *> ordered_ops;
  Operation *start;
  size_t num_rot_gates = 0;

  void allocWrapper(Value ref, Operation *anchor_point) {
    auto nlindex = container->getIndexOf(ref);
    if (nlindex >= qubits.size())
      for (auto i = qubits.size(); i < container->size(); i++)
        qubits.push_back(nullptr);
    auto *nlw = new NetlistWrapper(this, container->getNetlist(nlindex),
                                   anchor_point, ref);
    qubits[nlindex] = nlw;
  }

  /// @brief Gets the NetlistWrapper for ref, if it exists
  /// @returns The NetlistWrapper for the Netlist for ref or
  /// `nullptr` if no such Netlist exists
  NetlistWrapper *getWrapper(Value ref) {
    if (!isSupportedValue(ref))
      return nullptr;

    auto nlindex = container->getIndexOf(ref);
    // Can still be nullptr if the wrapper hasn't been initialized
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
    // Clean up
    for (size_t i = 0; i < qubits.size(); i++) {
      if (qubits[i] && !qubits[i]->hasOps()) {
        delete qubits[i];
        qubits[i] = nullptr;
      }
    }
  }

public:
  Subcircuit(Operation *cnot, Netlist *container) {
    start = cnot;
    this->container = container;
    qubits = SmallVector<NetlistWrapper *>(container->size(), nullptr);
    calculateInitialSubcircuit();
    pruneSubcircuit();
    for (auto op : ops)
      markProcessed(op);
  }

  ~Subcircuit() {
    for (auto wrapper : qubits)
      if (wrapper)
        delete wrapper;
  }

  Operation *getStart() { return start; }

  SmallVector<Value> getRefs() {
    SmallVector<Value> refs;
    for (auto wrapper : qubits)
      if (wrapper)
        refs.push_back(wrapper->getDef());

    return refs;
  }

  size_t numRefs() {
    size_t count = 0;
    for (auto wrapper : qubits)
      if (wrapper)
        count++;
    return count;
  }

  /// @brief Gets the operations in the subcircuit
  /// ordered by location in the containing block
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
};
