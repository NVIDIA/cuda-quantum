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
bool isControlledOp(Operation *op) {
  return isa<quake::XOp>(op) && op->getNumOperands() == 2;
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

  // Only allow single control
  if (opi.getControls().size() > 1)
    return true;
  return false;
}

inline bool isTwoQubitOp(Operation *op) {
  return quake::getQuantumOperands(op).size() == 2;
}

class Netlist {
  Value def;
  SmallVector<Operation *> users;

public:
  Netlist(Value v) {
    assert(isa<quake::RefType>(v.getType()));
    def = v;
  }

  size_t getIndexOf(Operation *op) {
    auto iter = std::find(users.begin(), users.end(), op);
    assert(iter != users.end());
    return std::distance(users.begin(), iter);
  }

  Operation *operator[](size_t index) { return users[index]; }

  Operation *getOp(size_t index) { return users[index]; }

  size_t size() { return users.size(); }

  Value getDef() { return def; }

  void append(Operation *op) { users.push_back(op); }
};

class NetlistContainer {
  SmallVector<Netlist *> netlists;

public:
  NetlistContainer(mlir::func::FuncOp func) {
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
          netlists[getIndexOf(operand)]->append(op);
    });
  }

  ~NetlistContainer() {
    for (auto netlist : netlists)
      delete netlist;
  }

  void allocNetlist(Operation *refop) {
    auto nlindex = netlists.size();
    refop->setAttr(
        "nlindex",
        mlir::IntegerAttr::get(mlir::IntegerType::get(refop->getContext(), 64),
                               nlindex));
    auto *nl = new Netlist(refop->getResult(0));
    netlists.push_back(nl);
  }

  size_t getIndexOf(Value ref) {
    auto refop = ref.getDefiningOp();
    assert(refop);
    if (!refop->hasAttr("nlindex"))
      allocNetlist(refop);
    auto nlindex = refop->getAttrOfType<IntegerAttr>("nlindex").getInt();
    return nlindex;
  }

  size_t size() { return netlists.size(); }

  Netlist *operator[](size_t index) { return netlists[index]; }

  Netlist *operator[](Value ref) { return netlists[getIndexOf(ref)]; }
};

class Subcircuit {
protected:
  SmallVector<std::pair<Value, Operation *>> anchor_points;

  void addAnchorPoint(Value qubit, Operation *op) {
    anchor_points.push_back({qubit, op});
  }

  bool isTerminationPoint(Operation *op) {
    return isCircuitBreaker(op) || (op->getBlock() != start->getBlock());
  }

  class NetlistWrapper {
    Subcircuit *subcircuit;
    Netlist *nl;
    // Inclusive
    size_t start_point;
    // Exclusive
    size_t end_point;

    size_t getIndexOf(Operation *op) { return nl->getIndexOf(op); }

    /// Returns `true` if processing should continue, `false` otherwise
    bool processOp(size_t op_idx) {
      auto op = nl->getOp(op_idx);

      // Currently, each operation can only be part of one subcircuit
      if (subcircuit->isTerminationPoint(op) || processed(op))
        return false;

      subcircuit->ops.insert(op);

      if (isTwoQubitOp(op)) {
        if (op->getOperand(0) == nl->getDef())
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
      if (!processOp(start_point))
        start_point++;
    }

    void pruneFrom(size_t idx) {
      for (; idx < nl->size(); idx++) {
        auto op = nl->getOp(idx);
        if (isTwoQubitOp(op)) {
          auto control = op->getOperand(0);
          auto target = op->getOperand(1);
          NetlistWrapper *otherWrapper = nullptr;
          if (nl->getDef() == control)
            otherWrapper = subcircuit->getWrapper(control);
          // If we are pruning along the target of a CNOT, we do not
          // need to prune along the control, as it will be unaffected
          else if (!isControlledOp(op))
            otherWrapper = subcircuit->getWrapper(target);

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
    NetlistWrapper(Subcircuit *subcircuit, Netlist *nl,
                   Operation *anchor_point) {
      this->nl = nl;
      this->subcircuit = subcircuit;
      processFrom(getIndexOf(anchor_point));
    }

    void addNewAnchorPoint(Operation *op) {
      auto index = getIndexOf(op);
      if (index >= start_point)
        return;
      processFrom(index);
    }

    Operation *getStart() {
      if (start_point == 0)
        return nl->getDef().getDefiningOp();
      return nl->getOp(start_point - 1);
    }

    Operation *getEnd() {
      if (end_point >= nl->size())
        return nullptr;
      return nl->getOp(end_point);
    }

    bool hasOps() { return end_point > start_point; }

    void prune() { pruneFrom(end_point); }

    Value getDef() { return nl->getDef(); }
  };

  NetlistContainer *container;
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
    auto *nlw = new NetlistWrapper(this, (*container)[nlindex], anchor_point);
    qubits[nlindex] = nlw;
  }

  NetlistWrapper *getWrapper(Value ref) {
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
    // Clean up
    for (size_t i = 0; i < qubits.size(); i++) {
      if (qubits[i] && !qubits[i]->hasOps()) {
        delete qubits[i];
        qubits[i] = nullptr;
      }
    }
  }

public:
  Subcircuit(Operation *cnot, NetlistContainer *container) {
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
  /// ordered by location
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
