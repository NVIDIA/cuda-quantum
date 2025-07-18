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

bool processed(Operation *op);

void markProcessed(Operation *op);

// AXIS-SPECIFIC: could allow controlled y and z here
bool isControlledOp(Operation *op);

bool isCircuitBreaker(Operation *op);

class Netlist {
  Value def;
  SmallVector<Operation *> users;

public:
  Netlist(Value v) {
    assert(isa<quake::ExtractRefOp>(v.getDefiningOp()));
    def = v;
    users =
        SmallVector<Operation *>(def.getUsers().begin(), def.getUsers().end());
    // getUsers returns users in reverse order
    std::reverse(users.begin(), users.end());
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
};

class NetlistContainer {
  SmallVector<Netlist *> netlists;

public:
  NetlistContainer() {}

  ~NetlistContainer() {
    for (auto netlist : netlists)
      delete netlist;
  }

  void allocNetlist(quake::ExtractRefOp refop) {
    auto nlindex = netlists.size();
    refop->setAttr(
        "nlindex",
        mlir::IntegerAttr::get(mlir::IntegerType::get(refop->getContext(), 64),
                               nlindex));
    auto *nl = new Netlist(refop.getResult());
    netlists.push_back(nl);
  }

  size_t getIndexOf(Value ref) {
    auto refop = dyn_cast<quake::ExtractRefOp>(ref.getDefiningOp());
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

      if (subcircuit->isTerminationPoint(op) || processed(op))
        return false;

      subcircuit->ops.insert(op);

      auto opi = dyn_cast<quake::OperatorInterface>(op);

      if (opi.getControls().size() > 0) {
        if (opi.getControls().front() == nl->getDef())
          subcircuit->addAnchorPoint(opi.getTarget(0), op);
        else
          subcircuit->addAnchorPoint(opi.getControls().front(), op);
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
      if (start_point == 0)
        if (!processOp(start_point))
          start_point++;
    }

    void pruneFrom(size_t idx) {
      for (; idx < nl->size(); idx++) {
        auto op = nl->getOp(idx);
        if (isControlledOp(op)) {
          auto control = op->getOperand(0);
          auto target = op->getOperand(1);
          auto other_def = nl->getDef() == control ? target : control;
          auto nl = subcircuit->getWrapper(other_def);
          assert(nl);
          nl->pruneFrom(op);
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

    void prune() { pruneFrom(end_point); }

    Value getDef() { return nl->getDef(); }
  };

  NetlistContainer *container;
  SmallVector<NetlistWrapper *> qubits;
  SetVector<Operation *> ops;
  Operation *start;

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
  }

public:
  Subcircuit(Operation *cnot, NetlistContainer *container);
  ~Subcircuit();

  SetVector<Operation *> getOps();

  Operation *getStart();

  SmallVector<Value> getRefs();

  size_t numRefs();

  /// @brief Gets the operations in the subcircuit
  /// ordered by location
  SmallVector<Operation *> getOrderedOps();
};
