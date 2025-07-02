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

unsigned calculateSkip(Operation *op);

Value getNextOperand(Value v);

OpResult getNextResult(Value v);

bool processed(Operation *op);

void markProcessed(Operation *op);

// AXIS-SPECIFIC: could allow controlled y and z here
bool isControlledOp(Operation *op);

bool isTerminationPoint(Operation *op);

class Subcircuit {
protected:
  SetVector<Operation *> ops;
  SetVector<Value> initial_wires;
  SetVector<Value> terminal_wires;
  Operation *start;
  // TODO: these three are really intermediate state
  // for constructing the subcircuit, it would be nice
  // to turn them into shared arguments instead
  SetVector<Value> termination_points;
  SetVector<Value> anchor_points;
  SetVector<Value> seen;

  bool isAfterTerminationPoint(Value wire) {
    return isTerminationPoint(wire.getDefiningOp());
  }

  void maybeAddAnchorPoint(Value v) {
    if (!seen.contains(v))
      anchor_points.insert(v);
  }

  void calculateSubcircuitForQubitForward(OpResult v) {
    seen.insert(v);
    if (!v.hasOneUse()) {
      termination_points.insert(v);
      return;
    }
    Operation *op = v.getUses().begin().getUser();

    if (isTerminationPoint(op)) {
      termination_points.insert(v);
      return;
    }

    ops.insert(op);

    // Controlled not, figure out whether we are tracking the control
    // or target, and add an anchor point to the other qubit
    if (op->getResults().size() > 1) {
      auto control = op->getResult(0);
      auto target = op->getResult(1);
      // Is this the control or target qubit?
      if (v.getResultNumber() == 0) {
        // Tracking the control qubit
        calculateSubcircuitForQubitForward(control);
        maybeAddAnchorPoint(target);
      } else {
        // Tracking the target qubit
        maybeAddAnchorPoint(control);
        calculateSubcircuitForQubitForward(target);
      }
    } else {
      // Otherwise, single qubit gate, just follow result
      calculateSubcircuitForQubitForward(getNextResult(v));
    }
  }

  void calculateSubcircuitForQubitBackward(Value v) {
    seen.insert(v);
    Operation *op = v.getDefiningOp();

    if (isTerminationPoint(op)) {
      termination_points.insert(v);
      return;
    }

    ops.insert(op);

    // Controlled not, figure out whether we are tracking the control
    // or target, and add an anchor point to the other qubit
    // Use getResults() as Rz has two operands but only one result
    if (op->getResults().size() > 1) {
      auto control = op->getOperand(0);
      auto target = op->getOperand(1);
      // Is this the control or target qubit?
      if (v == target) {
        // Tracking the control qubit
        calculateSubcircuitForQubitBackward(control);
        maybeAddAnchorPoint(target);
      } else {
        // Tracking the target qubit
        maybeAddAnchorPoint(control);
        calculateSubcircuitForQubitBackward(target);
      }
    } else {
      // Otherwise, single qubit gate, just follow operand
      calculateSubcircuitForQubitBackward(getNextOperand(v));
    }
  }

  void calculateInitialSubcircuit(Operation *op) {
    // AXIS-SPECIFIC: This could be any controlled operation
    auto cnot = dyn_cast<quake::XOp>(op);
    assert(cnot && cnot.getWires().size() == 2);

    auto result = cnot->getResult(0);
    auto operand = cnot->getOperand(0);
    ops.insert(cnot);
    anchor_points.insert(cnot->getResult(1));
    calculateSubcircuitForQubitForward(result);
    calculateSubcircuitForQubitBackward(operand);

    while (!anchor_points.empty()) {
      auto next = anchor_points.back();
      anchor_points.pop_back();
      calculateSubcircuitForQubitForward(dyn_cast<OpResult>(next));
      seen.remove(next);
      calculateSubcircuitForQubitBackward(next);
    }
  }

  // Prune operations after a termination point from the subcircuit
  void pruneWire(Value wire) {
    if (termination_points.contains(wire))
      termination_points.remove(wire);
    if (!wire.hasOneUse())
      return;
    Operation *op = wire.getUses().begin().getUser();

    ops.remove(op);

    // TODO: According to the paper, if the op is a CNot and the wire we are
    // pruning along is the target, then we do not have to prune along the
    // control wire. However, this prevents placing each subcircuit in a
    // separate block, so it is currently not supported auto opi =
    // dyn_cast<quake::OperatorInterface>(op); assert(opi); auto controls =
    // opi.getControls(); if (controls.size() > 0 &&
    //     std::find(controls.begin(), controls.end(), wire) == controls.end())
    //     { pruneSubcircuit(opi.getWires()[1]); return;
    // }

    for (auto result : op->getResults()) {
      pruneWire(result);
      // Adjust termination border
      for (auto operand : op->getOperands())
        if (ops.contains(operand.getDefiningOp()))
          termination_points.insert(operand);
    }
  }

  void pruneSubcircuit() {
    // The termination boundary should be defined by the first
    // termination point seen along each wire in the subcircuit
    // (this means that it is important to build subcircuits
    // by inspecting controlled gates in topological order)
    for (auto wire : termination_points) {
      if (!isAfterTerminationPoint(wire) && wire.hasOneUse())
        pruneWire(wire);
    }
  }

public:
  /// @brief Constructs a subcircuit with a phase polynomial starting from a
  /// cnot
  Subcircuit(Operation *cnot);

  /// @brief Reconstructs a subcircuit from a subcircuit function
  Subcircuit(func::FuncOp subcircuit_func);

  SetVector<Value> getInitialWires();

  SetVector<Value> getTerminalWires();

  bool isInSubcircuit(Operation *op);

  // TODO: would be nice to make Subcircuit iterable directly
  SetVector<Operation *> getOps();

  /// @brief returns the number of wires in the subcircuit
  size_t numWires();

  /// @brief returns the number of two-qubit operations in the subcircuit
  size_t numCNots();

  Operation *getStart();
};
