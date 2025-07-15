/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Subcircuit.h"
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

unsigned calculateSkip(Operation *op) {
  auto i = 0;
  for (auto type : op->getOperandTypes()) {
    if (isa<quake::WireType>(type))
      return i;
    i++;
  }

  return i;
}

Value getNextOperand(Value v) {
  auto result = dyn_cast<OpResult>(v);
  auto op = result.getDefiningOp();
  auto skip = calculateSkip(op);
  auto operandIDX = result.getResultNumber() + skip;
  return op->getOperand(operandIDX);
}

OpResult getNextResult(Value v) {
  assert(v.hasOneUse());
  auto correspondingOperand = v.getUses().begin();
  auto op = correspondingOperand.getUser();
  auto skip = calculateSkip(op);
  auto resultIDX = correspondingOperand.getOperand()->getOperandNumber() - skip;
  return op->getResult(resultIDX);
}

bool processed(Operation *op) { return op->hasAttr("processed"); }

void markProcessed(Operation *op) {
  op->setAttr("processed", OpBuilder(op).getUnitAttr());
}

// AXIS-SPECIFIC: could allow controlled y and z here
bool isControlledOp(Operation *op) {
  return isa<quake::XOp>(op) && op->getNumOperands() == 2;
}

bool isTerminationPoint(Operation *op) {
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

/// @brief Constructs a subcircuit with a phase polynomial starting from a
/// cnot
Subcircuit::Subcircuit(Operation *cnot) {
  calculateInitialSubcircuit(cnot);
  pruneSubcircuit();
  for (auto *op : ops)
    markProcessed(op);
  start = cnot;

  for (auto wire : termination_points)
    if (isAfterTerminationPoint(wire))
      initial_wires.insert(wire);
    else
      terminal_wires.insert(wire);
}

Subcircuit *Subcircuit::constructFromBlock(Block *b) {
  // First, some validation
  auto subcircuit = new Subcircuit();
  // Construct the subcircuit
  for (auto &op : b->getOperations()) {
    auto *opp = &op;
    if (opp == b->getTerminator())
      continue;
    // Ensure circuit only contains valid operations
    if (isTerminationPoint(opp))
      return nullptr;
    subcircuit->ops.insert(opp);
  }
  for (auto arg : b->getArguments())
    if (isa<quake::WireType>(arg.getType()))
      subcircuit->initial_wires.insert(arg);
  for (auto ret : b->getTerminator()->getOperands())
    subcircuit->terminal_wires.insert(ret);
  return subcircuit;
}

Subcircuit *Subcircuit::constructFromFunc(func::FuncOp subcircuit_func) {
  // First, some validation
  if (!subcircuit_func.getOperation()->hasAttr("subcircuit"))
    return nullptr;
  if (subcircuit_func.getBlocks().size() != 1)
    return nullptr;
  auto &body_block = subcircuit_func.getRegion().getBlocks().front();
  auto subcircuit = new Subcircuit();
  // Construct the subcircuit
  for (auto &op : body_block) {
    auto *opp = &op;
    if (opp == body_block.getTerminator())
      continue;
    // Ensure circuit only contains valid operations
    if (isTerminationPoint(opp))
      return nullptr;
    subcircuit->ops.insert(opp);
  }
  for (auto arg : body_block.getArguments())
    if (isa<quake::WireType>(arg.getType()))
      subcircuit->initial_wires.insert(arg);
  for (auto ret : body_block.getTerminator()->getOperands())
    subcircuit->terminal_wires.insert(ret);
  return subcircuit;
}

SetVector<Value> Subcircuit::getInitialWires() { return initial_wires; }

SetVector<Value> Subcircuit::getTerminalWires() { return terminal_wires; }

bool Subcircuit::isInSubcircuit(Operation *op) { return ops.contains(op); }

// TODO: would be nice to make Subcircuit iterable directly
SetVector<Operation *> Subcircuit::getOps() { return ops; }

/// @brief returns the number of wires in the subcircuit
size_t Subcircuit::numWires() { return getInitialWires().size(); }

/// @brief returns the number of two-qubit operations in the subcircuit
size_t Subcircuit::numCNots() {
  size_t num_cnots = 0;
  for (auto *op : ops)
    if (isControlledOp(op))
      num_cnots++;
  return num_cnots;
}

Operation *Subcircuit::getStart() { return start; }
