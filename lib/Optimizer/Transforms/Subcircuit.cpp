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

/// @brief Constructs a subcircuit with a phase polynomial starting from a
/// cnot
Subcircuit::Subcircuit(Operation *cnot, NetlistContainer *container) {
  start = cnot;
  this->container = container;
  qubits = SmallVector<NetlistWrapper *>(container->size(), nullptr);
  calculateInitialSubcircuit();
  pruneSubcircuit();
  for (auto op : ops)
    markProcessed(op);
}

Subcircuit::~Subcircuit() {
  for (auto wrapper : qubits)
    if (wrapper)
      delete wrapper;
}

SetVector<Operation *> Subcircuit::getOps() { return ops; }

SmallVector<Operation *> Subcircuit::getOrderedOps() {
  auto ordered = SmallVector<Operation *>(ops.begin(), ops.end());
  auto less = [&](Operation *a, Operation *b) { return a->isBeforeInBlock(b); };
  std::sort(ordered.begin(), ordered.end(), less);
  return ordered;
}

Operation *Subcircuit::getStart() { return start; }

SmallVector<Value> Subcircuit::getRefs() {
  SmallVector<Value> refs;
  for (auto wrapper : qubits)
    if (wrapper)
      refs.push_back(wrapper->getDef());

  return refs;
}

size_t Subcircuit::numRefs() {
  size_t count = 0;
  for (auto wrapper : qubits)
    if (wrapper)
      count++;
  return count;
}
