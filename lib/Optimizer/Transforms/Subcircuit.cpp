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
#include "cudaq/Optimizer/Builder/Factory.h"
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

SmallVector<Operation *> Subcircuit::getOrderedOps() {
  if (ordered_ops.size() == 0 && ops.size() > 0) {
    ordered_ops = SmallVector<Operation *>(ops.begin(), ops.end());
    auto less = [&](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    };
    std::sort(ordered_ops.begin(), ordered_ops.end(), less);
  }

  return ordered_ops;
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

bool Subcircuit::moveToFunc(mlir::ModuleOp &mod, llvm::StringRef name) {
  // Find where to place the subcircuit
  Operation *latest_start = &start->getBlock()->front();
  Operation *earliest_end = &start->getBlock()->back();

  for (auto nlw : qubits) {
    if (!nlw)
      continue;
    auto start = findAncestorInSubcircuitBlock(nlw->getStart());
    if (start && latest_start->isBeforeInBlock(start))
      latest_start = start;
    auto end = nlw->getEnd();
    if (!end)
      continue;
    end = findAncestorInSubcircuitBlock(end);
    if (end && end->isBeforeInBlock(earliest_end))
      earliest_end = end;
  }

  // Ensure the subcircuit is movable
  if (earliest_end->isBeforeInBlock(latest_start)) {
    // TODO: should we be so noisy here?
    start->emitWarning("Cannot separate subcircuit: end before start");
    latest_start->emitRemark("Start here");
    earliest_end->emitRemark("End here");
    // TODO: do we want to unmark as processed so ops can may be absorbed by
    // later subcircuits?
    return false;
  }

  // Set up function
  auto args = getRefs();
  SmallVector<Type> types(args.size(), quake::RefType::get(mod.getContext()));
  auto fun = cudaq::opt::factory::createFunction(name, {}, types, mod);
  fun.setPrivate();
  auto entry = fun.addEntryBlock();
  OpBuilder builder(fun);
  fun.getOperation()->setAttr("subcircuit", builder.getUnitAttr());

  auto add_arg = [&](Value v) {
    auto idx = args.size();
    args.push_back(v);
    fun.insertArgument(idx, v.getType(), {}, v.getDefiningOp()->getLoc());
    return fun.getArgument(idx);
  };

  // Clone operations into the new function
  builder.setInsertionPointToStart(entry);
  for (auto op : getOrderedOps()) {
    auto clone = builder.clone(*op);
    for (size_t i = 0; i < clone->getOperands().size(); i++) {
      auto operand = clone->getOperand(i);
      // Add constants as arguments (after the arguments for the refs)
      if (!quake::isQuantumType(operand.getType())) {
        auto arg = add_arg(operand);
        clone->setOperand(i, arg);
      }
      clone->removeAttr("processed");
    }
  }

  size_t i = 0;
  // Indirect the refs through the arguments
  for (auto ref : getRefs()) {
    auto arg = fun.getArgument(i++);
    ref.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return use.getOwner()->getBlock() == entry;
    });
  }

  builder.create<cudaq::cc::ReturnOp>(fun.getLoc());

  builder.setInsertionPointAfter(latest_start);
  // Invoke the function with the cloned subcircuit ops
  builder.create<func::CallOp>(start->getLoc(), fun, args);

  return true;
}
