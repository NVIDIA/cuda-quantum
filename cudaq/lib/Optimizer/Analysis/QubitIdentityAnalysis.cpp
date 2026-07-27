/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QubitIdentityAnalysis.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include <utility>

using namespace mlir;

using cudaq::quake::detail::QubitIdentityAnalysis;
using QubitId = QubitIdentityAnalysis::QubitId;
using BorrowKey = std::pair<Attribute, std::int32_t>;

// Propagate only an unambiguous one-to-one scalar wire correspondence.
// Reference, aggregate, and malformed shapes leave their results unmapped.
static void
propagateQubitIdsThroughWires(llvm::DenseMap<Value, QubitId> &qubitIds,
                              ValueRange wireInputs, ValueRange wireResults) {
  if (wireInputs.size() != wireResults.size() ||
      llvm::any_of(wireInputs,
                   [](Value input) {
                     return !isa<cudaq::quake::WireType>(input.getType());
                   }) ||
      llvm::any_of(wireResults, [](Value result) {
        return !isa<cudaq::quake::WireType>(result.getType());
      }))
    return;
  for (auto [input, result] : llvm::zip(wireInputs, wireResults)) {
    auto qubitId = qubitIds.find(input);
    if (qubitId != qubitIds.end())
      qubitIds.try_emplace(result, qubitId->second);
  }
}

// Thread qubit IDs through an operator only when every control and target is a
// scalar wire. Mixed and reusable-control forms remain unsupported boundaries.
static void
propagateQubitIdsThroughOperator(llvm::DenseMap<Value, QubitId> &qubitIds,
                                 cudaq::quake::OperatorInterface op) {
  auto wireInputs = cudaq::quake::getWireOperands(op);
  if (wireInputs.size() != op.getControls().size() + op.getTargets().size())
    return;
  propagateQubitIdsThroughWires(qubitIds, wireInputs, op.getWires());
}

// Build block-local qubit identities in program order. Block arguments and
// null wires introduce IDs, repeated borrows reuse their (wire set, identity)
// ID, and supported scalar-wire operators propagate IDs.
static void buildQubitIdMap(Block &block,
                            llvm::DenseMap<Value, QubitId> &qubitIds) {
  QubitId nextQubitId = 0;
  llvm::DenseMap<BorrowKey, QubitId> borrowedQubitIds;

  for (BlockArgument argument : block.getArguments())
    if (isa<cudaq::quake::WireType>(argument.getType()))
      qubitIds.try_emplace(argument, nextQubitId++);

  for (Operation &operation : block) {
    if (auto nullWire = dyn_cast<cudaq::quake::NullWireOp>(operation)) {
      qubitIds.try_emplace(nullWire.getResult(), nextQubitId++);
      continue;
    }
    if (auto borrowWire = dyn_cast<cudaq::quake::BorrowWireOp>(operation)) {
      BorrowKey key{borrowWire.getSetNameAttr(), borrowWire.getIdentity()};
      auto [qubitId, inserted] = borrowedQubitIds.try_emplace(key, nextQubitId);
      if (inserted)
        ++nextQubitId;
      qubitIds.try_emplace(borrowWire.getResult(), qubitId->second);
      continue;
    }
    if (auto operatorInterface =
            dyn_cast<cudaq::quake::OperatorInterface>(operation))
      propagateQubitIdsThroughOperator(qubitIds, operatorInterface);
    else if (auto measurement =
                 dyn_cast<cudaq::quake::MeasurementInterface>(operation))
      propagateQubitIdsThroughWires(qubitIds, measurement.getTargets(),
                                    measurement.getWires());
    else if (auto reset = dyn_cast<cudaq::quake::ResetOp>(operation))
      propagateQubitIdsThroughWires(qubitIds, ValueRange{reset.getTargets()},
                                    reset.getWires());
  }
}

QubitIdentityAnalysis::QubitIdentityAnalysis(Block &block) {
  buildQubitIdMap(block, qubitIds);
}

std::optional<QubitId>
QubitIdentityAnalysis::getQubitId(mlir::Value value) const {
  auto qubitId = qubitIds.find(value);
  if (qubitId == qubitIds.end())
    return std::nullopt;
  return qubitId->second;
}
