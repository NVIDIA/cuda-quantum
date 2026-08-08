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
#include "llvm/ADT/SmallVector.h"
#include <utility>

using namespace mlir;

using cudaq::quake::detail::QubitIdentityAnalysis;
using QubitId = QubitIdentityAnalysis::QubitId;
using BorrowKey = std::pair<Attribute, std::int32_t>;

// Initial construction preserves every identity it can prove independently.
// An unknown input leaves only its corresponding result unidentified.
static void
propagateKnownQubitIds(llvm::DenseMap<Value, QubitId> &qubitIds,
                       const cudaq::quake::detail::ScalarWireFlow &flow) {
  for (auto [input, result] : llvm::zip(flow.inputs, flow.results)) {
    auto qubitId = qubitIds.find(input);
    if (qubitId != qubitIds.end())
      qubitIds.try_emplace(result, qubitId->second);
  }
}

// Incremental maintenance is atomic: every result identity must be known before
// the live analysis can accept an inserted operation.
static bool registerQubitIds(llvm::DenseMap<Value, QubitId> &qubitIds,
                             const cudaq::quake::detail::ScalarWireFlow &flow) {
  llvm::SmallVector<QubitId> inputIds;
  inputIds.reserve(flow.inputs.size());
  for (Value input : flow.inputs) {
    auto qubitId = qubitIds.find(input);
    if (qubitId == qubitIds.end())
      return false;
    inputIds.push_back(qubitId->second);
  }

  for (auto [result, qubitId] : llvm::zip(flow.results, inputIds)) {
    auto entry = qubitIds.find(result);
    if (entry != qubitIds.end() && entry->second != qubitId)
      return false;
  }
  for (auto [result, qubitId] : llvm::zip(flow.results, inputIds))
    qubitIds.try_emplace(result, qubitId);
  return true;
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
    if (auto flow = cudaq::quake::detail::getScalarWireFlow(&operation))
      propagateKnownQubitIds(qubitIds, *flow);
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

bool QubitIdentityAnalysis::haveSameOrderedQubitIdentities(
    ValueRange lhs, ValueRange rhs) const {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [lhsValue, rhsValue] : llvm::zip(lhs, rhs)) {
    auto lhsId = getQubitId(lhsValue);
    auto rhsId = getQubitId(rhsValue);
    if (!lhsId || !rhsId || lhsId != rhsId)
      return false;
  }
  return true;
}

bool QubitIdentityAnalysis::registerOperation(Operation &operation) {
  if (auto flow = cudaq::quake::detail::getScalarWireFlow(&operation))
    return registerQubitIds(qubitIds, *flow);

  bool hasQuantumValue =
      llvm::any_of(operation.getOperandTypes(), cudaq::quake::isQuantumType) ||
      llvm::any_of(operation.getResultTypes(), cudaq::quake::isQuantumType);
  return !hasQuantumValue;
}

bool QubitIdentityAnalysis::replacementPreservesIdentities(
    Operation &operation, ValueRange replacement) const {
  if (operation.getNumResults() != replacement.size())
    return false;
  for (auto [result, replacementValue] :
       llvm::zip(operation.getResults(), replacement)) {
    if (!cudaq::quake::isQuantumType(result.getType()))
      continue;
    auto oldId = getQubitId(result);
    auto replacementId = getQubitId(replacementValue);
    if (!oldId || !replacementId || oldId != replacementId)
      return false;
  }
  return true;
}

void QubitIdentityAnalysis::eraseOperation(Operation &operation) {
  for (Value result : operation.getResults())
    qubitIds.erase(result);
}
