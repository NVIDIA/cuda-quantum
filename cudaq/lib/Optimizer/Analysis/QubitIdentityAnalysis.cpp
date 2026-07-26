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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <utility>

using namespace mlir;

using QubitIdentityAnalysis = cudaq::quake::detail::QubitIdentityAnalysis;
using QubitId = QubitIdentityAnalysis::QubitId;
using BorrowKey = std::pair<Attribute, std::int32_t>;
using QubitIdMap = llvm::DenseMap<Value, QubitId>;

// Propagate only an unambiguous one-to-one scalar wire correspondence.
// Reference, aggregate, and malformed shapes leave their results unmapped.
static bool
propagateQubitIds(llvm::DenseMap<Value, QubitId> &qubitIds,
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
    auto [entry, inserted] = qubitIds.try_emplace(result, qubitId);
    if (!inserted && entry->second != qubitId)
      return false;
  }
  return true;
}

static void updateWrappedIdentity(QubitIdMap &qubitIds,
                                  QubitIdMap &referenceQubitIds,
                                  cudaq::quake::WrapOp wrap) {
  auto referenceId = referenceQubitIds.find(wrap.getRefValue());
  if (referenceId == referenceQubitIds.end()) {
    // Without reference alias analysis, an untracked wrap target may alias any
    // tracked binding.
    referenceQubitIds.clear();
    return;
  }

  auto wireId = qubitIds.find(wrap.getWireValue());
  if (wireId == qubitIds.end() || wireId->second != referenceId->second)
    referenceQubitIds.erase(referenceId);
}

// Scan in program order because opaque effects and untracked wraps invalidate
// only the reference bindings active at that point. Block arguments remain
// unknown because valid IR does not guarantee distinct incoming wires.
static void buildQubitIdMap(Block &block, QubitIdMap &qubitIds) {
  QubitId nextQubitId = 0;
  llvm::DenseMap<BorrowKey, QubitId> borrowedQubitIds;
  QubitIdMap referenceQubitIds;

  for (Operation &operation : block) {
    if (auto alloca = dyn_cast<cudaq::quake::AllocaOp>(operation)) {
      if (isa<cudaq::quake::RefType>(alloca.getRefOrVec().getType()))
        referenceQubitIds.try_emplace(alloca.getRefOrVec(), nextQubitId++);
      continue;
    }
    if (auto unwrap = dyn_cast<cudaq::quake::UnwrapOp>(operation)) {
      auto referenceId = referenceQubitIds.find(unwrap.getRefValue());
      if (referenceId != referenceQubitIds.end())
        qubitIds.try_emplace(unwrap.getResult(), referenceId->second);
      continue;
    }
    if (auto wrap = dyn_cast<cudaq::quake::WrapOp>(operation)) {
      updateWrappedIdentity(qubitIds, referenceQubitIds, wrap);
      continue;
    }
    if (auto wrapNew = dyn_cast<cudaq::quake::WrapNewOp>(operation)) {
      auto wireId = qubitIds.find(wrapNew.getWireValue());
      if (wireId != qubitIds.end())
        referenceQubitIds.try_emplace(wrapNew.getResult(), wireId->second);
      continue;
    }
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
    if (isa<CallOpInterface>(operation) || operation.getNumRegions() != 0 ||
        !isMemoryEffectFree(&operation)) {
      referenceQubitIds.clear();
      continue;
    }
    if (auto flow = cudaq::quake::detail::getScalarWireFlow(&operation))
      (void)propagateQubitIds(qubitIds, *flow);
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
    return propagateQubitIds(qubitIds, *flow);

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
