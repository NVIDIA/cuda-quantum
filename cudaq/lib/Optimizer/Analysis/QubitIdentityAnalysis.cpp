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
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include <utility>

using namespace mlir;

using QubitIdentityAnalysis = cudaq::quake::detail::QubitIdentityAnalysis;
using QubitId = QubitIdentityAnalysis::QubitId;
using BorrowKey = std::pair<Attribute, std::int32_t>;
using QubitIdMap = llvm::DenseMap<Value, QubitId>;

static void propagateQubitIds(QubitIdMap &qubitIds, Operation *operation) {
  auto flow = cudaq::quake::detail::getScalarWireFlow(operation);
  if (!flow)
    return;
  for (auto [input, result] : llvm::zip(flow->inputs, flow->results)) {
    auto qubitId = qubitIds.find(input);
    if (qubitId != qubitIds.end())
      qubitIds.try_emplace(result, qubitId->second);
  }
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
    propagateQubitIds(qubitIds, &operation);
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
