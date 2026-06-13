/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <optional>

namespace {
struct QuantumLocation {
  mlir::Value base;
  std::optional<std::size_t> index;
  bool isWholeValue;
  bool baseIsLocalAlloca;
};

bool isValidMeasurementInput(mlir::Value val) {
  auto ty = val.getType();
  return mlir::isa<cudaq::quake::WireType, cudaq::quake::RefType,
                   cudaq::quake::VeqType>(ty);
}

bool isLocalAlloca(mlir::Value v) {
  return static_cast<bool>(v.getDefiningOp<cudaq::quake::AllocaOp>());
}

QuantumLocation getQuantumLocation(mlir::Value v) {
  if (auto extractRefOp = v.getDefiningOp<cudaq::quake::ExtractRefOp>()) {
    auto baseVeq = extractRefOp.getVeq();
    return {baseVeq,
            extractRefOp.hasConstantIndex()
                ? std::make_optional(extractRefOp.getConstantIndex())
                : std::nullopt,
            false, isLocalAlloca(baseVeq)};
  } else if (mlir::isa<cudaq::quake::VeqType>(v.getType())) {
    return {v, std::nullopt, true, isLocalAlloca(v)};
  } else {
    return {v, std::nullopt, false, isLocalAlloca(v)};
  }
}

/// Helper to determine if two quantum values may alias.  This is a conservative
/// check that returns true if the values may alias and false if they are known
/// not to alias.  This is used to determine if we can sink measurements through
/// extract_ref or not.
bool mayAliasQuantumValues(mlir::Value lhs, mlir::Value rhs) {

  if (!isValidMeasurementInput(lhs) || !isValidMeasurementInput(rhs))
    return true; // or assert, but for now we just return true if either value
                 // is not quantum

  if (lhs == rhs)
    return true;

  auto lhsLoc = getQuantumLocation(lhs);
  auto rhsLoc = getQuantumLocation(rhs);

  if (lhsLoc.base != rhsLoc.base) {
    if (lhsLoc.baseIsLocalAlloca && rhsLoc.baseIsLocalAlloca)
      return false; // different known unique bases cannot alias
    return true;    // otherwise, we conservatively assume they may alias
  } else {
    if (lhsLoc.isWholeValue || rhsLoc.isWholeValue)
      return true; // if either is a whole veq, they may alias
    if (!lhsLoc.index || !rhsLoc.index)
      return true; // if either index is unknown, they may alias
    if (!lhsLoc.baseIsLocalAlloca || !rhsLoc.baseIsLocalAlloca)
      return true; // if the base is not known unique, they may alias
    return *lhsLoc.index == *rhsLoc.index;
  }
}

/// Helper to determine if a downstream op from measurement operation touches
/// the same measured target.
bool touchesMeasuredTarget(cudaq::quake::MeasurementInterface msmtOp,
                           mlir::Operation *downstreamOp) {
  // want to iterate over all all of the targets that msmtOp measures, and check
  // if downstreamOp may aliias with any of those measured targets
  for (auto target : msmtOp.getTargets()) {
    for (auto operand : cudaq::quake::getQuantumOperands(downstreamOp))
      if (mayAliasQuantumValues(target, operand))
        return true;
  }

  // want to iterate over all of the quantum results of the msmtOp to check if
  // downstreamOp may alias with any of those results
  for ()

    return false;
}

} // namespace
