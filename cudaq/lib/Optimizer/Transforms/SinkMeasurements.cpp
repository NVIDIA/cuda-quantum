/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <optional>

#define DEBUG_TYPE "sink-measurements"

namespace cudaq::opt {
#define GEN_PASS_DEF_SINKMEASUREMENTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

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

bool isLocalAlloca(mlir::Value val) {
  return static_cast<bool>(val.getDefiningOp<cudaq::quake::AllocaOp>());
}

QuantumLocation getQuantumLocation(mlir::Value val) {
  if (auto extractRefOp = val.getDefiningOp<cudaq::quake::ExtractRefOp>()) {
    auto baseVeq = extractRefOp.getVeq();
    return {baseVeq,
            extractRefOp.hasConstantIndex()
                ? std::make_optional(extractRefOp.getConstantIndex())
                : std::nullopt,
            false, isLocalAlloca(baseVeq)};
  } else if (mlir::isa<cudaq::quake::VeqType>(val.getType())) {
    return {val, std::nullopt, true, isLocalAlloca(val)};
  } else {
    return {val, std::nullopt, false, isLocalAlloca(val)};
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
bool touchesMeasuredTarget(cudaq::quake::MeasurementInterface msmt,
                           mlir::Operation *candidate) {
  // want to iterate over all all of the targets that msmt measures, and check
  // if candidate may alias with any of those measured targets
  for (mlir::Value target : msmt.getTargets()) {
    for (auto operand : cudaq::quake::getQuantumOperands(candidate))
      if (mayAliasQuantumValues(target, operand))
        return true;

    // want to iterate over all of the quantum results of the candidate to check
    // if candidate may alias with any of measured targets. This check might be
    // overly conservative, but it's possible that candidate produces a quantum
    // result that is an alias of the measurement target, and if that's the
    // case, we don't want to sink the measurement past candidate
    for (mlir::Value result : cudaq::quake::getQuantumResults(candidate))
      if (mayAliasQuantumValues(target, result))
        return true;
  }
  return false;
}

/// Want to check if `candidate` uses any result produced by *producer
bool usesAnyResultOf(mlir::Operation *candidate, mlir::Operation *producer) {
  for (mlir::Value result : producer->getResults()) {
    for (auto operand : candidate->getOperands())
      if (operand == result)
        return true;
  }
  return false;
}

/// We want to check if a given candidate op is a barrier to sinking the
/// measurement op past it.
bool isMovementBarrier(cudaq::quake::MeasurementInterface msmt,
                       mlir::Operation *candidate) {
  // All the ways the candidate could be a barrier:

  // 1. candidate is a terminator op
  if (candidate->hasTrait<mlir::OpTrait::IsTerminator>())
    return true;

  // 2. candidate has regions (since we don't currently look into regions for
  // sinking)
  if (candidate->getNumRegions() > 0)
    return true;

  // 3. candidate is itself a measurement op
  if (candidate->hasTrait<cudaq::QuantumMeasure>())
    return true;

  // 4. candidate uses result of measurement
  if (usesAnyResultOf(candidate, msmt.getOperation()))
    return true;

  // 5. either a candidate's args or it's results alias the measurement target
  if (touchesMeasuredTarget(msmt, candidate))
    return true;

  // 6. candidate is a non-quantum op that may have side effects
  if (!isQuakeOperation(candidate) && !mlir::isMemoryEffectFree(candidate))
    return true;

  return false;
}

/// want to find the latest point in the block that we can sink the measurement
/// op to
mlir::Operation *findInsertionPoint(cudaq::quake::MeasurementInterface msmt) {
  mlir::Operation *measurementOp = msmt.getOperation();

  for (mlir::Operation *candidate = measurementOp->getNextNode(); candidate;
       candidate = candidate->getNextNode())
    if (isMovementBarrier(msmt, candidate))
      return candidate;

  return nullptr;
}

bool sinkMeasurementsInBlock(mlir::Block &block) {
  bool changed = false;
  mlir::SmallVector<cudaq::quake::MeasurementInterface> msmts;

  // find all measurements
  for (mlir::Operation &op : block) {
    if (auto msmt = mlir::dyn_cast<cudaq::quake::MeasurementInterface>(&op))
      msmts.push_back(msmt);
  }

  // for each measurement, find the latest point in the block where we can sink
  // the measurement op to
  for (cudaq::quake::MeasurementInterface msmt : msmts) {
    mlir::Operation *insertionPoint = findInsertionPoint(msmt);
    mlir::Operation *msmtOp = msmt.getOperation();
    mlir::Operation *nextOp = msmtOp->getNextNode();
    if (insertionPoint && insertionPoint != nextOp) {
      msmtOp->moveBefore(insertionPoint);

      changed = true;

      LLVM_DEBUG(llvm::dbgs() << "Sunk measurement op: " << *msmtOp << "\n");
    }
  }
  return changed;
}

} // namespace

class SinkMeasurementsPass
    : public cudaq::opt::impl::SinkMeasurementsBase<SinkMeasurementsPass> {
public:
  using SinkMeasurementsBase::SinkMeasurementsBase;

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();

    bool changed = false;
    for (mlir::Block &block : func.getBody())
      changed |= sinkMeasurementsInBlock(block);

    if (!changed)
      markAllAnalysesPreserved();
  }
};
