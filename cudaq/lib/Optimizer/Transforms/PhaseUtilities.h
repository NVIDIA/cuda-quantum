/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

namespace cudaq::opt {

/// Return the wire result types for a Quake operator with the given controls
/// and targets. Quake orders wire results by controls first, then targets.
inline llvm::SmallVector<mlir::Type>
getWireResultTypes(mlir::OpBuilder &builder, mlir::ValueRange controls,
                   mlir::ValueRange targets) {
  auto wireType = cudaq::quake::WireType::get(builder.getContext());
  llvm::SmallVector<mlir::Type> resultTypes;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      resultTypes.push_back(wireType);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      resultTypes.push_back(wireType);
  return resultTypes;
}

/// Update controls and targets to the corresponding wire results of a newly
/// created Quake operator.
template <typename Op>
inline void threadWireResults(Op op,
                              llvm::MutableArrayRef<mlir::Value> controls,
                              llvm::MutableArrayRef<mlir::Value> targets) {
  unsigned result = 0;
  for (mlir::Value &control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      control = op.getWires()[result++];
  for (mlir::Value &target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      target = op.getWires()[result++];
  assert(result == op.getWires().size() &&
         "gate result count does not match its wire operands");
}

/// Collect threaded values in Quake's wire-result order.
inline llvm::SmallVector<mlir::Value> getWireValues(mlir::ValueRange controls,
                                                    mlir::ValueRange targets) {
  llvm::SmallVector<mlir::Value> values;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      values.push_back(control);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      values.push_back(target);
  return values;
}

inline llvm::SmallVector<bool>
getControlPolarities(cudaq::quake::PhaseOp phase) {
  llvm::SmallVector<bool> polarities(phase.getControls().size(), false);
  if (auto negated = phase.getNegatedQubitControls())
    for (auto [index, value] : llvm::enumerate(*negated))
      polarities[index] = value;
  return polarities;
}

inline mlir::DenseBoolArrayAttr
makeNegatedControlsAttr(mlir::OpBuilder &builder,
                        llvm::ArrayRef<bool> polarities) {
  if (llvm::none_of(polarities, [](bool value) { return value; }))
    return {};
  return builder.getDenseBoolArrayAttr(polarities);
}

struct PhaseCorrection {
  llvm::SmallVector<mlir::Value> controls;
  mlir::Value anchor;
};

/// Emit an exact phase correction and return the latest wire values.
///
/// The correction is emitted immediately after the replacement that requires
/// it. A literal zero is omitted. Nonzero constant multiples of 2*pi are left
/// to PhaseOp's canonicalizer so that there is a single implementation of its
/// floating-point tolerance policy.
inline PhaseCorrection
emitPhaseCorrection(mlir::OpBuilder &rewriter, mlir::Location location,
                    mlir::Value phase, mlir::ValueRange controls,
                    mlir::DenseBoolArrayAttr negatedControls,
                    mlir::Value anchor) {
  PhaseCorrection result{llvm::SmallVector<mlir::Value>(controls), anchor};

  if (auto constant = phase.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto angle = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue());
        angle && angle.getValue().isZero())
      return result;

  auto resultTypes =
      getWireResultTypes(rewriter, result.controls, mlir::ValueRange{anchor});
  auto phaseOp = cudaq::quake::PhaseOp::create(
      rewriter, location, resultTypes, /*is_adj=*/false,
      mlir::ValueRange{phase}, result.controls, mlir::ValueRange{anchor},
      negatedControls);
  llvm::SmallVector<mlir::Value> targets{anchor};
  threadWireResults(phaseOp, result.controls, targets);
  result.anchor = targets.front();
  return result;
}

} // namespace cudaq::opt
