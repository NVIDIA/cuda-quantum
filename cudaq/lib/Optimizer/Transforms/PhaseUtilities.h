/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "QuakeOperatorUtilities.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include <cassert>

namespace cudaq::opt {

inline llvm::SmallVector<bool>
getControlPolarities(cudaq::quake::PhaseOp phase) {
  return getControlPolarities(phase.getControls(),
                              phase.getNegatedQubitControls());
}

inline mlir::DenseBoolArrayAttr
makeNegatedControlsAttr(mlir::OpBuilder &builder,
                        llvm::ArrayRef<bool> polarities) {
  if (llvm::none_of(polarities, [](bool value) { return value; }))
    return {};
  return builder.getDenseBoolArrayAttr(polarities);
}

/// Returns the signed angle for a `phase` op.
/// Also emits a negation operation if the phase is adjoint.
inline mlir::Value getSignedAngle(mlir::IRRewriter &rewriter,
                                  cudaq::quake::PhaseOp phase) {
  mlir::Value angle = phase.getParameter();
  if (phase.isAdj())
    angle = mlir::arith::NegFOp::create(rewriter, phase.getLoc(), angle);
  return angle;
}

/// Collect the current wire values for a phase's controls and anchor in the
/// order of its wire results.
inline llvm::SmallVector<mlir::Value>
getPhaseReplacements(cudaq::quake::PhaseOp phase, mlir::ValueRange controls,
                     mlir::Value anchor) {
  auto replacements = getWireValues(controls, {anchor});
  assert(replacements.size() == phase.getWires().size() &&
         "phase result count does not match its wire operands");
  return replacements;
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
