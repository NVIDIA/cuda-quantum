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

} // namespace cudaq::opt
