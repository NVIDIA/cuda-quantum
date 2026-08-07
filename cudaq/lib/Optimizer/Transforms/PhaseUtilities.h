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
#include <cassert>

namespace cudaq::opt {

/// Return whether \p anchor is a scalar value accepted by quake.phase.
inline bool isScalarPhaseAnchor(mlir::Value anchor) {
  return cudaq::quake::isScalarQubitTarget(anchor);
}

/// Collect the IR-visible quantum roots underlying a phase operand.
///
/// This is deliberately a structural, conservative relation rather than a
/// general quantum alias analysis. It captures wrappers and aggregate views
/// that can make a phase anchor overlap its control predicate.
inline void collectPhaseAnchorRoots(mlir::Value value,
                                    llvm::SmallVectorImpl<mlir::Value> &roots) {
  if (auto unwrap = value.getDefiningOp<cudaq::quake::UnwrapOp>())
    return collectPhaseAnchorRoots(unwrap.getRefValue(), roots);
  if (auto toControl = value.getDefiningOp<cudaq::quake::ToControlOp>())
    return collectPhaseAnchorRoots(toControl.getQubit(), roots);
  if (auto fromControl = value.getDefiningOp<cudaq::quake::FromControlOp>())
    return collectPhaseAnchorRoots(fromControl.getCtrlbit(), roots);
  if (auto extract = value.getDefiningOp<cudaq::quake::ExtractRefOp>())
    return collectPhaseAnchorRoots(extract.getVeq(), roots);
  if (auto relax = value.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    return collectPhaseAnchorRoots(relax.getInputVec(), roots);
  if (auto subveq = value.getDefiningOp<cudaq::quake::SubVeqOp>())
    return collectPhaseAnchorRoots(subveq.getVeq(), roots);
  if (auto member = value.getDefiningOp<cudaq::quake::GetMemberOp>())
    return collectPhaseAnchorRoots(member.getStruq(), roots);
  if (auto concat = value.getDefiningOp<cudaq::quake::ConcatOp>()) {
    for (mlir::Value member : concat.getTargets())
      collectPhaseAnchorRoots(member, roots);
    return;
  }
  if (auto struq = value.getDefiningOp<cudaq::quake::MakeStruqOp>()) {
    for (mlir::Value member : struq.getVeqs())
      collectPhaseAnchorRoots(member, roots);
    return;
  }
  roots.push_back(value);
}

/// Return whether two values may share an IR-visible quantum root.
inline bool phaseOperandsMayShareRoot(mlir::Value first, mlir::Value second) {
  llvm::SmallVector<mlir::Value> firstRoots;
  llvm::SmallVector<mlir::Value> secondRoots;
  collectPhaseAnchorRoots(first, firstRoots);
  collectPhaseAnchorRoots(second, secondRoots);
  for (mlir::Value firstRoot : firstRoots)
    for (mlir::Value secondRoot : secondRoots)
      if (firstRoot == secondRoot)
        return true;
  return false;
}

/// Return whether a scalar phase anchor may alias a control operand.
///
/// A vector may repeat the same reference (for example through quake.concat),
/// so different constant indices alone never prove two extracts distinct.
inline bool mayPhaseAnchorAliasControl(mlir::Value anchor,
                                       mlir::Value control) {
  return phaseOperandsMayShareRoot(anchor, control);
}

/// Return whether an unmaterialized static target may alias a control.
///
/// Any shared aggregate root is rejected before materializing an
/// ExtractRefOp. Different vector indices are not enough to prove distinct
/// qubits: a vector may repeat a reference.
inline bool
mayPhaseAnchorAliasControl(const cudaq::quake::StaticQubitTarget &anchor,
                           mlir::Value control) {
  return phaseOperandsMayShareRoot(anchor.source, control);
}

/// Return whether a planned phase anchor may alias any control.
inline bool
mayPhaseAnchorAliasControl(const cudaq::quake::StaticQubitTarget &anchor,
                           mlir::ValueRange controls) {
  for (mlir::Value control : controls)
    if (mayPhaseAnchorAliasControl(anchor, control))
      return true;
  return false;
}

/// Return whether a phase predicate may repeat or overlap a quantum reference.
///
/// Repeated or overlapping controls do not form a safely lowerable predicate.
/// This remains a conservative structural check, not a general alias analysis.
inline bool hasPotentiallyAliasedPhaseControls(mlir::ValueRange controls) {
  for (mlir::Value control : controls) {
    llvm::SmallVector<mlir::Value> roots;
    collectPhaseAnchorRoots(control, roots);
    for (std::size_t i = 0; i < roots.size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (roots[i] == roots[j])
          return true;
  }

  for (std::size_t i = 0; i < controls.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (phaseOperandsMayShareRoot(controls[i], controls[j]))
        return true;
  return false;
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
  auto replacements = cudaq::quake::getWireValues(controls, {anchor});
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

  auto resultTypes = cudaq::quake::getWireResultTypes(rewriter, result.controls,
                                                      mlir::ValueRange{anchor});
  auto phaseOp = cudaq::quake::PhaseOp::create(
      rewriter, location, resultTypes, /*is_adj=*/false,
      mlir::ValueRange{phase}, result.controls, mlir::ValueRange{anchor},
      negatedControls);
  llvm::SmallVector<mlir::Value> targets{anchor};
  cudaq::quake::threadWireResults(phaseOp, result.controls, targets);
  result.anchor = targets.front();
  return result;
}

} // namespace cudaq::opt
