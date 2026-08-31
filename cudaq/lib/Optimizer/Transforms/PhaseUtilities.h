/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
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

static bool areProvablyDistinctPhaseRefs(mlir::Value first,
                                         mlir::Value second) {
  // if control types, get a ref type instead
  first = cudaq::quake::unwrapFromControlVal(first);
  second = cudaq::quake::unwrapFromControlVal(second);

  // if `quake.extract_ref`, grab the `veq` that it came from
  auto firstExtract = first.getDefiningOp<cudaq::quake::ExtractRefOp>();
  auto secondExtract = second.getDefiningOp<cudaq::quake::ExtractRefOp>();

  // if you can't verify the source (e.g., came from an argument) or one of the
  // veqs is dynamically sized, no alias
  if (!firstExtract || !secondExtract || !firstExtract.hasConstantIndex() ||
      !secondExtract.hasConstantIndex()) {
    return false;
  }

  if (firstExtract.getConstantIndex() == secondExtract.getConstantIndex())
    return false;

  auto firstVeq = cudaq::quake::getKnownAllocaVeq(firstExtract.getVeq());
  auto secondVeq = cudaq::quake::getKnownAllocaVeq(secondExtract.getVeq());

  return firstVeq && firstVeq == secondVeq;
}

static bool phaseValuesMayShareRoot(mlir::Value first, mlir::Value second) {
  llvm::SmallVector<mlir::Value, 4> firstRoots;
  llvm::SmallVector<mlir::Value, 4> secondRoots;

  collectPhaseAnchorRoots(first, firstRoots);
  collectPhaseAnchorRoots(second, secondRoots);

  return llvm::any_of(firstRoots, [&](mlir::Value root) {
    return llvm::is_contained(secondRoots, root);
  });
}

static bool phaseOperandsMayShareRoot(mlir::ValueRange first,
                                      mlir::ValueRange second) {
  for (mlir::Value lhs : first)
    for (mlir::Value rhs : second) {
      if (areProvablyDistinctPhaseRefs(lhs, rhs))
        continue;
      if (phaseValuesMayShareRoot(lhs, rhs))
        return true;
    }

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

/// Return whether an un-materialized static target may alias a control.
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

/// The kinds of root identity tracked when deciding whether the anchored
/// R1/Rz fallback can use a phase anchor safely.
enum class PhaseAnchorFallbackRootKind {
  /// A qubit allocated locally by this function.
  FreshLocal,
  /// A value supplied by the caller of this function.
  FunctionInput,
  /// A value whose quantum provenance cannot be resolved conservatively.
  Unknown,
};

struct PhaseAnchorFallbackRoot {
  mlir::Value value;
  PhaseAnchorFallbackRootKind kind;
};

/// Return the reference that originated a scalar wire, if it is known.
inline mlir::Value getPhaseWireSourceReference(mlir::Value value) {
  if (auto unwrap = value.getDefiningOp<cudaq::quake::UnwrapOp>())
    return unwrap.getRefValue();
  if (auto toControl = value.getDefiningOp<cudaq::quake::ToControlOp>())
    return getPhaseWireSourceReference(toControl.getQubit());
  if (auto fromControl = value.getDefiningOp<cudaq::quake::FromControlOp>())
    return getPhaseWireSourceReference(fromControl.getCtrlbit());
  if (mlir::Operation *def = value.getDefiningOp())
    if (auto flow = cudaq::quake::detail::getThreadedWireFlow(def))
      for (auto [index, result] : llvm::enumerate(flow->results))
        if (value == result)
          return getPhaseWireSourceReference(flow->inputs[index]);
  return {};
}

/// Return whether a wrap provably preserves its target reference binding.
inline bool isProvenSelfPhaseWrap(cudaq::quake::WrapOp wrap) {
  mlir::Value source = getPhaseWireSourceReference(wrap.getWireValue());
  return source && source == wrap.getRefValue();
}

/// Return whether a reference-like root may have been rebound before `at`.
///
/// A `quake.wrap` associates a wire with an existing reference, so a local
/// allocation is no longer known to name its original physical qubit after a
/// matching (or structurally overlapping) non-self wrap. A wrap in another CFG
/// block or region is conservatively treated as preceding `at`; only a later
/// wrap in the same block is known not to affect the operand at the lowering
/// site.
inline bool mayHaveReboundPhaseRoot(mlir::Value root, mlir::Operation *at) {
  if (!at)
    return true;
  auto function = at->getParentOfType<mlir::FunctionOpInterface>();
  if (!function)
    return true;

  bool mayBeRebound = false;
  function.walk([&](cudaq::quake::WrapOp wrap) {
    if (!phaseValuesMayShareRoot(root, wrap.getRefValue()) ||
        isProvenSelfPhaseWrap(wrap))
      return;
    if (wrap->getBlock() != at->getBlock() || wrap->isBeforeInBlock(at))
      mayBeRebound = true;
  });
  return mayBeRebound;
}

/// Return whether an argument is supplied at a function boundary.
inline bool isFunctionEntryBlockArgument(mlir::BlockArgument argument) {
  mlir::Block *owner = argument.getOwner();
  auto function =
      mlir::dyn_cast<mlir::FunctionOpInterface>(owner->getParentOp());
  return function && !function.empty() &&
         &function.getFunctionBody().front() == owner;
}

/// Collect the physical-qubit origins of a phase operand for the anchored
/// fallback. Unlike collectPhaseAnchorRoots, this tracks unresolved value
/// provenance so the fallback rejects overlap visible in this function while
/// relying on the PhaseOp producer contract for opaque function inputs.
inline void collectPhaseAnchorFallbackRoots(
    mlir::Value value, llvm::SmallVectorImpl<PhaseAnchorFallbackRoot> &roots,
    mlir::Operation *at) {
  if (mlir::isa<cudaq::quake::RefType>(value.getType()) &&
      mayHaveReboundPhaseRoot(value, at)) {
    roots.push_back({value, PhaseAnchorFallbackRootKind::Unknown});
    return;
  }
  if (auto unwrap = value.getDefiningOp<cudaq::quake::UnwrapOp>())
    return collectPhaseAnchorFallbackRoots(unwrap.getRefValue(), roots,
                                           unwrap.getOperation());
  if (auto wrapNew = value.getDefiningOp<cudaq::quake::WrapNewOp>())
    return collectPhaseAnchorFallbackRoots(wrapNew.getWireValue(), roots,
                                           wrapNew.getOperation());
  if (auto toControl = value.getDefiningOp<cudaq::quake::ToControlOp>())
    return collectPhaseAnchorFallbackRoots(toControl.getQubit(), roots, at);
  if (auto fromControl = value.getDefiningOp<cudaq::quake::FromControlOp>())
    return collectPhaseAnchorFallbackRoots(fromControl.getCtrlbit(), roots, at);
  if (auto extract = value.getDefiningOp<cudaq::quake::ExtractRefOp>())
    return collectPhaseAnchorFallbackRoots(extract.getVeq(), roots, at);
  if (auto relax = value.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    return collectPhaseAnchorFallbackRoots(relax.getInputVec(), roots, at);
  if (auto subveq = value.getDefiningOp<cudaq::quake::SubVeqOp>())
    return collectPhaseAnchorFallbackRoots(subveq.getVeq(), roots, at);
  if (auto init = value.getDefiningOp<cudaq::quake::InitializeStateOp>())
    return collectPhaseAnchorFallbackRoots(init.getTargets(), roots, at);
  if (auto member = value.getDefiningOp<cudaq::quake::GetMemberOp>())
    return collectPhaseAnchorFallbackRoots(member.getStruq(), roots, at);
  if (auto concat = value.getDefiningOp<cudaq::quake::ConcatOp>()) {
    for (mlir::Value member : concat.getTargets())
      collectPhaseAnchorFallbackRoots(member, roots, at);
    return;
  }
  if (auto struq = value.getDefiningOp<cudaq::quake::MakeStruqOp>()) {
    for (mlir::Value member : struq.getVeqs())
      collectPhaseAnchorFallbackRoots(member, roots, at);
    return;
  }

  if (mlir::Operation *def = value.getDefiningOp()) {
    if (auto flow = cudaq::quake::detail::getThreadedWireFlow(def))
      for (auto [index, result] : llvm::enumerate(flow->results))
        if (value == result)
          return collectPhaseAnchorFallbackRoots(flow->inputs[index], roots,
                                                 def);

    if (mlir::isa<cudaq::quake::AllocaOp>(def)) {
      roots.push_back({value, mayHaveReboundPhaseRoot(value, at)
                                  ? PhaseAnchorFallbackRootKind::Unknown
                                  : PhaseAnchorFallbackRootKind::FreshLocal});
      return;
    }
    if (mlir::isa<cudaq::quake::NullWireOp>(def)) {
      roots.push_back({value, PhaseAnchorFallbackRootKind::FreshLocal});
      return;
    }
  } else if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    roots.push_back({value, isFunctionEntryBlockArgument(argument)
                                ? PhaseAnchorFallbackRootKind::FunctionInput
                                : PhaseAnchorFallbackRootKind::Unknown});
    return;
  }

  roots.push_back({value, PhaseAnchorFallbackRootKind::Unknown});
}

/// Return whether an anchor and control may overlap when the phase lowering
/// must use the anchor as the target of its R1/Rz fallback sequence.
inline bool phaseFallbackAnchorMayAliasControl(mlir::Value anchor,
                                               mlir::Value control,
                                               mlir::Operation *at) {
  // A prior non-self wrap can invalidate a syntactic distinctness proof.
  if (mayHaveReboundPhaseRoot(anchor, at) ||
      mayHaveReboundPhaseRoot(control, at))
    return true;
  if (areProvablyDistinctPhaseRefs(anchor, control))
    return false;
  if (mayPhaseAnchorAliasControl(anchor, control))
    return true;

  llvm::SmallVector<PhaseAnchorFallbackRoot, 4> anchorRoots;
  llvm::SmallVector<PhaseAnchorFallbackRoot, 4> controlRoots;
  collectPhaseAnchorFallbackRoots(anchor, anchorRoots, at);
  collectPhaseAnchorFallbackRoots(control, controlRoots, at);

  for (const PhaseAnchorFallbackRoot &anchorRoot : anchorRoots)
    for (const PhaseAnchorFallbackRoot &controlRoot : controlRoots) {
      if (anchorRoot.value == controlRoot.value ||
          anchorRoot.kind == PhaseAnchorFallbackRootKind::Unknown ||
          controlRoot.kind == PhaseAnchorFallbackRootKind::Unknown)
        return true;
    }
  return false;
}

/// Return whether a phase predicate may repeat or overlap a quantum reference.
///
/// Repeated or overlapping controls do not form a predicate that can be safely
/// lowered. This remains a conservative structural check, not a general alias
/// analysis.
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

  auto resultTypes = cudaq::quake::getWireResultTypes(result.controls,
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
