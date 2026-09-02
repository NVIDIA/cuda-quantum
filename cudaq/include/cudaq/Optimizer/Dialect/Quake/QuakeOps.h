/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Dialect/Traits.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include <cstddef>
#include <optional>

//===----------------------------------------------------------------------===//
// Canonicalizer functions.
//===----------------------------------------------------------------------===//

namespace cudaq::quake {
mlir::Value createConstantAlloca(mlir::PatternRewriter &builder,
                                 mlir::Location loc, mlir::OpResult result,
                                 mlir::ValueRange args);

void getResetEffectsImpl(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects,
    llvm::MutableArrayRef<mlir::OpOperand> targets);
void getMeasurementEffectsImpl(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects,
    llvm::MutableArrayRef<mlir::OpOperand> targets);
void getOperatorEffectsImpl(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects,
    llvm::MutableArrayRef<mlir::OpOperand> controls,
    llvm::MutableArrayRef<mlir::OpOperand> targets);

mlir::ParseResult genericOpParse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
void genericOpPrinter(mlir::OpAsmPrinter &_odsPrinter, mlir::Operation *op,
                      bool isAdj, mlir::OperandRange params,
                      mlir::OperandRange ctrls, mlir::OperandRange targs,
                      mlir::DenseBoolArrayAttr negatedQubitControlsAttr);
} // namespace cudaq::quake

//===----------------------------------------------------------------------===//
// Tablegen generated logic.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h.inc"

//===----------------------------------------------------------------------===//
// Utility functions to test the form of an operation.
//===----------------------------------------------------------------------===//

// Is \p op in the Quake dialect?
inline bool isQuakeOperation(mlir::Operation *op) {
  if (auto *dialect = op->getDialect())
    return dialect->getNamespace() == "quake";
  return false;
}

namespace cudaq::quake {
namespace detail {
/// Scalar-wire inputs and the results that carry the same qubits by position.
struct ScalarWireFlow {
  mlir::SmallVector<mlir::Value> inputs;
  mlir::SmallVector<mlir::Value> results;
};

/// Return the one-to-one scalar-wire flow for a memory-effect-free operator,
/// measurement, or reset that owns no regions and has no successors.
/// Operator inputs contain controls followed by targets in interface order.
/// Measurement and reset inputs contain targets in interface order.
/// Unsupported forms and mismatched input and result shapes return no value.
std::optional<ScalarWireFlow> getScalarWireFlow(mlir::Operation *operation);

/// Return the scalar-wire flow of an operator, measurement, or reset, ignoring
/// the operands that thread no wire. A `ref`, `veq`, or `control` operand has
/// no corresponding result, so only the wire operands are paired with the wire
/// results by position. Unlike `getScalarWireFlow`, this accepts the mixed
/// forms that arise when some operands are still in reference or control form.
/// Unsupported forms and mismatched input and result shapes return no value.
std::optional<ScalarWireFlow> getThreadedWireFlow(mlir::Operation *operation);
} // namespace detail

/// Returns true if and only if any quantum operand has type `!quake.ref` or
/// `!quake.veq`.
inline bool hasReference(mlir::Operation *op) {
  for (mlir::Value opnd : op->getOperands())
    if (isQuantumReferenceType(opnd.getType()))
      return true;
  return false;
}

/// Return the static size of a `!quake.veq` Value. Looks through RelaxSizeOp
/// when the surface type is dynamically sized but the inner value has a known
/// size.
inline std::optional<std::size_t> getVeqSize(mlir::Value v) {
  auto veqTy = mlir::dyn_cast<cudaq::quake::VeqType>(v.getType());
  if (!veqTy)
    return std::nullopt;
  if (veqTy.hasSpecifiedSize())
    return veqTy.getSize();
  if (auto relaxOp = v.getDefiningOp<cudaq::quake::RelaxSizeOp>()) {
    // RelaxSizeOp verifier guarantees input is VeqType when result is VeqType.
    return getVeqSize(relaxOp.getInputVec());
  }
  return std::nullopt;
}

/// A statically selectable scalar qubit represented by a top-level Quake
/// target. A vector target records the element that must be extracted; a
/// scalar reference or wire has no element index.
struct StaticQubitTarget {
  mlir::Value source;
  std::size_t sourceIndex;
  std::optional<std::size_t> elementIndex;
};

/// Return whether \p target is a single quantum value rather than an aggregate
/// vector.
bool isScalarQubitTarget(mlir::Value target);

/// Plan a statically selectable scalar target without creating IR.
inline std::optional<StaticQubitTarget>
planStaticQubitTarget(mlir::Value target, std::size_t sourceIndex) {
  if (isScalarQubitTarget(target))
    return StaticQubitTarget{target, sourceIndex, std::nullopt};
  if (auto size = getVeqSize(target); size && *size != 0)
    return StaticQubitTarget{target, sourceIndex, *size - 1};
  return std::nullopt;
}

/// Plan the last scalar target accepted by \p predicate without creating IR.
template <typename Predicate>
inline std::optional<StaticQubitTarget>
findLastStaticQubitTarget(mlir::ValueRange targets, Predicate predicate) {
  for (std::size_t i = targets.size(); i != 0; --i) {
    auto finalTarget = planStaticQubitTarget(targets[i - 1], i - 1);
    if (!finalTarget)
      continue;
    if (!finalTarget->elementIndex) {
      if (predicate(*finalTarget))
        return finalTarget;
      continue;
    }

    for (std::size_t element = *finalTarget->elementIndex + 1; element != 0;
         --element) {
      StaticQubitTarget candidate{finalTarget->source, finalTarget->sourceIndex,
                                  element - 1};
      if (predicate(candidate))
        return candidate;
    }
  }
  return std::nullopt;
}

/// Plan a deterministic final scalar target without creating IR.
std::optional<StaticQubitTarget>
findLastStaticQubitTarget(mlir::ValueRange targets);

/// Materialize a target selected by findLastStaticQubitTarget.
mlir::Value materializeStaticQubitTarget(mlir::OpBuilder &builder,
                                         mlir::Location location,
                                         const StaticQubitTarget &target);

/// Returns true if and only if any quantum operand has type `!quake.ref`.
inline bool hasNonVectorReference(mlir::Operation *op) {
  for (mlir::Value opnd : op->getOperands())
    if (isa<cudaq::quake::RefType>(opnd.getType()))
      return true;
  return false;
}

/// Returns true if and only if all quantum operands do not have type
/// `!quake.wire` or `!quake.control`.
inline bool isAllReferences(mlir::Operation *op) {
  for (mlir::Value opnd : op->getOperands())
    if (isQuantumValueType(opnd.getType()))
      return false;
  return true;
}

/// Returns true if and only if all quantum operands have type `!quake.wire` or
/// `!quake.control`.
inline bool isAllValues(mlir::Operation *op) {
  for (mlir::Value opnd : op->getOperands())
    if (isQuantumReferenceType(opnd.getType()))
      return false;
  return true;
}

/// Returns true if and only if \p op is in the intermediate quantum load/store
/// (QLS) form.
inline bool isWrapped(mlir::Operation *op) {
  for (mlir::Value val : op->getOperands())
    if (isa<cudaq::quake::WireType>(val.getType()) &&
        !val.getDefiningOp<cudaq::quake::UnwrapOp>())
      return false;
  for (mlir::Value val : op->getResults())
    if (isa<cudaq::quake::WireType>(val.getType()))
      for (auto *u : val.getUsers())
        if (!isa<cudaq::quake::WrapOp>(u))
          return false;
  return true;
}

/// Returns true if and only if \p op is fully in linear-value form.
/// Linear-value form is defined such that the Op, \p op, is not in full (or
/// partial) memory-SSA form and is not in the intermediate QLS form.
inline bool isLinearValueForm(mlir::Operation *op) {
  return isa<cudaq::quake::NullWireOp, cudaq::quake::SinkOp>(op) ||
         (isAllValues(op) && !isWrapped(op));
}
inline bool isLinearValueForm(mlir::Value val) {
  if (auto *op = val.getDefiningOp())
    return isLinearValueForm(op);
  return isQuantumValueType(val.getType());
}

template <typename OP>
constexpr bool isMeasure = std::is_same_v<OP, cudaq::quake::MxOp> ||
                           std::is_same_v<OP, cudaq::quake::MyOp> ||
                           std::is_same_v<OP, cudaq::quake::MzOp>;

/// Return true when \p op is a one-target operator for which a `veq` operand
/// in the target position means "apply this operator to every element of the
/// vector". Multi-qubit operators (`swap`, `exp_pauli`, custom unitaries) are
/// excluded: for those a `veq` target is the operand list of a single N-qubit
/// gate, not a broadcast.
inline bool isBroadcastOperator(mlir::Operation *op) {
  return mlir::isa<cudaq::quake::HOp, cudaq::quake::PhasedRxOp,
                   cudaq::quake::R1Op, cudaq::quake::RxOp, cudaq::quake::RyOp,
                   cudaq::quake::RzOp, cudaq::quake::SOp, cudaq::quake::TOp,
                   cudaq::quake::U2Op, cudaq::quake::U3Op, cudaq::quake::XOp,
                   cudaq::quake::YOp, cudaq::quake::ZOp>(op);
}

//===----------------------------------------------------------------------===//
// Control and wire helpers.
//===----------------------------------------------------------------------===//

/// Return true when any control vector cannot be expanded into a statically
/// known number of scalar references.
bool hasUnresolvedControlVeq(mlir::ValueRange controls);

/// Return one polarity per control, where `true` marks a negated control.
/// Controls without an explicit polarity are positive.
llvm::SmallVector<bool>
getControlPolarities(mlir::ValueRange controls,
                     std::optional<llvm::ArrayRef<bool>> negatedControls = {});
llvm::SmallVector<bool> getControlPolarities(OperatorInterface op);

/// The controls and polarities resulting from expanding statically sized
/// vector controls. Controls with unresolved vector sizes remain intact for
/// callers that can lower them without making the predicate scalar.
struct ExpandedControlVeqs {
  llvm::SmallVector<mlir::Value> controls;
  llvm::SmallVector<bool> polarities;
  bool didExpand = false;
};

/// Expand controls with statically known vector sizes into scalar references,
/// including vectors whose known size is visible through RelaxSizeOp. Unknown
/// vector controls are preserved for callers that support them.
ExpandedControlVeqs
expandKnownSizedControlVeqs(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::ValueRange controls,
                            llvm::ArrayRef<bool> polarities);

/// Return the wire result types for a Quake operator with the given controls
/// and targets. Quake orders wire results by controls first, then targets.
llvm::SmallVector<mlir::Type> getWireResultTypes(mlir::OpBuilder &builder,
                                                 mlir::ValueRange controls,
                                                 mlir::ValueRange targets);

/// Collect the threaded values of a Quake operator's controls and targets in
/// its wire-result order.
llvm::SmallVector<mlir::Value> getWireValues(mlir::ValueRange controls,
                                             mlir::ValueRange targets);

/// Update controls and targets to the corresponding wire results of the
/// newly created operator op. The ranges must hold the values op was
/// created with.
void threadWireResults(OperatorInterface op,
                       llvm::MutableArrayRef<mlir::Value> controls,
                       llvm::MutableArrayRef<mlir::Value> targets);

/// Create a Quake gate and update controls and targets to its latest wire
/// results. Reference operands are returned unchanged.
template <typename Op>
inline Op createAndThreadGate(mlir::OpBuilder &builder, mlir::Location location,
                              mlir::UnitAttr isAdj, mlir::ValueRange parameters,
                              llvm::MutableArrayRef<mlir::Value> controls,
                              llvm::MutableArrayRef<mlir::Value> targets,
                              mlir::DenseBoolArrayAttr negatedControls = {}) {
  auto resultTypes = getWireResultTypes(builder, controls, targets);
  auto op = Op::create(builder, location, resultTypes, isAdj, parameters,
                       controls, targets, negatedControls);
  threadWireResults(op, controls, targets);
  return op;
}

/// used to unwrap `!quake.control` from `quake.from_control`
inline mlir::Value unwrapFromControlVal(mlir::Value value) {
  while (auto fromControl = value.getDefiningOp<cudaq::quake::FromControlOp>())
    value = fromControl.getCtrlbit();
  return value;
}

/// take input `veq` and find it's defining op
inline mlir::Value getKnownAllocaVeq(mlir::Value veq) {
  if (auto relax = veq.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    veq = relax.getInputVec();

  if (!veq.getDefiningOp<cudaq::quake::AllocaOp>())
    return {};

  return veq;
}

} // namespace cudaq::quake
