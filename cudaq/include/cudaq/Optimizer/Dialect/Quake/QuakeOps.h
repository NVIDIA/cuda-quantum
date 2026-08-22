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
#include <optional>

//===----------------------------------------------------------------------===//
// Canonicalizer functions.
//===----------------------------------------------------------------------===//

namespace cudaq::quake {
constexpr const char ancillaAttrName[] = "quake.ancilla";

/// Marks \p op as allocating `ancillas`. See \a ancillaAttrName.
inline void markAsAncilla(mlir::Operation *op) {
  op->setAttr(ancillaAttrName, mlir::UnitAttr::get(op->getContext()));
}

/// Returns true if \p op is an allocation marked as an ancilla.
inline bool isAncilla(mlir::Operation *op) {
  return op && op->hasAttr(ancillaAttrName);
}

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

} // namespace cudaq::quake
