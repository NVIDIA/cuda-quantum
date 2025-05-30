/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Common/Traits.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

//===----------------------------------------------------------------------===//
// Canonicalizer functions.
//===----------------------------------------------------------------------===//

namespace quake {
mlir::Value createConstantAlloca(mlir::PatternRewriter &builder,
                                 mlir::Location loc, mlir::OpResult result,
                                 mlir::ValueRange args);

void getResetEffectsImpl(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects,
    mlir::ValueRange targets);
void getMeasurementEffectsImpl(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects,
    mlir::ValueRange targets);
void getOperatorEffectsImpl(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects,
    mlir::ValueRange controls, mlir::ValueRange targets);

mlir::ParseResult genericOpParse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result);
void genericOpPrinter(mlir::OpAsmPrinter &_odsPrinter, mlir::Operation *op,
                      bool isAdj, mlir::OperandRange params,
                      mlir::OperandRange ctrls, mlir::OperandRange targs,
                      mlir::DenseBoolArrayAttr negatedQubitControlsAttr);
} // namespace quake

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

namespace quake {
/// Returns true if and only if any quantum operand has type `!quake.ref` or
/// `!quake.veq`.
inline bool hasReference(mlir::Operation *op) {
  for (mlir::Value opnd : op->getOperands())
    if (isQuantumReferenceType(opnd.getType()))
      return true;
  return false;
}

/// Returns true if and only if any quantum operand has type `!quake.ref`.
inline bool hasNonVectorReference(mlir::Operation *op) {
  for (mlir::Value opnd : op->getOperands())
    if (isa<quake::RefType>(opnd.getType()))
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
    if (isa<quake::WireType>(val.getType()) &&
        !val.getDefiningOp<quake::UnwrapOp>())
      return false;
  for (mlir::Value val : op->getResults())
    if (isa<quake::WireType>(val.getType()))
      for (auto *u : val.getUsers())
        if (!isa<quake::WrapOp>(u))
          return false;
  return true;
}

/// Returns true if and only if \p op is fully in linear-value form.
/// Linear-value form is defined such that the Op, \p op, is not in full (or
/// partial) memory-SSA form and is not in the intermediate QLS form.
inline bool isLinearValueForm(mlir::Operation *op) {
  return isa<quake::NullWireOp, quake::SinkOp>(op) ||
         (isAllValues(op) && !isWrapped(op));
}
inline bool isLinearValueForm(mlir::Value val) {
  if (auto *op = val.getDefiningOp())
    return isLinearValueForm(op);
  return isQuantumValueType(val.getType());
}

template <typename OP>
constexpr bool isMeasure =
    std::is_same_v<OP, quake::MxOp> || std::is_same_v<OP, quake::MyOp> ||
    std::is_same_v<OP, quake::MzOp>;

} // namespace quake
