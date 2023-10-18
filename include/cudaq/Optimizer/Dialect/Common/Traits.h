/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/OpImplementation.h"

namespace quake {
mlir::LogicalResult verifyWireArityAndCoarity(mlir::Operation *op);

/// Returns true iff \p op is a Quantum Operator (unitary), measurement, or a
/// sink.
bool isSupportedMappingOperation(mlir::Operation *op);

/// Return the subset of a range that is `quake.wire` or `quake.ref`. That is -
/// it strips classical parameters off the beginning of the range.
mlir::ValueRange getWiresFromRange(mlir::ValueRange range);

/// Returns the operands from \p op that are not classical parameters.
mlir::ValueRange getWireOperands(mlir::Operation *op);

/// Returns the results from \p op that are not classical parameters.
mlir::ValueRange getWireResults(mlir::Operation *op);

/// Set the operands from \p op from \wires.
mlir::LogicalResult setWireOperands(mlir::Operation *op,
                                    mlir::ValueRange wires);
} // namespace quake

namespace cudaq {

template <typename ConcreteType>
class Hermitian : public mlir::OpTrait::TraitBase<ConcreteType, Hermitian> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    if (cast<ConcreteType>(op).isAdj())
      return op->emitOpError("may not be adjoint");
    return mlir::success();
  }
};

template <typename ConcreteType>
class QuantumGate : public mlir::OpTrait::TraitBase<ConcreteType, QuantumGate> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return quake::verifyWireArityAndCoarity(op);
  }
};

template <typename ConcreteType>
class Rotation : public mlir::OpTrait::TraitBase<ConcreteType, Rotation> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return mlir::success();
  }
};

template <typename ConcreteType>
class QuantumMeasure
    : public mlir::OpTrait::TraitBase<ConcreteType, QuantumMeasure> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return quake::verifyWireArityAndCoarity(op);
  }
};

template <typename ConcreteType>
class JumpWithUnwind
    : public mlir::OpTrait::TraitBase<ConcreteType, JumpWithUnwind> {
public:
  static mlir::LogicalResult verifyTrait(mlir::Operation *op) {
    return mlir::success();
  }
};

} // namespace cudaq
