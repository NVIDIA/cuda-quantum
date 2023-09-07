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
}

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
