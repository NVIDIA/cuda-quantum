/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "cudaq/Optimizer/Dialect/LinearOwnershipInterfaces.h.inc"

namespace cudaq {

/// See LinearOwnershipScope in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearOwnershipScope
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearOwnershipScope> {};

/// See LinearBorrowsOperands in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearBorrowsOperands
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearBorrowsOperands> {};

/// See LinearExclusiveRegions in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearExclusiveRegions
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearExclusiveRegions> {};

/// See LinearRepeatedRegions in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearRepeatedRegions
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearRepeatedRegions> {};

/// True when values of \p type are single-owner linear values, i.e. the type
/// implements LinearTypeInterface and reports isLinear(). Named differently
/// from `cudaq::quake::isLinearType` so the two cannot collide in a scope
/// that uses both namespaces.
inline bool isLinearOwnershipType(mlir::Type type) {
  auto linear = mlir::dyn_cast<LinearTypeInterface>(type);
  return linear && linear.isLinear();
}

} // namespace cudaq
