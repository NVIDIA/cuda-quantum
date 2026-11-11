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

#include "qlx/Interfaces/LinearOwnershipInterfaces.h.inc"

namespace qlx {

/// See QLX_LinearOwnershipScope in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearOwnershipScope
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearOwnershipScope> {};

/// See QLX_LinearBorrowsOperands in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearBorrowsOperands
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearBorrowsOperands> {};

/// See QLX_LinearExclusiveRegions in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearExclusiveRegions
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearExclusiveRegions> {};

/// See QLX_LinearRepeatedRegions in LinearOwnershipInterfaces.td.
template <typename ConcreteType>
class LinearRepeatedRegions
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearRepeatedRegions> {};

/// True when values of `type` are single-owner linear values.
inline bool isLinearType(mlir::Type type) {
  auto linear = mlir::dyn_cast<LinearTypeInterface>(type);
  return linear && linear.isLinear();
}

} // namespace qlx
