/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/InlinerInterface.h"
#include "mlir/IR/Types.h"

using CCInlinerInterface = cudaq::EnableInlinerInterface;

namespace cudaq::cc {

/// A linear type is any type that carries the `SSIType` trait. CC does not know
/// which dialects define such types; Quake's wire and cable are two of them.
inline bool isLinearType(mlir::Type ty) {
  return ty.hasTrait<mlir::TypeTrait::SSIType>();
}

mlir::LogicalResult verifyConvergentLinearTypesInRegions(mlir::Operation *op);

template <typename ConcreteType>
class LinearTypeArgsTrait
    : public mlir::OpTrait::TraitBase<ConcreteType, LinearTypeArgsTrait> {
public:
  static mlir::LogicalResult verifyRegionTrait(mlir::Operation *op) {
    return verifyConvergentLinearTypesInRegions(op);
  }
};
} // namespace cudaq::cc

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#include "cudaq/Optimizer/Dialect/CC/CCInterfaces.h.inc"
