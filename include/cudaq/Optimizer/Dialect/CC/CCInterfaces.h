/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Common/InlinerInterface.h"

using CCInlinerInterface = cudaq::EnableInlinerInterface;

namespace cudaq::cc {

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
