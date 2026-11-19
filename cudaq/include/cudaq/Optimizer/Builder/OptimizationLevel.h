/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include <optional>

namespace cudaq::runtime {

/// Read the quantum optimization level attribute, if any, from a function.
inline std::optional<unsigned> getOptimizationLevel(mlir::Operation *op) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(optimizationLevelAttrName);
  if (!attr || attr.getInt() < 0 || attr.getInt() > 3)
    return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}

/// Set the quantum optimization level attribute on a function.
inline void setOptimizationLevel(mlir::Operation *op, unsigned level) {
  op->setAttr(optimizationLevelAttrName,
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32), level));
}

/// Whether the function is marked with quantum optimization level 0.
inline bool isQuantumOptimizationDisabled(mlir::Operation *op) {
  auto level = getOptimizationLevel(op);
  return level && *level == 0;
}

} // namespace cudaq::runtime
