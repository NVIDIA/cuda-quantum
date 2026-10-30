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

/// Read the quantum optimization level attribute, if any, from an operation
/// (module or function).
inline std::optional<unsigned>
getOptimizationLevel(mlir::Operation *op) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(optimizationLevelAttrName);
  if (!attr)
    return std::nullopt;
  return static_cast<unsigned>(attr.getInt());
}

/// Set the quantum optimization level attribute (an `i32` from 0 to 3) on an
/// operation (module or function).
inline void setOptimizationLevel(mlir::Operation *op, unsigned level) {
  op->setAttr(optimizationLevelAttrName,
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(op->getContext(), 32), level));
}

/// Whether the operation is marked for quantum optimization level 0, i.e.
/// quantum optimization passes must skip it.
inline bool isQuantumOptimizationDisabled(mlir::Operation *op) {
  auto level = getOptimizationLevel(op);
  return level && *level == 0;
}

} // namespace cudaq::runtime
