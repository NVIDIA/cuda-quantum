/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/SmallVector.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h.inc"

namespace quake {
/// \returns true if \p `ty` is a quantum value or reference.
inline bool isaQuantumType(mlir::Type ty) {
  // NB: this intentionally excludes MeasureType.
  return mlir::isa<quake::RefType, quake::VeqType, quake::WireType,
                   quake::ControlType>(ty);
}

/// \returns true if \p `ty` is a Quake type.
inline bool isQuakeType(mlir::Type ty) {
  // This should correspond to the registered types in QuakeTypes.cpp.
  return isaQuantumType(ty) || mlir::isa<quake::MeasureType>(ty);
}

/// A quake value type is a linear type.
inline bool isLinearType(mlir::Type ty) {
  return mlir::isa<quake::WireType>(ty);
}

} // namespace quake
