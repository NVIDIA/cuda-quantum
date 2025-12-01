/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include "mlir/Interfaces/DataLayoutInterfaces.h"

namespace cudaq::cc {

class SpanLikeType : public mlir::Type {
public:
  using mlir::Type::Type;
  mlir::Type getElementType() const;
  static bool classof(mlir::Type type);
};

} // namespace cudaq::cc

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h.inc"

namespace cudaq::cc {

inline bool SpanLikeType::classof(mlir::Type type) {
  return mlir::isa<StdvecType, CharspanType>(type);
}

/// Returns true if and only if \p ty has dynamic extent. This is a recursive
/// test on composable types.
bool isDynamicType(mlir::Type ty);

/// Returns true if and only if the memory needed to store a value of type
/// \p ty is not known at compile time. This is a recursive test on composable
/// types. In contrast to `isDynamicType`, the size of the type is statically
/// known even if it contains pointers that may point to memory of dynamic size.
bool isDynamicallySizedType(mlir::Type ty);

/// Determine the number of hidden arguments, which is 0, 1, or 2.
inline unsigned numberOfHiddenArgs(bool thisPtr, bool sret) {
  return (thisPtr ? 1 : 0) + (sret ? 1 : 0);
}

// Checks if type is device_ptr.
bool isDevicePtr(mlir::Type argTy);

} // namespace cudaq::cc
