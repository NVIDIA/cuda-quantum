/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include <optional>

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h.inc"

namespace cudaq::quake {
/// A single factor in a Pauli tensor product.
enum class Pauli { I, X, Y, Z };

/// An ordered Pauli tensor-product word.
using PauliWord = llvm::SmallVector<Pauli>;

/// Convert a word containing only `I`, `X`, `Y`, and `Z` to Pauli symbols.
std::optional<PauliWord> symbolizePauliWord(llvm::StringRef value);

/// \returns true if \p `ty` is a quantum value or reference.
inline bool isQuantumType(mlir::Type ty) {
  // NB: this intentionally excludes MeasureType.
  return mlir::isa<cudaq::quake::RefType, cudaq::quake::VeqType,
                   cudaq::quake::WireType, cudaq::quake::ControlType,
                   cudaq::quake::StruqType, cudaq::quake::CableType>(ty);
}

/// \returns true if \p `ty` is a Quake type.
inline bool isQuakeType(mlir::Type ty) {
  // This should correspond to the registered types in QuakeTypes.cpp.
  return isQuantumType(ty) || mlir::isa<cudaq::quake::MeasureType>(ty);
}

/// \returns true if \p ty is a quantum reference type, excluding `struq`.
inline bool isNonStruqReferenceType(mlir::Type ty) {
  return mlir::isa<cudaq::quake::RefType, cudaq::quake::VeqType>(ty);
}

/// \returns true if \p ty is a quantum reference type.
inline bool isQuantumReferenceType(mlir::Type ty) {
  return isNonStruqReferenceType(ty) || mlir::isa<cudaq::quake::StruqType>(ty);
}

/// Quake's wire and cable types are linear types.
inline bool isLinearType(mlir::Type ty) {
  return mlir::isa<cudaq::quake::WireType, cudaq::quake::CableType>(ty);
}

/// All linear types and the ControlType are quantum value types.
inline bool isQuantumValueType(mlir::Type ty) {
  return isLinearType(ty) || mlir::isa<cudaq::quake::ControlType>(ty);
}

/// \returns true if and only if \p ty is a reference type and it has a constant
/// number of quantum references.
bool isConstantQuantumRefType(mlir::Type ty);

/// Get the number of references in \p ty. \p ty must be a reference type.
std::size_t getAllocationSize(mlir::Type ty);

/// Get the number of wires in \p ty. \p ty must be a value type.
inline std::size_t getWireCount(mlir::Type ty) {
  if (isa<cudaq::quake::WireType, cudaq::quake::ControlType>(ty))
    return 1;
  return cast<cudaq::quake::CableType>(ty).getSize();
}

} // namespace cudaq::quake
