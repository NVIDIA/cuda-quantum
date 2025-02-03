/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// This file provides the opaque struct types to be used with the obsolete LLVM
/// typed pointer type.

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace cudaq {
inline mlir::Type getQuantumTypeByName(mlir::StringRef type,
                                       mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getOpaque(type, context);
}

namespace opt {

// The following type creators are deprecated and should only be used in the
// older codegen passes. Use the creators in the cg namespace immediately below
// instead.
inline mlir::Type getOpaquePointerType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(context);
}

inline mlir::Type getQubitType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(
      getQuantumTypeByName("Qubit", context));
}

inline mlir::Type getArrayType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(
      getQuantumTypeByName("Array", context));
}

inline mlir::Type getResultType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(
      getQuantumTypeByName("Result", context));
}

inline mlir::Type getCharPointerType(mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
}

void initializeTypeConversions(mlir::LLVMTypeConverter &typeConverter);

} // namespace opt

namespace cg {

// The following type creators replace the ones above. They are configurable on
// the fly to either use opaque structs or opaque pointers. The default is to
// use pointers to opaque structs, which is no longer supported in modern LLVM.

inline mlir::Type getOpaquePointerType(mlir::MLIRContext *context) {
  return cc::PointerType::get(mlir::NoneType::get(context));
}

inline mlir::Type getQubitType(mlir::MLIRContext *context,
                               bool useOpaquePtr = false) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(getQuantumTypeByName("Qubit", context));
}

inline mlir::Type getArrayType(mlir::MLIRContext *context,
                               bool useOpaquePtr = false) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(getQuantumTypeByName("Array", context));
}

inline mlir::Type getResultType(mlir::MLIRContext *context,
                                bool useOpaquePtr = false) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(getQuantumTypeByName("Result", context));
}

inline mlir::Type getCharPointerType(mlir::MLIRContext *context,
                                     bool useOpaquePtr = false) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(mlir::IntegerType::get(context, 8));
}

} // namespace cg
} // namespace cudaq
