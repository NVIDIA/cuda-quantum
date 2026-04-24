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

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace cudaq {
inline mlir::Type getQuantumTypeByName(mlir::StringRef type,
                                       mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getOpaque(type, context);
}

namespace opt {
void initializeTypeConversions(mlir::LLVMTypeConverter &typeConverter);
} // namespace opt

namespace cg {

// These type creators are configurable on the fly to either use opaque structs
// or opaque pointers. The default is to use opaque pointers, which are the
// default in any modern LLVM version.

inline mlir::Type getOpaquePointerType(mlir::MLIRContext *context) {
  return cc::PointerType::get(mlir::NoneType::get(context));
}

inline mlir::Type getQubitType(mlir::MLIRContext *context,
                               bool useOpaquePtr = true) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(getQuantumTypeByName("Qubit", context));
}

inline mlir::Type getArrayType(mlir::MLIRContext *context,
                               bool useOpaquePtr = true) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(getQuantumTypeByName("Array", context));
}

inline mlir::Type getResultType(mlir::MLIRContext *context,
                                bool useOpaquePtr = true) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(getQuantumTypeByName("Result", context));
}

inline mlir::Type getCharPointerType(mlir::MLIRContext *context,
                                     bool useOpaquePtr = true) {
  if (useOpaquePtr)
    return getOpaquePointerType(context);
  return cc::PointerType::get(mlir::IntegerType::get(context, 8));
}

// LLVM Types:
// The factory builder will build opaque pointers for modern MLIR.

inline mlir::Type getLLVMQubitType(mlir::MLIRContext *context) {
  return opt::factory::getPointerType(getQuantumTypeByName("Qubit", context));
}

inline mlir::Type getLLVMArrayType(mlir::MLIRContext *context) {
  return opt::factory::getPointerType(getQuantumTypeByName("Array", context));
}

inline mlir::Type getLLVMResultType(mlir::MLIRContext *context) {
  return opt::factory::getPointerType(getQuantumTypeByName("Result", context));
}

inline mlir::Type getLLVMCharPointerType(mlir::MLIRContext *context) {
  return opt::factory::getPointerType(mlir::IntegerType::get(context, 8));
}

} // namespace cg
} // namespace cudaq
