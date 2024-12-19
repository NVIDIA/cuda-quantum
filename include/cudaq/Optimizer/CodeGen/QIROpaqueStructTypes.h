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

namespace cudaq::opt {

inline mlir::Type getQuantumTypeByName(mlir::StringRef type,
                                       mlir::MLIRContext *context) {
  return mlir::LLVM::LLVMStructType::getOpaque(type, context);
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

} // namespace cudaq::opt
