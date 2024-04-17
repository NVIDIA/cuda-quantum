/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// This file provides some common QIR function name string s
/// for use throughout our MLIR lowering infrastructure.

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace cudaq::opt {

/// QIS Function name strings
constexpr static const char QIRQISPrefix[] = "__quantum__qis__";
constexpr static const char QIRMeasureBody[] = "__quantum__qis__mz__body";
constexpr static const char QIRMeasure[] = "__quantum__qis__mz";
constexpr static const char QIRMeasureToRegister[] =
    "__quantum__qis__mz__to__register";

constexpr static const char QIRCnot[] = "__quantum__qis__cnot";
constexpr static const char QIRCphase[] = "__quantum__qis__cphase";
constexpr static const char QIRReadResultBody[] =
    "__quantum__qis__read_result__body";

constexpr static const char NVQIRInvokeWithControlBits[] =
    "invokeWithControlQubits";
constexpr static const char NVQIRInvokeRotationWithControlBits[] =
    "invokeRotationWithControlQubits";
constexpr static const char NVQIRInvokeWithControlRegisterOrBits[] =
    "invokeWithControlRegisterOrQubits";
constexpr static const char NVQIRPackSingleQubitInArray[] =
    "packSingleQubitInArray";
constexpr static const char NVQIRReleasePackedQubitArray[] =
    "releasePackedQubitArray";

/// QIR Array function name strings
constexpr static const char QIRArrayGetElementPtr1d[] =
    "__quantum__rt__array_get_element_ptr_1d";
constexpr static const char QIRArrayQubitAllocateArray[] =
    "__quantum__rt__qubit_allocate_array";
constexpr static const char QIRArrayQubitAllocateArrayWithStateFP64[] =
    "__quantum__rt__qubit_allocate_array_with_state_fp64";
constexpr static const char QIRArrayQubitAllocateArrayWithStateFP32[] =
    "__quantum__rt__qubit_allocate_array_with_state_fp32";
constexpr static const char QIRQubitAllocate[] =
    "__quantum__rt__qubit_allocate";
constexpr static const char QIRArrayQubitReleaseArray[] =
    "__quantum__rt__qubit_release_array";
constexpr static const char QIRArrayQubitReleaseQubit[] =
    "__quantum__rt__qubit_release";
constexpr static const char QIRArraySlice[] = "__quantum__rt__array_slice";
constexpr static const char QIRArrayGetSize[] =
    "__quantum__rt__array_get_size_1d";
constexpr static const char QIRArrayConcatArray[] =
    "__quantum__rt__array_concatenate";
constexpr static const char QIRArrayCreateArray[] =
    "__quantum__rt__array_create_1d";

/// QIR Base/Adaptive Profile record output function names
constexpr static const char QIRRecordOutput[] =
    "__quantum__rt__result_record_output";

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
