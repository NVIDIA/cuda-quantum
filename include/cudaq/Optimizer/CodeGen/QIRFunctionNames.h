/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// This file provides some common QIR function names for use throughout our
/// MLIR lowering infrastructure.

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace cudaq::opt {

/// QIS Function name strings
static constexpr const char QIRQISPrefix[] = "__quantum__qis__";
static constexpr const char QIRMeasureBody[] = "__quantum__qis__mz__body";
static constexpr const char QIRMeasure[] = "__quantum__qis__mz";
static constexpr const char QIRMeasureToRegister[] =
    "__quantum__qis__mz__to__register";

static constexpr const char QIRCnot[] = "__quantum__qis__cnot";
static constexpr const char QIRCphase[] = "__quantum__qis__cphase";
static constexpr const char QIRReadResultBody[] =
    "__quantum__qis__read_result__body";

static constexpr const char QIRCustomOp[] = "__quantum__qis__custom_unitary";

static constexpr const char NVQIRInvokeWithControlBits[] =
    "invokeWithControlQubits";
static constexpr const char NVQIRInvokeRotationWithControlBits[] =
    "invokeRotationWithControlQubits";
static constexpr const char NVQIRInvokeU3RotationWithControlBits[] =
    "invokeU3RotationWithControlQubits";
static constexpr const char NVQIRInvokeWithControlRegisterOrBits[] =
    "invokeWithControlRegisterOrQubits";
static constexpr const char NVQIRPackSingleQubitInArray[] =
    "packSingleQubitInArray";
static constexpr const char NVQIRReleasePackedQubitArray[] =
    "releasePackedQubitArray";

/// QIR Array function name strings
static constexpr const char QIRArrayGetElementPtr1d[] =
    "__quantum__rt__array_get_element_ptr_1d";
static constexpr const char QIRArrayQubitAllocateArray[] =
    "__quantum__rt__qubit_allocate_array";
static constexpr const char QIRArrayQubitAllocateArrayWithStateFP64[] =
    "__quantum__rt__qubit_allocate_array_with_state_fp64";
static constexpr const char QIRArrayQubitAllocateArrayWithStateFP32[] =
    "__quantum__rt__qubit_allocate_array_with_state_fp32";
static constexpr const char QIRArrayQubitAllocateArrayWithStateComplex64[] =
    "__quantum__rt__qubit_allocate_array_with_state_complex64";
static constexpr const char QIRArrayQubitAllocateArrayWithStateComplex32[] =
    "__quantum__rt__qubit_allocate_array_with_state_complex32";
static constexpr const char QIRArrayQubitAllocateArrayWithStatePtr[] =
    "__quantum__rt__qubit_allocate_array_with_state_ptr";
static constexpr const char QIRArrayQubitAllocateArrayWithCudaqStatePtr[] =
    "__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr";
static constexpr const char QIRQubitAllocate[] =
    "__quantum__rt__qubit_allocate";
static constexpr const char QIRArrayQubitReleaseArray[] =
    "__quantum__rt__qubit_release_array";
static constexpr const char QIRArrayQubitReleaseQubit[] =
    "__quantum__rt__qubit_release";
static constexpr const char QIRArraySlice[] = "__quantum__rt__array_slice";
static constexpr const char QIRArrayGetSize[] =
    "__quantum__rt__array_get_size_1d";
static constexpr const char QIRArrayConcatArray[] =
    "__quantum__rt__array_concatenate";
static constexpr const char QIRArrayCreateArray[] =
    "__quantum__rt__array_create_1d";

/// QIR Base/Adaptive Profile record output function names
static constexpr const char QIRRecordOutput[] =
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
