/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

namespace cudaq::opt::marshal {

/// This value is used to indicate that a kernel does not return a result.
static constexpr std::uint64_t NoResultOffset =
    std::numeric_limits<std::int32_t>::max();

/// Generate code for packing arguments as raw data.
inline bool isCodegenPackedData(std::size_t kind) {
  return kind == 0 || kind == 1;
}

/// Generate code that gathers the arguments for conversion and synthesis.
inline bool isCodegenArgumentGather(std::size_t kind) {
  return kind == 0 || kind == 2;
}

inline bool isStateType(mlir::Type ty) {
  if (auto ptrTy = dyn_cast<cc::PointerType>(ty))
    return isa<quake::StateType>(ptrTy.getElementType());
  return false;
}

/// Creates the function signature for a thunk function. The signature is always
/// the same for all thunk functions.
///
/// Every thunk function has an identical signature, making it callable from a
/// generic "kernel launcher" in the CUDA-Q runtime.
///
/// This signature is defined as: `(ptr, bool) -> {ptr, i64}`.
///
/// The first argument is a pointer to a data buffer that encodes all the
/// arguments (and static return) values to (and from) the kernel in the
/// pointer-free encoding. The second argument indicates if this call is to a
/// remote process (if true). The result is a pointer and size (span) if the
/// kernel returns a dynamically sized result, otherwise it will be
/// `{nullptr, 0}`. It is the responsibility of calling code to free any
/// dynamic result buffer(s) and convert those to `std::vector` objects.
inline mlir::FunctionType getThunkType(mlir::MLIRContext *ctx) {
  auto ptrTy = cc::PointerType::get(mlir::IntegerType::get(ctx, 8));
  return mlir::FunctionType::get(ctx, {ptrTy, mlir::IntegerType::get(ctx, 1)},
                                 {opt::factory::getDynamicBufferType(ctx)});
}

mlir::Value genComputeReturnOffset(mlir::Location loc, mlir::OpBuilder &builder,
                                   mlir::FunctionType funcTy,
                                   cc::StructType msgStructTy);

/// Create a function that determines the return value offset in the message
/// buffer.
void genReturnOffsetFunction(mlir::Location loc, mlir::OpBuilder &builder,
                             mlir::FunctionType devKernelTy,
                             cc::StructType msgStructTy,
                             const std::string &classNameStr);

cc::PointerType getPointerToPointerType(mlir::OpBuilder &builder);

bool isDynamicSignature(mlir::FunctionType devFuncTy);

std::pair<mlir::Value, bool>
unpackAnyStdVectorBool(mlir::Location loc, mlir::OpBuilder &builder,
                       mlir::ModuleOp module, mlir::Value arg, mlir::Type ty,
                       mlir::Value heapTracker);

mlir::Value genSizeOfDynamicMessageBuffer(
    mlir::Location loc, mlir::OpBuilder &builder, mlir::ModuleOp module,
    cc::StructType structTy,
    mlir::ArrayRef<std::tuple<unsigned, mlir::Value, mlir::Type>> zippy,
    mlir::Value tmp);

mlir::Value genSizeOfDynamicCallbackBuffer(
    mlir::Location loc, mlir::OpBuilder &builder, mlir::ModuleOp module,
    cc::StructType structTy,
    mlir::ArrayRef<std::tuple<unsigned, mlir::Value, mlir::Type>> zippy,
    mlir::Value tmp);

void populateMessageBuffer(
    mlir::Location loc, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Value msgBufferBase,
    mlir::ArrayRef<std::tuple<unsigned, mlir::Value, mlir::Type>> zippy,
    mlir::Value addendum = {}, mlir::Value addendumScratch = {});

void populateCallbackBuffer(
    mlir::Location loc, mlir::OpBuilder &builder, mlir::ModuleOp module,
    mlir::Value msgBufferBase,
    mlir::ArrayRef<std::tuple<unsigned, mlir::Value, mlir::Type>> zippy,
    mlir::Value addendum = {}, mlir::Value addendumScratch = {});

/// A kernel function that takes a quantum type argument (also known as a pure
/// device kernel) cannot be called directly from C++ (classical) code. It must
/// be called via other quantum code.
bool hasLegalType(mlir::FunctionType funTy);

mlir::MutableArrayRef<mlir::BlockArgument>
dropAnyHiddenArguments(mlir::MutableArrayRef<mlir::BlockArgument> args,
                       mlir::FunctionType funcTy, bool hasThisPointer);

std::pair<bool, mlir::func::FuncOp>
lookupHostEntryPointFunc(mlir::StringRef mangledEntryPointName,
                         mlir::ModuleOp module, mlir::func::FuncOp funcOp);

/// Generate code to initialize the std::vector<T>, \p sret, from an initializer
/// list with data at \p data and length \p size. Use the library helper
/// routine. This function takes two !llvm.ptr arguments.
void genStdvecBoolFromInitList(mlir::Location loc, mlir::OpBuilder &builder,
                               mlir::Value sret, mlir::Value data,
                               mlir::Value size);

/// Generate a `std::vector<T>` (where `T != bool`) from an initializer list.
/// This is done with the assumption that `std::vector` is implemented as a
/// triple of pointers. The original content of the vector is freed and the new
/// content, which is already on the stack, is moved into the `std::vector`.
void genStdvecTFromInitList(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::Value sret, mlir::Value data,
                            mlir::Value tSize, mlir::Value vecSize);

// Alloca a pointer to a pointer and initialize it to nullptr.
mlir::Value createEmptyHeapTracker(mlir::Location loc,
                                   mlir::OpBuilder &builder);

// If there are temporaries, call the helper to free them.
void maybeFreeHeapAllocations(mlir::Location loc, mlir::OpBuilder &builder,
                              mlir::Value heapTracker);

/// Translate the buffer data to a sequence of arguments suitable to the
/// actual kernel call.
///
/// \param inTy      The actual expected type of the argument.
/// \param structTy  The modified buffer type over all the arguments at the
/// current level.
std::pair<mlir::Value, mlir::Value>
processInputValue(mlir::Location loc, mlir::OpBuilder &builder,
                  mlir::Value trailingData, mlir::Value ptrPackedStruct,
                  mlir::Type inTy, std::int32_t off,
                  cc::StructType packedStructTy);

std::pair<mlir::Value, mlir::Value>
processCallbackInputValue(mlir::Location loc, mlir::OpBuilder &builder,
                          mlir::Value trailingData, mlir::Value ptrPackedStruct,
                          mlir::Type inTy, std::int32_t off,
                          cc::StructType packedStructTy);

} // namespace cudaq::opt::marshal
