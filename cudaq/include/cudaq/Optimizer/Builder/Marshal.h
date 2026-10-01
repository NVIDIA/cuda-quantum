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
    return isa<cudaq::quake::StateType>(ptrTy.getElementType());
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
unpackAnySequenceBool(mlir::Location loc, mlir::OpBuilder &builder,
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
void genSequenceBoolFromInitList(mlir::Location loc, mlir::OpBuilder &builder,
                                 mlir::Value sret, mlir::Value data,
                                 mlir::Value size);

/// Generate a `std::vector<T>` (where `T != bool`) from an initializer list.
/// This is done with the assumption that `std::vector` is implemented as a
/// triple of pointers. The original content of the vector is freed and the new
/// content, which is already on the stack, is moved into the `std::vector`.
void genSequenceTFromInitList(mlir::Location loc, mlir::OpBuilder &builder,
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
/// \param module    The module being transformed. It is passed explicitly since
/// \p builder's insertion point may be in a region not yet attached to it.
/// \param inTy      The actual expected type of the argument.
/// \param structTy  The modified buffer type over all the arguments at the
/// current level.
std::pair<mlir::Value, mlir::Value>
processInputValue(mlir::Location loc, mlir::OpBuilder &builder,
                  mlir::ModuleOp module, mlir::Value trailingData,
                  mlir::Value ptrPackedStruct, mlir::Type inTy,
                  std::int32_t off, cc::StructType packedStructTy);

std::pair<mlir::Value, mlir::Value>
processCallbackInputValue(mlir::Location loc, mlir::OpBuilder &builder,
                          mlir::ModuleOp module, mlir::Value trailingData,
                          mlir::Value ptrPackedStruct, mlir::Type inTy,
                          std::int32_t off, cc::StructType packedStructTy);

/// Given a pointer to a real host side value, build the real device-side SSA
/// value (recursively, for nested dynamic types) it corresponds to. Used to
/// reduce a callback's real host return value into a device-side value suitable
/// for storing into a returning host-to-QPU communication buffer.
///
/// The value may refer to heap memory allocated with `malloc` (for a
/// recursively dynamic type), so it is valid after the frame it was built in
/// has returned.
///
/// If \p ownResult is false, the value refers to the storage of the host value.
/// That value must not be destroyed while the result is in use. If
/// \p ownResult is true, the result is a deep copy in `malloc` memory of its
/// own, which the caller must free, and the host value may be destroyed. In
/// that case the `__nvqpp_vectorCopyCtor` intrinsic must be loaded in
/// \p module.
mlir::Value reduceHostToDeviceValue(mlir::Location loc,
                                    mlir::OpBuilder &builder,
                                    mlir::ModuleOp module, mlir::Type devTy,
                                    mlir::Value hostPtr,
                                    bool ownResult = false);

/// Copy the dynamic parts of the device-side value \p val to the heap,
/// recursively, and return the new value. Used for a result that refers to
/// memory that will not outlive the function that returns it. The heap storage
/// is the responsibility of the calling side. The `__nvqpp_vectorCopyCtor` and
/// `malloc` intrinsics must be loaded in the module.
mlir::Value copyDynamicValueToHeap(mlir::Location loc, mlir::OpBuilder &builder,
                                   mlir::Value val);

/// The calling side of `copyDynamicValueToHeap`. Copy the dynamic parts of the
/// device-side value \p val, whose storage is on the heap, to the stack of the
/// current function, recursively, and free the heap storage. Returns the new
/// value. The `__nvqpp_vectorCopyToStack` and `free` intrinsics must be loaded
/// in the module.
mlir::Value copyDynamicValueToStack(mlir::Location loc,
                                    mlir::OpBuilder &builder, mlir::Value val);

/// Destroy every `std::vector<bool>` in the real host-ABI argument that
/// \p hostPtr points to, where the argument has device type \p devTy. The
/// callback side builds a real `std::vector<bool>` for such an argument, which
/// owns its storage. The other parts of an argument refer to the storage of the
/// communication buffer, and are left alone. The `__nvqpp_vector_bool_destroy`
/// intrinsic must be loaded in \p module.
void destroyHostBoolVectors(mlir::Location loc, mlir::OpBuilder &builder,
                            mlir::ModuleOp module, mlir::Type devTy,
                            mlir::Value hostPtr);

/// Release the heap storage held by the real host-ABI value that \p hostPtr
/// points to, where the value has device type \p devTy. A host value of a
/// dynamic type is composed of strings, vectors, structs, and trivial types, so
/// this recursively destroys each such member and then the vector's own
/// storage, as the value's destructor would. The value must not be used after.
void destroyHostValue(mlir::Location loc, mlir::OpBuilder &builder,
                      mlir::ModuleOp module, mlir::Type devTy,
                      mlir::Value hostPtr);

/// Any dynamic type is supported as a result type. Given a real device-side
/// value of type \p devTy, decompose it into the buffer's dynamic-output-slot
/// shape. A span decomposes to a `{ptr, count}` pair whose pointer already
/// refers to fully-realized device values; a struct decomposes member by
/// member, `recursing` only into dynamic members.
mlir::Value valueToBufferSlot(mlir::Location loc, mlir::OpBuilder &builder,
                              mlir::Type devTy, mlir::Value realVal);

/// The inverse of `valueToBufferSlot`. Given an already-loaded buffer
/// output-slot value, reconstruct the real device-side value of type \p devTy.
mlir::Value bufferSlotToValue(mlir::Location loc, mlir::OpBuilder &builder,
                              mlir::Type devTy, mlir::Value slotVal);

/// Build, in place, the real host value that corresponds to the device-side
/// value \p devVal of dynamic type \p devTy. \p hostDest points to
/// uninitialized memory of the host representation of \p devTy, such as the
/// `sret` block of a host function. This is the reverse of
/// `reduceHostToDeviceValue`, for a kernel's result.
///
/// The heap storage that the device value refers to belongs to the host value
/// afterwards. A vector of static elements adopts its storage as is. Where the
/// host's elements are shaped differently from the device's, a new array of
/// host elements is built, which adopts the storage of the elements, and the
/// device's array is released. The `malloc` and `free` intrinsics must already
/// be loaded in \p module.
void buildHostValueFromDeviceValue(mlir::Location loc, mlir::OpBuilder &builder,
                                   mlir::ModuleOp module, mlir::Type devTy,
                                   mlir::Value devVal, mlir::Value hostDest);

} // namespace cudaq::opt::marshal
