/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "clang/Basic/Version.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/Passes.h"
#include <cxxabi.h>
#include <regex>

namespace cudaq::opt {
#define GEN_PASS_DEF_GENERATEKERNELEXECUTION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "kernel-execution"

using namespace mlir;

namespace {
// Define some constant function name strings.
static constexpr const char cudaqRegisterLambdaName[] =
    "cudaqRegisterLambdaName";
static constexpr const char cudaqRegisterArgsCreator[] =
    "cudaqRegisterArgsCreator";
static constexpr const char cudaqRegisterKernelName[] =
    "cudaqRegisterKernelName";

/// This value is used to indicate that a kernel does not return a result.
static constexpr std::uint64_t NoResultOffset =
    std::numeric_limits<std::int32_t>::max();

class GenerateKernelExecution
    : public cudaq::opt::impl::GenerateKernelExecutionBase<
          GenerateKernelExecution> {
public:
  using GenerateKernelExecutionBase::GenerateKernelExecutionBase;

  /// Creates the function signature for a thunk function. The signature is
  /// always the same for all thunk functions.
  FunctionType getThunkType(MLIRContext *ctx) {
    auto ptrTy = cudaq::cc::PointerType::get(IntegerType::get(ctx, 8));
    return FunctionType::get(ctx, {ptrTy, IntegerType::get(ctx, 1)},
                             {cudaq::opt::factory::getDynamicBufferType(ctx)});
  }

  /// Add LLVM code with the OpBuilder that computes the size in bytes
  /// of a `std::vector<T>` array in the same way as a `std::vector<T>::size()`.
  /// This assumes the vector is laid out in memory as the following structure.
  ///
  /// <code>
  ///   struct vector {
  ///     T* begin;
  ///     T* end;
  ///     T* allocated_end;
  ///   };
  /// </code>
  ///
  /// The first two elements are pointers to the beginning and end of the data
  /// in the vector, respectively. This data is kept in a contiguous memory
  /// range. The following implementation follows what Clang CodeGen produces
  /// for `std::vector<T>::size()` without the final `sdiv` op that divides the
  /// `sizeof(data[N])` by the `sizeof(T)`. The result is the total required
  /// memory size for the vector data itself in \e bytes.
  ///
  /// In order to handle a std::string value it is assumed to be laid out in
  /// memory as the following structure.
  ///
  /// <code>
  ///   struct vector {
  ///     i8* data;
  ///     i64 length;
  ///     [i8 x 16] inlinedata;
  ///   };
  /// </code>
  ///
  /// In the string case, the size can just be read from the data structure.
  Value getVectorSize(Location loc, OpBuilder &builder,
                      cudaq::cc::PointerType ptrTy, Value arg) {
    // Create the i64 type
    Type i64Ty = builder.getI64Type();

    // We're given ptr<struct<...>>, get that struct type (struct<T*,T*,T*>)
    auto inpStructTy = cast<cudaq::cc::StructType>(ptrTy.getElementType());

    if (inpStructTy.getMember(1) == i64Ty) {
      // This is a string, so just read the length out.
      auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
      auto lenPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI64Ty, arg, SmallVector<cudaq::cc::ComputePtrArg>{1});
      return builder.create<cudaq::cc::LoadOp>(loc, lenPtr);
    }

    // For the following GEP calls, we'll expect them to return T**
    auto ptrTtype = cudaq::cc::PointerType::get(inpStructTy.getMember(0));

    // Get the pointer to the pointer of the end of the array
    Value endPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTtype, arg, SmallVector<cudaq::cc::ComputePtrArg>{1});

    // Get the pointer to the pointer of the beginning of the array
    Value beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTtype, arg, SmallVector<cudaq::cc::ComputePtrArg>{0});

    // Load to a T*
    endPtr = builder.create<cudaq::cc::LoadOp>(loc, endPtr);
    beginPtr = builder.create<cudaq::cc::LoadOp>(loc, beginPtr);

    // Map those pointers to integers
    Value endInt = builder.create<cudaq::cc::CastOp>(loc, i64Ty, endPtr);
    Value beginInt = builder.create<cudaq::cc::CastOp>(loc, i64Ty, beginPtr);

    // Subtracting these will give us the size in bytes.
    return builder.create<arith::SubIOp>(loc, endInt, beginInt);
  }

  /// Helper that converts a byte length to a length of i64.
  Value convertLengthBytesToLengthI64(Location loc, OpBuilder &builder,
                                      Value length) {
    auto eight = builder.create<arith::ConstantIntOp>(loc, 8, 64);
    return builder.create<arith::DivSIOp>(loc, length, eight);
  }

  /// This computes a vector's size and handles recursive vector types. This
  /// first value returned is the size of the top level (outermost) vector in
  /// bytes. The second value is the recursive size of all the vectors within
  /// the outer vector.
  std::pair<Value, Value>
  computeRecursiveVectorSize(Location loc, OpBuilder &builder, Value hostArg,
                             cudaq::cc::PointerType hostVecTy,
                             cudaq::cc::SpanLikeType stdvecTy) {
    Value topLevelSize;
    Value recursiveSize;
    auto eleTy = stdvecTy.getElementType();
    if (auto sTy = dyn_cast<cudaq::cc::SpanLikeType>(eleTy)) {
      // This is the recursive case. vector<vector<...>>. Convert size of
      // vectors to i64s.
      topLevelSize = computeHostVectorLengthInBytes(
          loc, builder, hostArg, stdvecTy.getElementType(), hostVecTy);
      auto nested = fetchHostVectorFront(loc, builder, hostArg, hostVecTy);
      auto tmp = builder.create<cudaq::cc::AllocaOp>(loc, builder.getI64Type());
      builder.create<cudaq::cc::StoreOp>(loc, topLevelSize, tmp);
      // Convert bytes to units of i64. (Divide by 8)
      auto topLevelCount =
          convertLengthBytesToLengthI64(loc, builder, topLevelSize);
      // Now walk the vectors recursively.
      auto topLevelIndex = builder.create<cudaq::cc::CastOp>(
          loc, builder.getI64Type(), topLevelCount,
          cudaq::cc::CastOpMode::Unsigned);
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, topLevelIndex,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = block.getArgument(0);
            auto sub = builder.create<cudaq::cc::ComputePtrOp>(loc, hostVecTy,
                                                               nested, i);
            auto p =
                computeRecursiveVectorSize(loc, builder, sub, hostVecTy, sTy);
            auto subSz = builder.create<cudaq::cc::LoadOp>(loc, tmp);
            auto sum = builder.create<arith::AddIOp>(loc, p.second, subSz);
            builder.create<cudaq::cc::StoreOp>(loc, sum, tmp);
          });
      recursiveSize = builder.create<cudaq::cc::LoadOp>(loc, tmp);
    } else {
      // Non-recusive case. Just compute the size of the top-level vector<T>.
      topLevelSize = getVectorSize(loc, builder, hostVecTy, hostArg);
      recursiveSize = topLevelSize;
    }
    return {topLevelSize, recursiveSize};
  }

  /// This computes a dynamic struct's size and handles recursive dynamic types.
  /// This first value returned is the initial value of the top level
  /// (outermost) struct to be saved in the buffer. More specifically, any
  /// (recursive) member that is a vector is replaced by a i64 byte size. The
  /// offset of the trailing data is, as always, implicit. The second value is
  /// the recursive size of all the dynamic components within the outer struct.
  std::pair<Value, Value> computeRecursiveDynamicStructSize(
      Location loc, OpBuilder &builder, cudaq::cc::StructType structTy,
      Value arg, Value totalSize, cudaq::cc::StructType genTy) {
    Value retval = builder.create<cudaq::cc::UndefOp>(loc, genTy);
    auto argTy = cast<cudaq::cc::PointerType>(arg.getType());
    for (auto iter : llvm::enumerate(structTy.getMembers())) {
      auto memTy = iter.value();
      std::int32_t off = iter.index();
      auto structMemTy =
          cast<cudaq::cc::StructType>(argTy.getElementType()).getMember(off);
      auto structMemPtrTy = cudaq::cc::PointerType::get(structMemTy);
      auto memPtrVal = builder.create<cudaq::cc::ComputePtrOp>(
          loc, structMemPtrTy, arg, ArrayRef<cudaq::cc::ComputePtrArg>{off});
      if (cudaq::cc::isDynamicType(memTy)) {
        if (auto sTy = dyn_cast<cudaq::cc::StructType>(memTy)) {
          auto gTy = cast<cudaq::cc::StructType>(structMemTy);
          auto pr = computeRecursiveDynamicStructSize(
              loc, builder, sTy, memPtrVal, totalSize, gTy);
          retval = builder.create<cudaq::cc::InsertValueOp>(
              loc, retval.getType(), retval, pr.first, off);
          totalSize = builder.create<arith::AddIOp>(loc, totalSize, pr.second);
          continue;
        }
        auto memStdVecTy = cast<cudaq::cc::SpanLikeType>(memTy);
        Type eTy = memStdVecTy.getElementType();
        auto stlVecTy = cudaq::opt::factory::stlVectorType(eTy);
        auto ptrMemTy = cudaq::cc::PointerType::get(stlVecTy);
        auto pr = computeRecursiveVectorSize(loc, builder, memPtrVal, ptrMemTy,
                                             memStdVecTy);
        retval = builder.create<cudaq::cc::InsertValueOp>(
            loc, retval.getType(), retval, pr.second, off);
        totalSize = builder.create<arith::AddIOp>(loc, totalSize, pr.first);
        continue;
      }
      auto memVal = builder.create<cudaq::cc::LoadOp>(loc, memPtrVal);
      retval = builder.create<cudaq::cc::InsertValueOp>(loc, retval.getType(),
                                                        retval, memVal, off);
    }
    return {retval, totalSize};
  }

  /// Copy a vector's data, which must be \p bytes in length, from \p hostArg to
  /// \p outputBuffer. The hostArg must have a pointer type that is compatible
  /// with the triple pointer std::vector base implementation.
  Value copyVectorData(Location loc, OpBuilder &builder, Value bytes,
                       Value hostArg, Value outputBuffer) {
    auto notVolatile = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    auto inStructTy = cast<cudaq::cc::StructType>(
        cast<cudaq::cc::PointerType>(hostArg.getType()).getElementType());
    auto beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, cudaq::cc::PointerType::get(inStructTy.getMember(0)), hostArg,
        SmallVector<cudaq::cc::ComputePtrArg>{0});
    auto fromBuff = builder.create<cudaq::cc::LoadOp>(loc, beginPtr);
    auto i8Ty = builder.getI8Type();
    auto vecFromBuff = cudaq::opt::factory::createCast(
        builder, loc, cudaq::cc::PointerType::get(i8Ty), fromBuff);
    builder.create<func::CallOp>(
        loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
        SmallVector<Value>{outputBuffer, vecFromBuff, bytes, notVolatile});
    auto i8ArrTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty));
    auto buf1 =
        cudaq::opt::factory::createCast(builder, loc, i8ArrTy, outputBuffer);
    // Increment outputBuffer by size bytes.
    return builder.create<cudaq::cc::ComputePtrOp>(
        loc, outputBuffer.getType(), buf1, SmallVector<Value>{bytes});
  }

  /// Given that \p arg is a SpanLikeType value, compute its extent size (the
  /// number of elements in the outermost vector times `sizeof(int64_t)`) and
  /// total recursive size (both values are in bytes). We add the extent size
  /// into the message buffer field and increase the size of the addend by the
  /// total recursive size.
  std::pair<Value, Value> insertVectorSizeAndIncrementExtraBytes(
      Location loc, OpBuilder &builder, Value arg,
      cudaq::cc::PointerType ptrInTy, cudaq::cc::SpanLikeType stdvecTy,
      Value stVal, std::int32_t idx, Value extraBytes) {
    auto [extentSize, recursiveSize] =
        computeRecursiveVectorSize(loc, builder, arg, ptrInTy, stdvecTy);
    stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                     stVal, extentSize, idx);
    extraBytes = builder.create<arith::AddIOp>(loc, extraBytes, recursiveSize);
    return {stVal, extraBytes};
  }

  Value genComputeReturnOffset(Location loc, OpBuilder &builder,
                               FunctionType funcTy,
                               cudaq::cc::StructType msgStructTy,
                               Value nullSt) {
    auto i64Ty = builder.getI64Type();
    if (funcTy.getNumResults() == 0)
      return builder.create<arith::ConstantIntOp>(loc, NoResultOffset, 64);
    auto members = msgStructTy.getMembers();
    std::int32_t numKernelArgs = funcTy.getNumInputs();
    auto resTy = cudaq::cc::PointerType::get(members[numKernelArgs]);
    auto gep = builder.create<cudaq::cc::ComputePtrOp>(
        loc, resTy, nullSt,
        SmallVector<cudaq::cc::ComputePtrArg>{numKernelArgs});
    return builder.create<cudaq::cc::CastOp>(loc, i64Ty, gep);
  }

  /// Create a function that determines the return value offset in the message
  /// buffer.
  void genReturnOffsetFunction(Location loc, OpBuilder &builder,
                               FunctionType devKernelTy,
                               cudaq::cc::StructType msgStructTy,
                               const std::string &classNameStr) {
    auto *ctx = builder.getContext();
    auto i64Ty = builder.getI64Type();
    auto funcTy = FunctionType::get(ctx, {}, {i64Ty});
    auto returnOffsetFunc = builder.create<func::FuncOp>(
        loc, classNameStr + ".returnOffset", funcTy);
    OpBuilder::InsertionGuard guard(builder);
    auto *entry = returnOffsetFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    auto ptrTy = cudaq::cc::PointerType::get(msgStructTy);
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    auto basePtr = builder.create<cudaq::cc::CastOp>(loc, ptrTy, zero);
    auto result =
        genComputeReturnOffset(loc, builder, devKernelTy, msgStructTy, basePtr);
    builder.create<func::ReturnOp>(loc, result);
  }

  /// Creates a function that can take a block of pointers to argument values
  /// and using the compiler's knowledge of a kernel encodes those argument
  /// values into a message buffer. The message buffer is a pointer-free block
  /// of memory allocated on the heap on the host. Once the argument values are
  /// packed into the message buffer, they can be passed to altLaunchKernel or
  /// the corresponding thunk function.
  ///
  /// The created function takes two arguments. The first argument is a pointer
  /// to a block containing the argument values to be encoded. The second
  /// argument a pointer to a pointer into which the message buffer value will
  /// be written for return. This function returns to size of the message
  /// buffer. (Message buffers are at least the size of \p structTy but may be
  /// extended.)
  func::FuncOp genKernelArgsCreatorFunction(Location loc, OpBuilder &builder,
                                            FunctionType devKernelTy,
                                            cudaq::cc::StructType msgStructTy,
                                            const std::string &classNameStr,
                                            FunctionType hostFuncTy,
                                            bool hasThisPtr) {
    auto *ctx = builder.getContext();
    Type i8Ty = builder.getI8Type();
    Type ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto ptrPtrType = cudaq::cc::PointerType::get(ptrI8Ty);
    Type i64Ty = builder.getI64Type();
    auto structPtrTy = cudaq::cc::PointerType::get(msgStructTy);
    auto getHostArgType = [&](unsigned idx) {
      bool hasSRet = cudaq::opt::factory::hasHiddenSRet(hostFuncTy);
      unsigned count = cudaq::cc::numberOfHiddenArgs(hasThisPtr, hasSRet);
      return hostFuncTy.getInput(count + idx);
    };

    // Create the function that we'll fill.
    auto funcType = FunctionType::get(ctx, {ptrPtrType, ptrPtrType}, {i64Ty});
    auto argsCreatorFunc = builder.create<func::FuncOp>(
        loc, classNameStr + ".argsCreator", funcType);
    OpBuilder::InsertionGuard guard(builder);
    auto *entry = argsCreatorFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Get the original function args
    auto kernelArgTypes = devKernelTy.getInputs().drop_front(startingArgIdx);

    // Init the struct
    Value stVal = builder.create<cudaq::cc::UndefOp>(loc, msgStructTy);

    // Get the variadic void* args
    auto variadicArgs = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(ptrI8Ty)),
        entry->getArgument(0));

    // Initialize the counter for extra size.
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    Value extraBytes = zero;

    // Process all the arguments for the original call by looping over the
    // kernel's arguments.
    bool hasTrailingData = false;
    DenseMap<std::int32_t, Value> replacementArgs;
    for (auto kaIter : llvm::enumerate(kernelArgTypes)) {
      std::int32_t idx = kaIter.index();

      // The current cudaq kernel arg and message buffer element type.
      Type currArgTy = kaIter.value();
      Type currEleTy = msgStructTy.getMember(idx);

      // Skip any elements that are callables or empty structures.
      if (isa<cudaq::cc::CallableType>(currEleTy))
        continue;
      if (auto strTy = dyn_cast<cudaq::cc::StructType>(currEleTy))
        if (strTy.isEmpty())
          continue;

      // Get the pointer to the argument from out of the block of pointers,
      // which are the variadic args.
      Value argPtrPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrPtrType, variadicArgs,
          SmallVector<cudaq::cc::ComputePtrArg>{idx});
      Value argPtr = builder.create<cudaq::cc::LoadOp>(loc, ptrI8Ty, argPtrPtr);

      if (auto stdvecTy = dyn_cast<cudaq::cc::SpanLikeType>(currArgTy)) {
        // If this is a vector argument, then we will add data to the message
        // buffer's addendum (unless the vector is length 0).
        auto ptrInTy = cudaq::cc::PointerType::get(
            cudaq::opt::factory::stlVectorType(stdvecTy.getElementType()));

        Value arg = builder.create<cudaq::cc::CastOp>(loc, ptrInTy, argPtr);
        if (stdvecTy.getElementType() == builder.getI1Type()) {
          // Create a mock vector of i8 and populate the bools, 1 per char.
          Value temp = builder.create<cudaq::cc::AllocaOp>(
              loc, ptrInTy.getElementType());
          builder.create<func::CallOp>(loc, std::nullopt,
                                       cudaq::stdvecBoolUnpackToInitList,
                                       ArrayRef<Value>{temp, arg});
          replacementArgs[idx] = temp;
          arg = temp;
        }

        auto [p1, p2] = insertVectorSizeAndIncrementExtraBytes(
            loc, builder, arg, ptrInTy, stdvecTy, stVal, idx, extraBytes);
        stVal = p1;
        extraBytes = p2;
        hasTrailingData = true;
        continue;
      }

      if (auto strTy = dyn_cast<cudaq::cc::StructType>(currArgTy)) {
        Value v = argPtr;
        if (!cudaq::cc::isDynamicType(strTy)) {
          // struct is static size, so just load the value (byval ptr).
          v = builder.create<cudaq::cc::CastOp>(
              loc, cudaq::cc::PointerType::get(currEleTy), v);
          v = builder.create<cudaq::cc::LoadOp>(loc, v);
          stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                           stVal, v, idx);
          continue;
        }
        auto genTy = cast<cudaq::cc::StructType>(currEleTy);
        Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
        Type hostArgTy = getHostArgType(idx);
        v = builder.create<cudaq::cc::CastOp>(loc, hostArgTy, v);
        auto [quakeVal, recursiveSize] = computeRecursiveDynamicStructSize(
            loc, builder, strTy, v, zero, genTy);
        stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                         stVal, quakeVal, idx);
        extraBytes =
            builder.create<arith::AddIOp>(loc, extraBytes, recursiveSize);
        hasTrailingData = true;
        continue;
      }
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(currEleTy)) {
        if (isa<cudaq::cc::StateType>(ptrTy.getElementType())) {
          // Special case: if the argument is a `cudaq::state*`, then just pass
          // the pointer. We can do that in this case because the synthesis step
          // (which will receive the argument data) is assumed to run in the
          // same memory space.
          argPtr = builder.create<cudaq::cc::CastOp>(loc, currEleTy, argPtr);
          stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                           stVal, argPtr, idx);
        }
        continue;
      }

      // cast to the struct element type, void* -> TYPE *
      argPtr = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(currEleTy), argPtr);
      Value loadedVal =
          builder.create<cudaq::cc::LoadOp>(loc, currEleTy, argPtr);
      stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                       stVal, loadedVal, idx);
    }

    // Compute the struct size
    Value structSize =
        builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, msgStructTy);

    // Here we do have vector args
    Value extendedStructSize =
        hasTrailingData
            ? builder.create<arith::AddIOp>(loc, structSize, extraBytes)
            : structSize;
    // If no vector args, handle this simple case and drop out
    Value buff = builder
                     .create<func::CallOp>(loc, ptrI8Ty, "malloc",
                                           ValueRange(extendedStructSize))
                     .getResult(0);

    Value casted = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, buff);
    builder.create<cudaq::cc::StoreOp>(loc, stVal, casted);
    if (hasTrailingData) {
      auto arrTy = cudaq::cc::ArrayType::get(i8Ty);
      auto ptrArrTy = cudaq::cc::PointerType::get(arrTy);
      auto cast1 = builder.create<cudaq::cc::CastOp>(loc, ptrArrTy, buff);
      Value vecToBuffer = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI8Ty, cast1, SmallVector<Value>{structSize});
      for (auto iter : llvm::enumerate(msgStructTy.getMembers())) {
        std::int32_t idx = iter.index();
        if (idx == static_cast<std::int32_t>(kernelArgTypes.size()))
          break;
        // Get the corresponding cudaq kernel arg type
        auto currArgTy = kernelArgTypes[idx];
        if (auto stdvecTy = dyn_cast<cudaq::cc::SpanLikeType>(currArgTy)) {
          auto bytes = builder.create<cudaq::cc::ExtractValueOp>(
              loc, builder.getI64Type(), stVal, idx);
          Value argPtrPtr = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrPtrType, variadicArgs,
              ArrayRef<cudaq::cc::ComputePtrArg>{idx});
          auto ptrInTy = cudaq::cc::PointerType::get(
              cudaq::opt::factory::stlVectorType(stdvecTy.getElementType()));
          Value arg =
              builder.create<cudaq::cc::LoadOp>(loc, ptrI8Ty, argPtrPtr);
          arg = builder.create<cudaq::cc::CastOp>(loc, ptrInTy, arg);
          vecToBuffer = encodeVectorData(loc, builder, bytes, stdvecTy, arg,
                                         vecToBuffer, ptrInTy);
          if (stdvecTy.getElementType() == builder.getI1Type()) {
            auto ptrI1Ty = cudaq::cc::PointerType::get(builder.getI1Type());
            assert(replacementArgs.count(idx) && "must be in map");
            auto arg = replacementArgs[idx];
            auto heapPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, cudaq::cc::PointerType::get(ptrI1Ty), arg,
                ArrayRef<cudaq::cc::ComputePtrArg>{0});
            auto loadHeapPtr = builder.create<cudaq::cc::LoadOp>(loc, heapPtr);
            auto i8Ty = builder.getI8Type();
            Value heapCast = builder.create<cudaq::cc::CastOp>(
                loc, cudaq::cc::PointerType::get(i8Ty), loadHeapPtr);
            builder.create<func::CallOp>(loc, std::nullopt, "free",
                                         ArrayRef<Value>{heapCast});
          }
        } else if (auto strTy = dyn_cast<cudaq::cc::StructType>(currArgTy)) {
          if (cudaq::cc::isDynamicType(strTy)) {
            Value argPtrPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, ptrPtrType, variadicArgs,
                ArrayRef<cudaq::cc::ComputePtrArg>{idx});
            Value arg =
                builder.create<cudaq::cc::LoadOp>(loc, ptrI8Ty, argPtrPtr);
            Type hostArgTy = getHostArgType(idx);
            arg = builder.create<cudaq::cc::CastOp>(loc, hostArgTy, arg);
            auto structPtrArrTy = cudaq::cc::PointerType::get(
                cudaq::cc::ArrayType::get(msgStructTy));
            auto temp =
                builder.create<cudaq::cc::CastOp>(loc, structPtrArrTy, buff);
            vecToBuffer = encodeDynamicStructData(loc, builder, strTy, arg,
                                                  temp, vecToBuffer);
          }
        }
      }
    }
    builder.create<cudaq::cc::StoreOp>(loc, buff, entry->getArgument(1));
    builder.create<func::ReturnOp>(loc, ValueRange{extendedStructSize});
    return argsCreatorFunc;
  }

  /// If the kernel has an sret argument, then we rewrite the kernel's signature
  /// on the target. Note that this requires that the target has the ability to
  /// pass stack pointers as function arguments. These stack pointers will
  /// obviously only necessarily be valid to the target executing the kernel.
  void updateQPUKernelAsSRet(OpBuilder &builder, func::FuncOp funcOp,
                             FunctionType newFuncTy) {
    auto funcTy = funcOp.getFunctionType();
    // We add exactly 1 sret argument regardless of how many fields are folded
    // into it.
    assert(newFuncTy.getNumInputs() == funcTy.getNumInputs() + 1 &&
           "sret should be a single argument");
    auto *ctx = funcOp.getContext();
    auto eleTy = cudaq::opt::factory::getSRetElementType(funcTy);
    NamedAttrList attrs;
    attrs.set(LLVM::LLVMDialect::getStructRetAttrName(), TypeAttr::get(eleTy));
    funcOp.insertArgument(0, newFuncTy.getInput(0), attrs.getDictionary(ctx),
                          funcOp.getLoc());
    auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
    OpBuilder::InsertionGuard guard(builder);
    SmallVector<Operation *> returnsToErase;
    // Update all func.return to store values to the sret block.
    funcOp->walk([&](func::ReturnOp retOp) {
      auto loc = retOp.getLoc();
      builder.setInsertionPoint(retOp);
      auto cast = builder.create<cudaq::cc::CastOp>(loc, elePtrTy,
                                                    funcOp.getArgument(0));
      if (funcOp.getNumResults() > 1) {
        for (int i = 0, end = funcOp.getNumResults(); i != end; ++i) {
          auto mem = builder.create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(funcTy.getResult(i)), cast,
              SmallVector<cudaq::cc::ComputePtrArg>{i});
          builder.create<cudaq::cc::StoreOp>(loc, retOp.getOperands()[i], mem);
        }
      } else if (auto stdvecTy =
                     dyn_cast<cudaq::cc::SpanLikeType>(funcTy.getResult(0))) {
        auto stdvec = retOp.getOperands()[0];
        auto eleTy = [&]() -> Type {
          // TODO: Fold this conversion into the StdvecDataOp builder. We will
          // never get a data buffer which is not byte addressable and where
          // the width is less than 8.
          if (auto intTy = dyn_cast<IntegerType>(stdvecTy.getElementType()))
            if (intTy.getWidth() < 8)
              return builder.getI8Type();
          return stdvecTy.getElementType();
        }();
        auto i8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
        auto ptrTy = cudaq::cc::PointerType::get(eleTy);
        auto data = builder.create<cudaq::cc::StdvecDataOp>(loc, ptrTy, stdvec);
        auto mem0 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(i8Ty), cast,
            SmallVector<cudaq::cc::ComputePtrArg>{0});
        auto mem1 = builder.create<cudaq::cc::CastOp>(
            loc, cudaq::cc::PointerType::get(ptrTy), mem0);
        builder.create<cudaq::cc::StoreOp>(loc, data, mem1);
        auto i64Ty = builder.getI64Type();
        auto size = builder.create<cudaq::cc::StdvecSizeOp>(loc, i64Ty, stdvec);
        auto mem2 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(i64Ty), cast,
            SmallVector<cudaq::cc::ComputePtrArg>{1});
        builder.create<cudaq::cc::StoreOp>(loc, size, mem2);
      } else {
        builder.create<cudaq::cc::StoreOp>(loc, retOp.getOperands()[0], cast);
      }
      builder.create<func::ReturnOp>(loc);
      returnsToErase.push_back(retOp);
    });
    for (auto *op : returnsToErase)
      op->erase();
    for (std::size_t i = 0, end = funcOp.getNumResults(); i != end; ++i)
      funcOp.eraseResult(0);
  }

  /// In the thunk, we need to unpack any `std::vector` objects encoded in the
  /// packet. Since these have dynamic size, they are encoded as trailing bytes
  /// by offset and size. The offset is implicit from the values of the
  /// arguments. All sizes are encoded as `int64_t`.
  ///
  /// A vector of vector of ... T is encoded as a int64_t (length). At the
  /// offset of the level `i` vector will be a sequence of sizes for the level
  /// `i+1` vectors. For the leaf vector level, `n`, the blocks of data for each
  /// vector will be immediately following for each vector at level `n` for the
  /// branch of the tree being encoded.
  ///
  /// For example, a variable defined and initialized as
  /// ```
  /// vector<vector<vector<char>>> example =
  ///    {{{'a'}, {'b', 'c'}, {'z'}}, {{'d' 'e', 'f'}}};
  /// ```
  ///
  /// and passed as an argument to a kernel will be encoded as the following
  /// block. The block will have a structure with the declared arguments
  /// followed by an addendum of variable data, where the vector data is
  /// encoded.
  ///
  /// ```
  ///   arguments: { ..., 1, ... }
  ///   addendum: [[3; 1 2 1, a, b c, z] [1; 3, d e f]]
  /// ```
  std::pair<Value, Value> unpackStdVector(Location loc, OpBuilder &builder,
                                          cudaq::cc::SpanLikeType stdvecTy,
                                          Value vecSize, Value trailingData) {
    // Convert the pointer-free std::vector<T> to a span structure to be
    // passed. A span structure is a pointer and a size (in element
    // units). Note that this structure may be recursive.
    auto i8Ty = builder.getI8Type();
    auto arrI8Ty = cudaq::cc::ArrayType::get(i8Ty);
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto bytesTy = cudaq::cc::PointerType::get(arrI8Ty);
    Type eleTy = stdvecTy.getElementType();
    auto innerStdvecTy = dyn_cast<cudaq::cc::SpanLikeType>(eleTy);
    std::size_t eleSize =
        innerStdvecTy ? /*(i64Type/8)*/ 8 : dataLayout->getTypeSize(eleTy);
    auto eleSizeVal = [&]() -> Value {
      if (eleSize)
        return builder.create<arith::ConstantIntOp>(loc, eleSize, 64);
      assert(isa<cudaq::cc::StructType>(eleTy) ||
             (isa<cudaq::cc::ArrayType>(eleTy) &&
              !cast<cudaq::cc::ArrayType>(eleTy).isUnknownSize()));
      auto i64Ty = builder.getI64Type();
      return builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, eleTy);
    }();
    auto vecLength = builder.create<arith::DivSIOp>(loc, vecSize, eleSizeVal);
    if (innerStdvecTy) {
      // Recursive case: std::vector<std::vector<...>>
      // TODO: Uses stack allocation, however it may be better to use heap
      // allocation. It's not clear the QPU has heap memory allocation. If this
      // uses heap allocation, then the thunk must free that memory *after* the
      // kernel proper returns.
      auto vecTmp = builder.create<cudaq::cc::AllocaOp>(loc, eleTy, vecLength);
      auto currentEnd = builder.create<cudaq::cc::AllocaOp>(loc, ptrI8Ty);
      auto i64Ty = builder.getI64Type();
      auto arrI64Ty = cudaq::cc::ArrayType::get(i64Ty);
      auto arrTy = cudaq::cc::PointerType::get(arrI64Ty);
      auto innerVec =
          builder.create<cudaq::cc::CastOp>(loc, arrTy, trailingData);
      auto trailingBytes =
          builder.create<cudaq::cc::CastOp>(loc, bytesTy, trailingData);
      trailingData = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI8Ty, trailingBytes, vecSize);
      builder.create<cudaq::cc::StoreOp>(loc, trailingData, currentEnd);
      // Loop over each subvector in the vector and recursively unpack it into
      // the vecTmp variable. Leaf vectors do not need a fresh variable. This
      // effectively translates all the size/offset information for all the
      // subvectors into temps.
      Value vecLengthIndex = builder.create<cudaq::cc::CastOp>(
          loc, builder.getI64Type(), vecLength,
          cudaq::cc::CastOpMode::Unsigned);
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, vecLengthIndex,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = block.getArgument(0);
            auto innerPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, cudaq::cc::PointerType::get(i64Ty), innerVec,
                SmallVector<cudaq::cc::ComputePtrArg>{i});
            Value innerVecSize =
                builder.create<cudaq::cc::LoadOp>(loc, innerPtr);
            Value tmp = builder.create<cudaq::cc::LoadOp>(loc, currentEnd);
            auto unpackPair =
                unpackStdVector(loc, builder, innerStdvecTy, innerVecSize, tmp);
            auto ptrInnerTy = cudaq::cc::PointerType::get(innerStdvecTy);
            auto subVecPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, ptrInnerTy, vecTmp,
                SmallVector<cudaq::cc::ComputePtrArg>{i});
            builder.create<cudaq::cc::StoreOp>(loc, unpackPair.first,
                                               subVecPtr);
            builder.create<cudaq::cc::StoreOp>(loc, unpackPair.second,
                                               currentEnd);
          });
      auto coerceResult = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(stdvecTy), vecTmp);
      trailingData = builder.create<cudaq::cc::LoadOp>(loc, currentEnd);
      Value result = builder.create<cudaq::cc::StdvecInitOp>(
          loc, stdvecTy, coerceResult, vecLength);
      return {result, trailingData};
    }
    // Must divide by byte, 8 bits.
    // The data is at trailingData and is valid for vecLength of eleTy.
    auto castData = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(eleTy), trailingData);
    Value stdVecResult = builder.create<cudaq::cc::StdvecInitOp>(
        loc, stdvecTy, castData, vecLength);
    auto arrTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty));
    Value casted = builder.create<cudaq::cc::CastOp>(loc, arrTy, trailingData);
    trailingData =
        builder.create<cudaq::cc::ComputePtrOp>(loc, ptrI8Ty, casted, vecSize);
    return {stdVecResult, trailingData};
  }

  /// Translate the buffer data to a sequence of arguments suitable to the
  /// actual kernel call.
  ///
  /// \param inTy      The actual expected type of the argument.
  /// \param structTy  The modified buffer type over all the arguments at the
  /// current level.
  std::pair<Value, Value> processInputValue(Location loc, OpBuilder &builder,
                                            Value trailingData, Value val,
                                            Type inTy, std::int64_t off,
                                            cudaq::cc::StructType structTy) {
    if (isa<cudaq::cc::CallableType>(inTy))
      return {builder.create<cudaq::cc::UndefOp>(loc, inTy), trailingData};
    if (auto stdVecTy = dyn_cast<cudaq::cc::SpanLikeType>(inTy)) {
      Value vecSize = builder.create<cudaq::cc::ExtractValueOp>(
          loc, builder.getI64Type(), val, off);
      return unpackStdVector(loc, builder, stdVecTy, vecSize, trailingData);
    }
    if (auto strTy = dyn_cast<cudaq::cc::StructType>(inTy)) {
      if (!cudaq::cc::isDynamicType(strTy)) {
        if (strTy.isEmpty())
          return {builder.create<cudaq::cc::UndefOp>(loc, inTy), trailingData};
        return {builder.create<cudaq::cc::ExtractValueOp>(loc, inTy, val, off),
                trailingData};
      }
      // The struct contains dynamic components. Extract them and build up the
      // struct value to be passed as an argument.
      Type buffMemTy = structTy.getMember(off);
      Value strVal = builder.create<cudaq::cc::UndefOp>(loc, inTy);
      Value subVal =
          builder.create<cudaq::cc::ExtractValueOp>(loc, buffMemTy, val, off);
      // Convert the argument type, strTy, to a buffer type.
      auto memberArgTy = cast<cudaq::cc::StructType>(
          cudaq::opt::factory::genArgumentBufferType(strTy));
      for (auto iter : llvm::enumerate(strTy.getMembers())) {
        auto memValPair =
            processInputValue(loc, builder, trailingData, subVal, iter.value(),
                              iter.index(), memberArgTy);
        trailingData = memValPair.second;
        strVal = builder.create<cudaq::cc::InsertValueOp>(
            loc, inTy, strVal, memValPair.first, iter.index());
      }
      return {strVal, trailingData};
    }
    return {builder.create<cudaq::cc::ExtractValueOp>(loc, inTy, val, off),
            trailingData};
  }

  /// Generate the thunk function. This function is called by the library
  /// callback function to "unpack" the arguments and pass them to the kernel
  /// function on the QPU side. The thunk will also save any return values to
  /// the memory block so that the calling function will be able to receive them
  /// when the kernel returns. Each thunk is custom generated to manage the
  /// arguments and return value of the corresponding kernel.
  func::FuncOp genThunkFunction(Location loc, OpBuilder &builder,
                                const std::string &classNameStr,
                                cudaq::cc::StructType structTy,
                                FunctionType funcTy, func::FuncOp funcOp) {
    Type structPtrTy = cudaq::cc::PointerType::get(structTy);
    auto *ctx = builder.getContext();
    auto thunkTy = getThunkType(ctx);
    auto thunk =
        builder.create<func::FuncOp>(loc, classNameStr + ".thunk", thunkTy);
    OpBuilder::InsertionGuard guard(builder);
    auto *thunkEntry = thunk.addEntryBlock();
    builder.setInsertionPointToStart(thunkEntry);
    auto castOp = builder.create<cudaq::cc::CastOp>(loc, structPtrTy,
                                                    thunkEntry->getArgument(0));
    auto isClientServer = thunkEntry->getArgument(1);
    Value val = builder.create<cudaq::cc::LoadOp>(loc, castOp);
    auto i64Ty = builder.getI64Type();

    // Compute the struct size without the trailing bytes, structSize.
    Value structSize =
        builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, structTy);

    // Compute location of trailing bytes.
    auto bufferPtrTy =
        cudaq::opt::factory::getIndexedObjectType(builder.getI8Type());
    Value extendedBuffer = builder.create<cudaq::cc::CastOp>(
        loc, bufferPtrTy, thunkEntry->getArgument(0));
    auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
    Value trailingData = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrI8Ty, extendedBuffer, structSize);

    // Unpack the arguments in the struct and build the argument list for
    // the call to the kernel code.
    SmallVector<Value> args;
    const bool hiddenSRet = cudaq::opt::factory::hasHiddenSRet(funcTy);
    FunctionType newFuncTy = [&]() {
      if (hiddenSRet) {
        auto sretPtrTy = cudaq::cc::PointerType::get(
            cudaq::opt::factory::getSRetElementType(funcTy));
        SmallVector<Type> inputTys = {sretPtrTy};
        inputTys.append(funcTy.getInputs().begin(), funcTy.getInputs().end());
        return FunctionType::get(ctx, inputTys, {});
      }
      return funcTy;
    }();
    int offset = funcTy.getNumInputs();
    if (hiddenSRet) {
      // Use the end of the argument block for the return values.
      auto eleTy = structTy.getMember(offset);
      auto mem = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(eleTy), castOp,
          SmallVector<cudaq::cc::ComputePtrArg>{offset});
      auto sretPtrTy = cudaq::cc::PointerType::get(
          cudaq::opt::factory::getSRetElementType(funcTy));
      auto sretMem = builder.create<cudaq::cc::CastOp>(loc, sretPtrTy, mem);
      args.push_back(sretMem);

      // Rewrite the original kernel's signature and return op(s).
      updateQPUKernelAsSRet(builder, funcOp, newFuncTy);
    }
    for (auto inp : llvm::enumerate(funcTy.getInputs())) {
      auto valPair = processInputValue(loc, builder, trailingData, val,
                                       inp.value(), inp.index(), structTy);
      trailingData = valPair.second;
      args.push_back(valPair.first);
    }
    auto call = builder.create<func::CallOp>(loc, newFuncTy.getResults(),
                                             funcOp.getName(), args);
    // If and only if the kernel returns non-sret results, then take those
    // values and store them in the results section of the struct. They will
    // eventually be returned to the original caller.
    if (!hiddenSRet && funcTy.getNumResults() == 1) {
      auto eleTy = structTy.getMember(offset);
      auto mem = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(eleTy), castOp,
          SmallVector<cudaq::cc::ComputePtrArg>{offset});
      builder.create<cudaq::cc::StoreOp>(loc, call.getResult(0), mem);
    }

    // If the original result was a std::vector<T>, then depending on whether
    // this is client-server or not, the thunk function packs the dynamic return
    // data into a message buffer or just returns a pointer to the shared heap
    // allocation, resp.
    bool hasVectorResult = funcTy.getNumResults() == 1 &&
                           isa<cudaq::cc::SpanLikeType>(funcTy.getResult(0));
    if (hasVectorResult) {
      auto *currentBlock = builder.getBlock();
      auto *reg = currentBlock->getParent();
      auto *thenBlock = builder.createBlock(reg);
      auto *elseBlock = builder.createBlock(reg);
      builder.setInsertionPointToEnd(currentBlock);
      builder.create<cf::CondBranchOp>(loc, isClientServer, thenBlock,
                                       elseBlock);
      builder.setInsertionPointToEnd(thenBlock);
      int offset = funcTy.getNumInputs();
      auto gepRes = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(structTy.getMember(offset)), castOp,
          SmallVector<cudaq::cc::ComputePtrArg>{offset});
      auto gepRes2 = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(thunkTy.getResults()[0]), gepRes);
      // createDynamicResult packs the input values and the dynamic results
      // into a single buffer to pass back as a message.
      auto res = builder.create<func::CallOp>(
          loc, thunkTy.getResults()[0], "__nvqpp_createDynamicResult",
          ValueRange{thunkEntry->getArgument(0), structSize, gepRes2});
      builder.create<func::ReturnOp>(loc, res.getResult(0));
      builder.setInsertionPointToEnd(elseBlock);
    }
    // zeroDynamicResult is used by models other than client-server. It assumes
    // that no messages need to be sent, the CPU and QPU code share a memory
    // space, and therefore skips making any copies.
    auto zeroRes =
        builder.create<func::CallOp>(loc, thunkTy.getResults()[0],
                                     "__nvqpp_zeroDynamicResult", ValueRange{});
    builder.create<func::ReturnOp>(loc, zeroRes.getResult(0));
    return thunk;
  }

  /// Generate code to initialize the std::vector<T>, \p sret, from an
  /// initializer list with data at \p data and length \p size. Use the library
  /// helper routine. This function takes two !llvm.ptr arguments.
  void genStdvecBoolFromInitList(Location loc, OpBuilder &builder, Value sret,
                                 Value data, Value size) {
    auto ptrTy = cudaq::cc::PointerType::get(builder.getContext());
    auto castData = builder.create<cudaq::cc::CastOp>(loc, ptrTy, data);
    auto castSret = builder.create<cudaq::cc::CastOp>(loc, ptrTy, sret);
    builder.create<func::CallOp>(loc, std::nullopt,
                                 cudaq::stdvecBoolCtorFromInitList,
                                 ArrayRef<Value>{castSret, castData, size});
  }

  /// Generate a `std::vector<T>` (where `T != bool`) from an initializer list.
  /// This is done with the assumption that `std::vector` is implemented as a
  /// triple of pointers. The original content of the vector is freed and the
  /// new content, which is already on the stack, is moved into the
  /// `std::vector`.
  void genStdvecTFromInitList(Location loc, OpBuilder &builder, Value sret,
                              Value data, Value tSize, Value vecSize) {
    auto i8Ty = builder.getI8Type();
    auto stlVectorTy =
        cudaq::cc::PointerType::get(cudaq::opt::factory::stlVectorType(i8Ty));
    auto ptrTy = cudaq::cc::PointerType::get(i8Ty);
    auto castSret = builder.create<cudaq::cc::CastOp>(loc, stlVectorTy, sret);
    auto ptrPtrTy = cudaq::cc::PointerType::get(ptrTy);
    auto sret0 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrPtrTy, castSret, SmallVector<cudaq::cc::ComputePtrArg>{0});
    Value vecPtr = builder.create<cudaq::cc::LoadOp>(loc, ptrTy, sret0);
    builder.create<func::CallOp>(loc, std::nullopt, "free", ValueRange{vecPtr});
    auto arrI8Ty = cudaq::cc::ArrayType::get(i8Ty);
    auto ptrArrTy = cudaq::cc::PointerType::get(arrI8Ty);
    auto buffPtr0 = builder.create<cudaq::cc::CastOp>(loc, ptrTy, data);
    builder.create<cudaq::cc::StoreOp>(loc, buffPtr0, sret0);
    auto sret1 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrPtrTy, castSret, SmallVector<cudaq::cc::ComputePtrArg>{1});
    Value byteLen = builder.create<arith::MulIOp>(loc, tSize, vecSize);
    auto buffPtr = builder.create<cudaq::cc::CastOp>(loc, ptrArrTy, data);
    auto endPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTy, buffPtr, SmallVector<cudaq::cc::ComputePtrArg>{byteLen});
    builder.create<cudaq::cc::StoreOp>(loc, endPtr, sret1);
    auto sret2 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrPtrTy, castSret, SmallVector<cudaq::cc::ComputePtrArg>{2});
    builder.create<cudaq::cc::StoreOp>(loc, endPtr, sret2);
  }

  static MutableArrayRef<BlockArgument>
  dropAnyHiddenArguments(MutableArrayRef<BlockArgument> args,
                         FunctionType funcTy, bool hasThisPointer) {
    const bool hiddenSRet = cudaq::opt::factory::hasHiddenSRet(funcTy);
    const unsigned count =
        cudaq::cc::numberOfHiddenArgs(hasThisPointer, hiddenSRet);
    if (count > 0 && args.size() >= count &&
        std::all_of(args.begin(), args.begin() + count, [](auto i) {
          return isa<cudaq::cc::PointerType>(i.getType());
        }))
      return args.drop_front(count);
    return args;
  }

  // Return the vector's length, computed on the CPU side, in bytes.
  Value computeHostVectorLengthInBytes(Location loc, OpBuilder &builder,
                                       Value hostArg, Type eleTy,
                                       cudaq::cc::PointerType hostVecTy) {
    auto rawSize = getVectorSize(loc, builder, hostVecTy, hostArg);
    if (isa<cudaq::cc::SpanLikeType>(eleTy)) {
      auto three = builder.create<arith::ConstantIntOp>(loc, 3, 64);
      return builder.create<arith::DivSIOp>(loc, rawSize, three);
    }
    return rawSize;
  }

  Value fetchHostVectorFront(Location loc, OpBuilder &builder, Value hostArg,
                             cudaq::cc::PointerType hostVecTy) {
    auto inpStructTy = cast<cudaq::cc::StructType>(hostVecTy.getElementType());
    auto ptrTtype = cudaq::cc::PointerType::get(inpStructTy.getMember(0));
    auto beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTtype, hostArg, SmallVector<cudaq::cc::ComputePtrArg>{0});
    auto ptrArrSTy = cudaq::opt::factory::getIndexedObjectType(inpStructTy);
    auto vecPtr = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(ptrArrSTy), beginPtr);
    return builder.create<cudaq::cc::LoadOp>(loc, vecPtr);
  }

  Value recursiveVectorDataCopy(Location loc, OpBuilder &builder, Value hostArg,
                                Value buffPtr, cudaq::cc::SpanLikeType stdvecTy,
                                cudaq::cc::PointerType hostVecTy) {
    auto vecLen = computeHostVectorLengthInBytes(loc, builder, hostArg,
                                                 stdvecTy, hostVecTy);
    auto nested = fetchHostVectorFront(loc, builder, hostArg, hostVecTy);
    auto vecLogicalLen = convertLengthBytesToLengthI64(loc, builder, vecLen);
    auto vecLenIndex = builder.create<cudaq::cc::CastOp>(
        loc, builder.getI64Type(), vecLogicalLen,
        cudaq::cc::CastOpMode::Unsigned);
    auto buffPtrTy = cast<cudaq::cc::PointerType>(buffPtr.getType());
    auto tmp = builder.create<cudaq::cc::AllocaOp>(loc, buffPtrTy);
    auto buffArrTy = cudaq::cc::ArrayType::get(buffPtrTy.getElementType());
    auto castPtr = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(buffArrTy), buffPtr);
    auto newEnd = builder.create<cudaq::cc::ComputePtrOp>(
        loc, buffPtrTy, castPtr, SmallVector<cudaq::cc::ComputePtrArg>{vecLen});
    builder.create<cudaq::cc::StoreOp>(loc, newEnd, tmp);
    auto i64Ty = builder.getI64Type();
    auto arrI64Ty = cudaq::cc::ArrayType::get(i64Ty);
    auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
    auto ptrArrTy = cudaq::cc::PointerType::get(arrI64Ty);
    auto vecBasePtr = builder.create<cudaq::cc::CastOp>(loc, ptrArrTy, buffPtr);
    auto nestedArr = builder.create<cudaq::cc::CastOp>(loc, hostVecTy, nested);
    auto hostArrVecTy = cudaq::cc::PointerType::get(
        cudaq::cc::ArrayType::get(hostVecTy.getElementType()));
    cudaq::opt::factory::createInvariantLoop(
        builder, loc, vecLenIndex,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value i = block.getArgument(0);
          auto currBuffPtr = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrI64Ty, vecBasePtr, ArrayRef<cudaq::cc::ComputePtrArg>{i});
          auto upCast =
              builder.create<cudaq::cc::CastOp>(loc, hostArrVecTy, nestedArr);
          auto hostSubVec = builder.create<cudaq::cc::ComputePtrOp>(
              loc, hostVecTy, upCast, ArrayRef<cudaq::cc::ComputePtrArg>{i});
          Value buff = builder.create<cudaq::cc::LoadOp>(loc, tmp);
          // Compute and save the byte size.
          auto vecSz = computeHostVectorLengthInBytes(
              loc, builder, hostSubVec, stdvecTy.getElementType(), hostVecTy);
          builder.create<cudaq::cc::StoreOp>(loc, vecSz, currBuffPtr);
          // Recursively copy vector data.
          auto endBuff = encodeVectorData(loc, builder, vecSz, stdvecTy,
                                          hostSubVec, buff, hostVecTy);
          builder.create<cudaq::cc::StoreOp>(loc, endBuff, tmp);
        });
    return builder.create<cudaq::cc::LoadOp>(loc, tmp);
  }

  /// Recursively encode a `std::vector` into a buffer's addendum. The data is
  /// read from \p hostArg. The data is \p bytes size long if this is a leaf
  /// vector, otherwise the size is computed on-the-fly during the encoding of
  /// the ragged array.
  /// \return The new pointer to the end of the addendum block.
  Value encodeVectorData(Location loc, OpBuilder &builder, Value bytes,
                         cudaq::cc::SpanLikeType stdvecTy, Value hostArg,
                         Value bufferAddendum, cudaq::cc::PointerType ptrInTy) {
    auto eleTy = stdvecTy.getElementType();
    if (auto subVecTy = dyn_cast<cudaq::cc::SpanLikeType>(eleTy))
      return recursiveVectorDataCopy(loc, builder, hostArg, bufferAddendum,
                                     subVecTy, ptrInTy);
    return copyVectorData(loc, builder, bytes, hostArg, bufferAddendum);
  }

  /// Recursively encode a struct which has dynamically sized members (such as
  /// vectors). The vector members are encoded as i64 sizes with the data
  /// attached to the buffer addendum.
  /// \return The new pointer to the end of the addendum block.
  Value encodeDynamicStructData(Location loc, OpBuilder &builder,
                                cudaq::cc::StructType deviceTy, Value hostArg,
                                Value bufferArg, Value bufferAddendum) {
    for (auto iter : llvm::enumerate(deviceTy.getMembers())) {
      auto memTy = iter.value();
      if (auto vecTy = dyn_cast<cudaq::cc::SpanLikeType>(memTy)) {
        Type eTy = vecTy.getElementType();
        auto hostTy = cudaq::opt::factory::stlVectorType(eTy);
        auto ptrHostTy = cudaq::cc::PointerType::get(hostTy);
        auto ptrI64Ty = cudaq::cc::PointerType::get(builder.getI64Type());
        std::int32_t offset = iter.index();
        auto sizeAddr = builder.create<cudaq::cc::ComputePtrOp>(
            loc, ptrI64Ty, bufferArg,
            ArrayRef<cudaq::cc::ComputePtrArg>{0, 0, offset});
        auto size = builder.create<cudaq::cc::LoadOp>(loc, sizeAddr);
        auto vecAddr = builder.create<cudaq::cc::ComputePtrOp>(
            loc, ptrHostTy, hostArg,
            ArrayRef<cudaq::cc::ComputePtrArg>{offset});
        bufferAddendum = encodeVectorData(loc, builder, size, vecTy, vecAddr,
                                          bufferAddendum, ptrHostTy);
      } else if (auto strTy = dyn_cast<cudaq::cc::StructType>(memTy)) {
        if (cudaq::cc::isDynamicType(strTy)) {
          auto ptrStrTy = cudaq::cc::PointerType::get(strTy);
          std::int32_t idx = iter.index();
          auto strAddr = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrStrTy, bufferArg,
              ArrayRef<cudaq::cc::ComputePtrArg>{idx});
          bufferAddendum = encodeDynamicStructData(loc, builder, strTy, strAddr,
                                                   bufferArg, bufferAddendum);
        }
      } else if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(memTy)) {
        // This is like vector type if the array has dynamic size. If it has a
        // constant size, it is like a struct with n identical members.
        TODO_loc(loc, "array type");
      }
    }
    return bufferAddendum;
  }

  static std::pair<bool, func::FuncOp>
  lookupHostEntryPointFunc(StringRef mangledEntryPointName, ModuleOp module,
                           func::FuncOp funcOp) {
    if (mangledEntryPointName.equals("BuilderKernel.EntryPoint") ||
        mangledEntryPointName.contains("_PyKernelEntryPointRewrite")) {
      // No host entry point needed.
      return {false, func::FuncOp{}};
    }
    if (auto *decl = module.lookupSymbol(mangledEntryPointName))
      if (auto func = dyn_cast<func::FuncOp>(decl)) {
        func.eraseBody();
        return {true, func};
      }
    funcOp.emitOpError("could not generate the host-side kernel function (" +
                       mangledEntryPointName + ")");
    return {true, func::FuncOp{}};
  }

  /// Generate an all new entry point body, calling launchKernel in the runtime
  /// library. Pass along the thunk, so the runtime can call the quantum
  /// circuit. These entry points are `operator()` member functions in a class,
  /// so account for the `this` argument here.
  void genNewHostEntryPoint1(Location loc, OpBuilder &builder,
                             FunctionType funcTy,
                             cudaq::cc::StructType structTy,
                             LLVM::GlobalOp kernelNameObj, func::FuncOp thunk,
                             func::FuncOp rewriteEntry, bool addThisPtr) {
    auto *ctx = builder.getContext();
    auto i64Ty = builder.getI64Type();
    auto offset = funcTy.getNumInputs();
    auto thunkTy = getThunkType(ctx);
    auto structPtrTy = cudaq::cc::PointerType::get(structTy);
    Block *rewriteEntryBlock = rewriteEntry.addEntryBlock();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(rewriteEntryBlock);
    Value stVal = builder.create<cudaq::cc::UndefOp>(loc, structTy);

    // Process all the arguments for the original call, ignoring any hidden
    // arguments (such as the `this` pointer).
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    Value extraBytes = zero;
    bool hasTrailingData = false;
    SmallVector<BlockArgument> blockArgs{dropAnyHiddenArguments(
        rewriteEntryBlock->getArguments(), funcTy, addThisPtr)};
    std::int32_t idx = 0;
    SmallVector<Value> blockValues(blockArgs.size());
    std::copy(blockArgs.begin(), blockArgs.end(), blockValues.begin());
    for (auto iter = blockArgs.begin(), end = blockArgs.end(); iter != end;
         ++iter, ++idx) {
      Value arg = *iter;
      Type inTy = arg.getType();
      Type quakeTy = funcTy.getInput(idx);
      // If the argument is a callable, skip it.
      if (isa<cudaq::cc::CallableType>(quakeTy))
        continue;
      // If the argument is an empty struct, skip it.
      if (auto strTy = dyn_cast<cudaq::cc::StructType>(quakeTy))
        if (strTy.isEmpty())
          continue;

      if (auto stdvecTy = dyn_cast<cudaq::cc::SpanLikeType>(quakeTy)) {
        // Per the CUDA-Q spec, an entry point kernel must take a `[const]
        // std::vector<T>` value argument.
        // Should the spec stipulate that pure device kernels must pass by
        // read-only reference, i.e., take `const std::vector<T> &` arguments?
        auto ptrInTy = cast<cudaq::cc::PointerType>(inTy);
        // If this is a std::vector<bool>, unpack it.
        if (stdvecTy.getElementType() == builder.getI1Type()) {
          // Create a mock vector of i8 and populate the bools, 1 per char.
          Value temp = builder.create<cudaq::cc::AllocaOp>(
              loc, ptrInTy.getElementType());
          builder.create<func::CallOp>(loc, std::nullopt,
                                       cudaq::stdvecBoolUnpackToInitList,
                                       ArrayRef<Value>{temp, arg});
          arg = blockValues[idx] = temp;
        }
        // FIXME: call the `size` member function. For expediency, assume this
        // is an std::vector and the size is the scaled delta between the
        // first two pointers. Use the unscaled size for now.
        auto [p1, p2] = insertVectorSizeAndIncrementExtraBytes(
            loc, builder, arg, ptrInTy, stdvecTy, stVal, idx, extraBytes);
        stVal = p1;
        extraBytes = p2;
        hasTrailingData = true;
        continue;
      }
      if (auto strTy = dyn_cast<cudaq::cc::StructType>(quakeTy)) {
        if (!isa<cudaq::cc::PointerType>(arg.getType())) {
          // If argument is not a pointer, then struct was promoted into a
          // register.
          auto *parent = builder.getBlock()->getParentOp();
          auto module = parent->getParentOfType<ModuleOp>();
          auto tmp = builder.create<cudaq::cc::AllocaOp>(loc, quakeTy);
          auto cast = builder.create<cudaq::cc::CastOp>(
              loc, cudaq::cc::PointerType::get(arg.getType()), tmp);
          if (cudaq::opt::factory::isX86_64(module)) {
            builder.create<cudaq::cc::StoreOp>(loc, arg, cast);
            if (cudaq::opt::factory::structUsesTwoArguments(quakeTy)) {
              auto arrTy = cudaq::cc::ArrayType::get(builder.getI8Type());
              auto cast = builder.create<cudaq::cc::CastOp>(
                  loc, cudaq::cc::PointerType::get(arrTy), tmp);
              auto hiPtr = builder.create<cudaq::cc::ComputePtrOp>(
                  loc, cudaq::cc::PointerType::get(builder.getI8Type()), cast,
                  cudaq::cc::ComputePtrArg{8});
              ++iter;
              Value nextArg = *iter;
              auto cast2 = builder.create<cudaq::cc::CastOp>(
                  loc, cudaq::cc::PointerType::get(nextArg.getType()), hiPtr);
              builder.create<cudaq::cc::StoreOp>(loc, nextArg, cast2);
            }
          } else {
            builder.create<cudaq::cc::StoreOp>(loc, arg, cast);
          }
          // Load the assembled (sub-)struct and insert into the buffer value.
          Value v = builder.create<cudaq::cc::LoadOp>(loc, tmp);
          stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                           stVal, v, idx);
          continue;
        }
        if (!cudaq::cc::isDynamicType(strTy)) {
          // struct is static size, so just load the value (byval ptr).
          Value v = builder.create<cudaq::cc::LoadOp>(loc, arg);
          stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                           stVal, v, idx);
          continue;
        }
        auto genTy = cast<cudaq::cc::StructType>(
            cudaq::opt::factory::genArgumentBufferType(strTy));
        Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
        auto [quakeVal, recursiveSize] = computeRecursiveDynamicStructSize(
            loc, builder, strTy, arg, zero, genTy);
        stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                         stVal, quakeVal, idx);
        extraBytes =
            builder.create<arith::AddIOp>(loc, extraBytes, recursiveSize);
        hasTrailingData = true;
        continue;
      }
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(inTy)) {
        if (isa<cudaq::cc::StateType>(ptrTy.getElementType())) {
          // Special case: if the argument is a `cudaq::state*`, then just pass
          // the pointer. We can do that in this case because the synthesis step
          // (which will receive the argument data) is assumed to run in the
          // same memory space.
          Value argPtr = builder.create<cudaq::cc::CastOp>(loc, inTy, arg);
          stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                           stVal, argPtr, idx);
        }
        continue;
      }

      stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                       stVal, arg, idx);
    }

    // Compute the struct size without the trailing bytes, structSize, and with
    // the trailing bytes, extendedStructSize.
    auto nullSt = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, zero);
    Value structSize =
        builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, structTy);
    Value extendedStructSize =
        builder.create<arith::AddIOp>(loc, structSize, extraBytes);

    // Allocate our struct to save the argument to.
    auto i8Ty = builder.getI8Type();
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto buff =
        builder.create<cudaq::cc::AllocaOp>(loc, i8Ty, extendedStructSize);

    Value temp = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, buff);

    // Store the arguments to the argument section.
    builder.create<cudaq::cc::StoreOp>(loc, stVal, temp);

    auto structPtrArrTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(structTy));
    temp = builder.create<cudaq::cc::CastOp>(loc, structPtrArrTy, buff);

    // Append the vector data to the end of the struct.
    if (hasTrailingData) {
      Value vecToBuffer = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI8Ty, buff, SmallVector<Value>{structSize});
      // Ignore any hidden `this` argument.
      for (auto inp : llvm::enumerate(blockValues)) {
        Value arg = inp.value();
        Type inTy = arg.getType();
        std::int32_t idx = inp.index();
        Type quakeTy = funcTy.getInput(idx);
        if (auto stdvecTy = dyn_cast<cudaq::cc::SpanLikeType>(quakeTy)) {
          auto bytes = builder.create<cudaq::cc::ExtractValueOp>(
              loc, builder.getI64Type(), stVal, idx);
          assert(stdvecTy == funcTy.getInput(idx));
          auto ptrInTy = cast<cudaq::cc::PointerType>(inTy);
          vecToBuffer = encodeVectorData(loc, builder, bytes, stdvecTy, arg,
                                         vecToBuffer, ptrInTy);
          if (stdvecTy.getElementType() == builder.getI1Type()) {
            auto ptrI1Ty = cudaq::cc::PointerType::get(builder.getI1Type());
            auto heapPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, cudaq::cc::PointerType::get(ptrI1Ty), arg,
                ArrayRef<cudaq::cc::ComputePtrArg>{0});
            auto loadHeapPtr = builder.create<cudaq::cc::LoadOp>(loc, heapPtr);
            Value heapCast = builder.create<cudaq::cc::CastOp>(
                loc, cudaq::cc::PointerType::get(i8Ty), loadHeapPtr);
            builder.create<func::CallOp>(loc, std::nullopt, "free",
                                         ArrayRef<Value>{heapCast});
          }
        } else if (auto strTy = dyn_cast<cudaq::cc::StructType>(quakeTy)) {
          if (cudaq::cc::isDynamicType(strTy))
            vecToBuffer = encodeDynamicStructData(loc, builder, strTy, arg,
                                                  temp, vecToBuffer);
        }
      }
    }

    // Prepare to call the `launchKernel` runtime library entry point.
    Value loadKernName = builder.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(kernelNameObj.getType()),
        kernelNameObj.getSymName());
    Value loadThunk =
        builder.create<func::ConstantOp>(loc, thunkTy, thunk.getName());
    auto castLoadKernName =
        builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, loadKernName);
    auto castLoadThunk =
        builder.create<cudaq::cc::FuncToPtrOp>(loc, ptrI8Ty, loadThunk);
    auto castTemp = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, temp);

    auto resultOffset =
        genComputeReturnOffset(loc, builder, funcTy, structTy, nullSt);

    // Generate the call to `launchKernel`.
    builder.create<func::CallOp>(
        loc, std::nullopt, cudaq::runtime::launchKernelFuncName,
        ArrayRef<Value>{castLoadKernName, castLoadThunk, castTemp,
                        extendedStructSize, resultOffset});
    const bool hiddenSRet = cudaq::opt::factory::hasHiddenSRet(funcTy);

    // If and only if this kernel returns a value, unpack and load the
    // result value(s) from the struct returned by `launchKernel` and return
    // them to our caller.
    SmallVector<Value> results;
    const bool multiResult = funcTy.getResults().size() > 1;
    for (auto res : llvm::enumerate(funcTy.getResults())) {
      int off = res.index() + offset;
      if (auto vecTy = dyn_cast<cudaq::cc::SpanLikeType>(res.value())) {
        auto eleTy = vecTy.getElementType();
        auto ptrTy = cudaq::cc::PointerType::get(eleTy);
        auto gep0 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(ptrTy), temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off, 0});
        auto dataPtr = builder.create<cudaq::cc::LoadOp>(loc, gep0);
        auto lenPtrTy = cudaq::cc::PointerType::get(builder.getI64Type());
        auto gep1 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, lenPtrTy, temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off, 1});
        auto vecLen = builder.create<cudaq::cc::LoadOp>(loc, gep1);
        if (vecTy.getElementType() == builder.getI1Type()) {
          genStdvecBoolFromInitList(loc, builder,
                                    rewriteEntryBlock->getArguments().front(),
                                    dataPtr, vecLen);
        } else {
          cudaq::IRBuilder irBuilder(builder);
          Value tSize = irBuilder.getByteSizeOfType(loc, eleTy);
          if (!tSize) {
            TODO_loc(loc, "unhandled vector element type");
            return;
          }
          genStdvecTFromInitList(loc, builder,
                                 rewriteEntryBlock->getArguments().front(),
                                 dataPtr, tSize, vecLen);
        }
        offset++;
      } else {
        auto gep0 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(structTy.getMember(off)), temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off});
        auto gep = cudaq::opt::factory::createCast(
            builder, loc, cudaq::cc::PointerType::get(res.value()), gep0);
        Value loadVal = builder.create<cudaq::cc::LoadOp>(loc, gep);
        if (hiddenSRet) {
          auto sretPtr = [&]() -> Value {
            if (multiResult)
              return builder.create<cudaq::cc::ComputePtrOp>(
                  loc, cudaq::cc::PointerType::get(res.value()),
                  rewriteEntryBlock->getArguments().front(),
                  SmallVector<cudaq::cc::ComputePtrArg>{off});
            return builder.create<cudaq::cc::CastOp>(
                loc, cudaq::cc::PointerType::get(res.value()),
                rewriteEntryBlock->getArguments().front());
          }();
          builder.create<cudaq::cc::StoreOp>(loc, loadVal, sretPtr);
        } else {
          results.push_back(loadVal);
        }
      }
    }
    builder.create<func::ReturnOp>(loc, results);
  }

  void genNewHostEntryPoint2(Location loc, OpBuilder &builder,
                             FunctionType devFuncTy,
                             LLVM::GlobalOp kernelNameObj,
                             func::FuncOp hostFunc, bool addThisPtr) {
    const bool hiddenSRet = cudaq::opt::factory::hasHiddenSRet(devFuncTy);
    const unsigned count =
        cudaq::cc::numberOfHiddenArgs(addThisPtr, hiddenSRet);
    auto *ctx = builder.getContext();
    auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());

    // 0) Pointer our builder into the entry block of the function.
    Block *hostFuncEntryBlock = hostFunc.addEntryBlock();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(hostFuncEntryBlock);

    // 1) Allocate and initialize a std::vector<void*> object.
    auto stdVec = builder.create<cudaq::cc::AllocaOp>(
        loc, cudaq::opt::factory::stlVectorType(i8PtrTy));
    auto arrPtrTy = cudaq::cc::ArrayType::get(ctx, i8PtrTy, count);
    Value buffer = builder.create<cudaq::cc::AllocaOp>(loc, arrPtrTy);
    auto i64Ty = builder.getI64Type();
    auto buffSize = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, arrPtrTy);
    auto ptrPtrTy = cudaq::cc::PointerType::get(i8PtrTy);
    auto cast1 = builder.create<cudaq::cc::CastOp>(loc, ptrPtrTy, buffer);
    auto ptr3Ty = cudaq::cc::PointerType::get(ptrPtrTy);
    auto stdVec0 = builder.create<cudaq::cc::CastOp>(loc, ptr3Ty, stdVec);
    builder.create<cudaq::cc::StoreOp>(loc, cast1, stdVec0);
    auto cast2 = builder.create<cudaq::cc::CastOp>(loc, i64Ty, buffer);
    auto endBuff = builder.create<arith::AddIOp>(loc, cast2, buffSize);
    auto cast3 = builder.create<cudaq::cc::CastOp>(loc, ptrPtrTy, endBuff);
    auto stdVec1 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptr3Ty, stdVec, ArrayRef<cudaq::cc::ComputePtrArg>{1});
    builder.create<cudaq::cc::StoreOp>(loc, cast3, stdVec1);
    auto stdVec2 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptr3Ty, stdVec, ArrayRef<cudaq::cc::ComputePtrArg>{2});
    builder.create<cudaq::cc::StoreOp>(loc, cast3, stdVec2);
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    auto nullPtr = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, zero);

    // 2) Iterate over the arguments passed in and populate the vector.
    SmallVector<BlockArgument> blockArgs{dropAnyHiddenArguments(
        hostFuncEntryBlock->getArguments(), devFuncTy, addThisPtr)};
    for (auto iter : llvm::enumerate(blockArgs)) {
      std::int32_t i = iter.index();
      auto pos = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrPtrTy, buffer, ArrayRef<cudaq::cc::ComputePtrArg>{i});
      auto blkArg = iter.value();
      if (isa<cudaq::cc::PointerType>(blkArg.getType())) {
        auto castArg = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, blkArg);
        builder.create<cudaq::cc::StoreOp>(loc, castArg, pos);
        continue;
      }
      auto temp = builder.create<cudaq::cc::AllocaOp>(loc, blkArg.getType());
      builder.create<cudaq::cc::StoreOp>(loc, blkArg, temp);
      auto castTemp = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, temp);
      builder.create<cudaq::cc::StoreOp>(loc, castTemp, pos);
    }

    auto resultBuffer = builder.create<cudaq::cc::AllocaOp>(loc, i8PtrTy);
    builder.create<cudaq::cc::StoreOp>(loc, nullPtr, resultBuffer);
    auto castResultBuffer =
        builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, resultBuffer);
    auto castStdvec = builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, stdVec);
    Value loadKernName = builder.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(kernelNameObj.getType()),
        kernelNameObj.getSymName());
    auto castKernelNameObj =
        builder.create<cudaq::cc::CastOp>(loc, i8PtrTy, loadKernName);
    builder.create<func::CallOp>(
        loc, std::nullopt, cudaq::runtime::launchKernelVersion2FuncName,
        ArrayRef<Value>{castKernelNameObj, castStdvec, castResultBuffer});

    // FIXME: Drop any results on the floor for now and return random data left
    // on the stack. (Maintains parity with existing kernel launch.)
    if (hostFunc.getFunctionType().getResults().empty()) {
      builder.create<func::ReturnOp>(loc);
      return;
    }
    // There can only be 1 return type in C++, so this is safe.
    Value garbage = builder.create<cudaq::cc::UndefOp>(
        loc, hostFunc.getFunctionType().getResult(0));
    builder.create<func::ReturnOp>(loc, garbage);
  }

  /// A kernel function that takes a quantum type argument (also known as a pure
  /// device kernel) cannot be called directly from C++ (classical) code. It
  /// must be called via other quantum code.
  bool hasLegalType(FunctionType funTy) {
    for (auto ty : funTy.getInputs())
      if (quake::isQuantumType(ty))
        return false;
    for (auto ty : funTy.getResults())
      if (quake::isQuantumType(ty))
        return false;
    return true;
  }

  void runOnOperation() override {
    auto module = getOperation();
    DataLayoutAnalysis dla(module); // caches module's data layout information.
    dataLayout = &dla.getAtOrAbove(module);
    std::error_code ec;
    llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Failed to open output file '" << outputFilename << "'\n";
      std::exit(ec.value());
    }
    auto *ctx = module.getContext();
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    auto mangledNameMap =
        module->getAttrOfType<DictionaryAttr>("quake.mangled_name_map");
    if (!mangledNameMap || mangledNameMap.empty())
      return;
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (altLaunchVersion == 1)
      if (failed(irBuilder.loadIntrinsic(
              module, cudaq::runtime::launchKernelFuncName))) {
        module.emitError("could not load altLaunchKernel intrinsic.");
        return;
      }
    if (altLaunchVersion == 2)
      if (failed(irBuilder.loadIntrinsic(
              module, cudaq::runtime::launchKernelVersion2FuncName))) {
        module.emitError("could not load altLaunchKernel intrinsic.");
        return;
      }

    auto loc = module.getLoc();
    auto ptrType = cudaq::cc::PointerType::get(builder.getI8Type());
    auto regKern = builder.create<func::FuncOp>(
        loc, cudaqRegisterKernelName, FunctionType::get(ctx, {ptrType}, {}));
    regKern.setPrivate();
    auto regArgs = builder.create<func::FuncOp>(
        loc, cudaqRegisterArgsCreator,
        FunctionType::get(ctx, {ptrType, ptrType}, {}));
    regArgs.setPrivate();

    if (failed(irBuilder.loadIntrinsic(module, "malloc"))) {
      module.emitError("could not load malloc");
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, "free"))) {
      module.emitError("could not load free");
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::stdvecBoolCtorFromInitList))) {
      module.emitError(std::string("could not load ") +
                       cudaq::stdvecBoolCtorFromInitList);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::stdvecBoolUnpackToInitList))) {
      module.emitError(std::string("could not load ") +
                       cudaq::stdvecBoolUnpackToInitList);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, cudaq::llvmMemCopyIntrinsic))) {
      module.emitError(std::string("could not load ") +
                       cudaq::llvmMemCopyIntrinsic);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_zeroDynamicResult"))) {
      module.emitError("could not load __nvqpp_zeroDynamicResult");
      return;
    }
    if (failed(
            irBuilder.loadIntrinsic(module, "__nvqpp_createDynamicResult"))) {
      module.emitError("could not load __nvqpp_createDynamicResult");
      return;
    }

    // Gather a work list of functions that are entry-point kernels.
    SmallVector<func::FuncOp> workList;
    for (auto &op : *module.getBody())
      if (auto funcOp = dyn_cast<func::FuncOp>(op))
        if (funcOp.getName().startswith(cudaq::runtime::cudaqGenPrefixName) &&
            hasLegalType(funcOp.getFunctionType()))
          workList.push_back(funcOp);

    LLVM_DEBUG(llvm::dbgs()
               << workList.size() << " kernel entry functions to process\n");
    for (auto funcOp : workList) {
      if (funcOp->hasAttr(cudaq::generatorAnnotation))
        continue;
      auto loc = funcOp.getLoc();
      [[maybe_unused]] auto className =
          funcOp.getName().drop_front(cudaq::runtime::cudaqGenPrefixLength);
      LLVM_DEBUG(llvm::dbgs() << "processing function " << className << '\n');
      auto classNameStr = className.str();

      // Create a constant with the name of the kernel as a C string.
      auto kernelNameObj = builder.create<LLVM::GlobalOp>(
          loc, cudaq::opt::factory::getStringType(ctx, className.size() + 1),
          /*isConstant=*/true, LLVM::Linkage::External,
          classNameStr + ".kernelName",
          builder.getStringAttr(classNameStr + '\0'), /*alignment=*/0);

      // Create a new struct type to pass arguments and results.
      auto funcTy = funcOp.getFunctionType();
      auto structTy = cudaq::opt::factory::buildInvokeStructType(funcTy);

      if (!mangledNameMap.contains(funcOp.getName()))
        continue;
      auto mangledAttr = mangledNameMap.getAs<StringAttr>(funcOp.getName());
      assert(mangledAttr && "funcOp must appear in mangled name map");
      StringRef mangledName = mangledAttr.getValue();
      auto [hostEntryNeeded, hostFunc] =
          lookupHostEntryPointFunc(mangledName, module, funcOp);
      FunctionType hostFuncTy;
      const bool hasThisPtr = !funcOp->hasAttr("no_this");
      if (hostEntryNeeded) {
        if (hostFunc) {
          hostFuncTy = hostFunc.getFunctionType();
        } else {
          // Fatal error was already raised in lookupHostEntryPointFunc().
          return;
        }
      } else {
        // Autogenerate an assumed host side function signature for the purpose
        // of constructing the argsCreator function.
        hostFuncTy =
            cudaq::opt::factory::toHostSideFuncType(funcTy, hasThisPtr, module);
      }

      func::FuncOp thunk;
      func::FuncOp argsCreatorFunc;

      if (altLaunchVersion == 1) {
        // Generate the function that computes the return offset.
        genReturnOffsetFunction(loc, builder, funcTy, structTy, classNameStr);

        // Generate thunk, `<kernel>.thunk`, to call back to the MLIR code.
        thunk = genThunkFunction(loc, builder, classNameStr, structTy, funcTy,
                                 funcOp);

        // Generate the argsCreator function used by synthesis.
        if (startingArgIdx == 0) {
          argsCreatorFunc = genKernelArgsCreatorFunction(
              loc, builder, funcTy, structTy, classNameStr, hostFuncTy,
              hasThisPtr);
        } else {
          // We are operating in a very special case where we want the
          // argsCreator function to ignore the first `startingArgIdx`
          // arguments. In this situation, the argsCreator function will not be
          // compatible with the other helper functions created in this pass, so
          // it is assumed that the caller is OK with that.
          auto structTy_argsCreator =
              cudaq::opt::factory::buildInvokeStructType(funcTy,
                                                         startingArgIdx);
          argsCreatorFunc = genKernelArgsCreatorFunction(
              loc, builder, funcTy, structTy_argsCreator, classNameStr,
              hostFuncTy, hasThisPtr);
        }
      }

      // Generate a new mangled function on the host side to call the
      // callback function.
      if (hostEntryNeeded) {
        if (altLaunchVersion == 1)
          genNewHostEntryPoint1(loc, builder, funcTy, structTy, kernelNameObj,
                                thunk, hostFunc, hasThisPtr);
        else
          genNewHostEntryPoint2(loc, builder, funcTy, kernelNameObj, hostFunc,
                                hasThisPtr);
      }

      // Generate a function at startup to register this kernel as having
      // been processed for kernel execution.
      auto initFun = builder.create<LLVM::LLVMFuncOp>(
          loc, classNameStr + ".kernelRegFunc",
          LLVM::LLVMFunctionType::get(cudaq::opt::factory::getVoidType(ctx),
                                      {}));
      {
        OpBuilder::InsertionGuard guard(builder);
        auto *initFunEntry = initFun.addEntryBlock();
        builder.setInsertionPointToStart(initFunEntry);
        auto kernRef = builder.create<LLVM::AddressOfOp>(
            loc, cudaq::opt::factory::getPointerType(kernelNameObj.getType()),
            kernelNameObj.getSymName());
        auto castKernRef =
            builder.create<cudaq::cc::CastOp>(loc, ptrType, kernRef);
        builder.create<func::CallOp>(loc, std::nullopt, cudaqRegisterKernelName,
                                     ValueRange{castKernRef});

        if (altLaunchVersion == 1) {
          // Register the argsCreator too
          auto ptrPtrType = cudaq::cc::PointerType::get(ptrType);
          auto argsCreatorFuncType = FunctionType::get(
              ctx, {ptrPtrType, ptrPtrType}, {builder.getI64Type()});
          Value loadArgsCreator = builder.create<func::ConstantOp>(
              loc, argsCreatorFuncType, argsCreatorFunc.getName());
          auto castLoadArgsCreator = builder.create<cudaq::cc::FuncToPtrOp>(
              loc, ptrType, loadArgsCreator);
          builder.create<func::CallOp>(
              loc, std::nullopt, cudaqRegisterArgsCreator,
              ValueRange{castKernRef, castLoadArgsCreator});
        }

        // Check if this is a lambda mangled name
        auto demangledPtr = abi::__cxa_demangle(mangledName.str().c_str(),
                                                nullptr, nullptr, nullptr);
        if (demangledPtr) {
          std::string demangledName(demangledPtr);
          demangledName = std::regex_replace(
              demangledName, std::regex("::operator()(.*)"), "");
          if (demangledName.find("$_") != std::string::npos) {
            auto insertPoint = builder.saveInsertionPoint();
            builder.setInsertionPointToStart(module.getBody());

            // Create the function if it doesn't already exist.
            if (!module.lookupSymbol<LLVM::LLVMFuncOp>(cudaqRegisterLambdaName))
              builder.create<LLVM::LLVMFuncOp>(
                  module.getLoc(), cudaqRegisterLambdaName,
                  LLVM::LLVMFunctionType::get(
                      cudaq::opt::factory::getVoidType(ctx),
                      {cudaq::opt::factory::getPointerType(ctx),
                       cudaq::opt::factory::getPointerType(ctx)}));

            // Create this global name, it is unique for any lambda
            // bc classNameStr contains the parentFunc + varName
            auto lambdaName = builder.create<LLVM::GlobalOp>(
                loc,
                cudaq::opt::factory::getStringType(ctx,
                                                   demangledName.size() + 1),
                /*isConstant=*/true, LLVM::Linkage::External,
                classNameStr + ".lambdaName",
                builder.getStringAttr(demangledName + '\0'), /*alignment=*/0);

            builder.restoreInsertionPoint(insertPoint);
            auto lambdaRef = builder.create<LLVM::AddressOfOp>(
                loc, cudaq::opt::factory::getPointerType(lambdaName.getType()),
                lambdaName.getSymName());

            auto castLambdaRef = builder.create<cudaq::cc::CastOp>(
                loc, cudaq::opt::factory::getPointerType(ctx), lambdaRef);
            auto castKernelRef = builder.create<cudaq::cc::CastOp>(
                loc, cudaq::opt::factory::getPointerType(ctx), castKernRef);
            builder.create<LLVM::CallOp>(
                loc, std::nullopt, cudaqRegisterLambdaName,
                ValueRange{castLambdaRef, castKernelRef});
          }
        }

        builder.create<LLVM::ReturnOp>(loc, ValueRange{});
      }

      // Create a global with a default ctor to be run at program startup.
      // The ctor will execute the above function, which will register this
      // kernel as having been processed.
      cudaq::opt::factory::createGlobalCtorCall(
          module, FlatSymbolRefAttr::get(ctx, initFun.getName()));
      LLVM_DEBUG(llvm::dbgs() << module << '\n');
    }
    out.keep();
  }

  const DataLayout *dataLayout = nullptr;
};
} // namespace
