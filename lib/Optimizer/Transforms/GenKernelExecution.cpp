/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
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

/// This value is used to indicate that a kernel does not return a result.
static constexpr std::uint64_t NoResultOffset =
    std::numeric_limits<std::int32_t>::max();

/// Generate code for packing arguments as raw data.
static bool isCodegenPackedData(std::size_t kind) {
  return kind == 0 || kind == 1;
}

/// Generate code that gathers the arguments for conversion and synthesis.
static bool isCodegenArgumentGather(std::size_t kind) {
  return kind == 0 || kind == 2;
}

static bool isStateType(Type ty) {
  if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty))
    return isa<cudaq::cc::StateType>(ptrTy.getElementType());
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
static FunctionType getThunkType(MLIRContext *ctx) {
  auto ptrTy = cudaq::cc::PointerType::get(IntegerType::get(ctx, 8));
  return FunctionType::get(ctx, {ptrTy, IntegerType::get(ctx, 1)},
                           {cudaq::opt::factory::getDynamicBufferType(ctx)});
}

/// Generate code to read the length from a host-side string object. (On the
/// device side, a string is encoded as a span.) The length of a string is the
/// number of bytes of data.
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
/// This implementation does \e not support wide characters.
static Value genStringLength(Location loc, OpBuilder &builder, Value stringArg,
                             ModuleOp module) {
  Type stringTy = stringArg.getType();
  assert(isa<cudaq::cc::PointerType>(stringTy) &&
         isa<cudaq::cc::ArrayType>(
             cast<cudaq::cc::PointerType>(stringTy).getElementType()) &&
         "host side string expected");
  auto callArg = builder.create<cudaq::cc::CastOp>(
      loc, cudaq::cc::PointerType::get(builder.getI8Type()), stringArg);
  StringRef helperName = module->getAttr(cudaq::runtime::sizeofStringAttrName)
                             ? cudaq::runtime::getPauliWordSize
                             : cudaq::runtime::bindingGetStringSize;
  auto lenRes = builder.create<func::CallOp>(loc, builder.getI64Type(),
                                             helperName, ValueRange{callArg});
  return lenRes.getResult(0);
}

/// Generate code that computes the size in bytes of a `std::vector<T>` array
/// in the same way as a `std::vector<T>::size()`. This assumes the vector is
/// laid out in memory as the following structure.
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
static Value genVectorSize(Location loc, OpBuilder &builder, Value vecArg) {
  auto vecTy = cast<cudaq::cc::PointerType>(vecArg.getType());
  auto vecStructTy = cast<cudaq::cc::StructType>(vecTy.getElementType());
  assert(vecStructTy.getNumMembers() == 3 &&
         vecStructTy.getMember(0) == vecStructTy.getMember(1) &&
         vecStructTy.getMember(0) == vecStructTy.getMember(2) &&
         "host side vector expected");
  auto vecElePtrTy = cudaq::cc::PointerType::get(vecStructTy.getMember(0));

  // Get the pointer to the pointer of the end of the array
  Value endPtr = builder.create<cudaq::cc::ComputePtrOp>(
      loc, vecElePtrTy, vecArg, ArrayRef<cudaq::cc::ComputePtrArg>{1});

  // Get the pointer to the pointer of the beginning of the array
  Value beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
      loc, vecElePtrTy, vecArg, ArrayRef<cudaq::cc::ComputePtrArg>{0});

  // Load to a T*
  endPtr = builder.create<cudaq::cc::LoadOp>(loc, endPtr);
  beginPtr = builder.create<cudaq::cc::LoadOp>(loc, beginPtr);

  // Map those pointers to integers
  Type i64Ty = builder.getI64Type();
  Value endInt = builder.create<cudaq::cc::CastOp>(loc, i64Ty, endPtr);
  Value beginInt = builder.create<cudaq::cc::CastOp>(loc, i64Ty, beginPtr);

  // Subtracting these will give us the size in bytes.
  return builder.create<arith::SubIOp>(loc, endInt, beginInt);
}

static Value genComputeReturnOffset(Location loc, OpBuilder &builder,
                                    FunctionType funcTy,
                                    cudaq::cc::StructType msgStructTy) {
  if (funcTy.getNumResults() == 0)
    return builder.create<arith::ConstantIntOp>(loc, NoResultOffset, 64);
  std::int32_t numKernelArgs = funcTy.getNumInputs();
  auto i64Ty = builder.getI64Type();
  return builder.create<cudaq::cc::OffsetOfOp>(
      loc, i64Ty, msgStructTy, ArrayRef<std::int32_t>{numKernelArgs});
}

/// Create a function that determines the return value offset in the message
/// buffer.
static void genReturnOffsetFunction(Location loc, OpBuilder &builder,
                                    FunctionType devKernelTy,
                                    cudaq::cc::StructType msgStructTy,
                                    const std::string &classNameStr) {
  auto *ctx = builder.getContext();
  auto i64Ty = builder.getI64Type();
  auto funcTy = FunctionType::get(ctx, {}, {i64Ty});
  auto returnOffsetFunc =
      builder.create<func::FuncOp>(loc, classNameStr + ".returnOffset", funcTy);
  OpBuilder::InsertionGuard guard(builder);
  auto *entry = returnOffsetFunc.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto result = genComputeReturnOffset(loc, builder, devKernelTy, msgStructTy);
  builder.create<func::ReturnOp>(loc, result);
}

static cudaq::cc::PointerType getByteAddressableType(OpBuilder &builder) {
  return cudaq::cc::PointerType::get(
      cudaq::cc::ArrayType::get(builder.getI8Type()));
}

static cudaq::cc::PointerType getPointerToPointerType(OpBuilder &builder) {
  return cudaq::cc::PointerType::get(
      cudaq::cc::PointerType::get(builder.getI8Type()));
}

static bool isDynamicSignature(FunctionType devFuncTy) {
  for (auto t : devFuncTy.getInputs())
    if (cudaq::cc::isDynamicType(t))
      return true;
  for (auto t : devFuncTy.getResults())
    if (cudaq::cc::isDynamicType(t))
      return true;
  return false;
}

static std::pair<Value, Value>
genByteSizeAndElementCount(Location loc, OpBuilder &builder, ModuleOp module,
                           Type eleTy, Value size, Value arg, Type t) {
  // If this is a vector<vector<...>>, convert the bytes of vector to bytes of
  // length (i64).
  if (auto sty = dyn_cast<cudaq::cc::StdvecType>(eleTy)) {
    auto eTy = cast<cudaq::cc::PointerType>(arg.getType()).getElementType();
    auto fTy = cast<cudaq::cc::StructType>(eTy).getMember(0);
    auto tTy = cast<cudaq::cc::PointerType>(fTy).getElementType();
    auto i64Ty = builder.getI64Type();
    auto eleSize = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, tTy);
    Value count = builder.create<arith::DivSIOp>(loc, size, eleSize);
    auto ate = builder.create<arith::ConstantIntOp>(loc, 8, 64);
    size = builder.create<arith::MulIOp>(loc, count, ate);
    return {size, count};
  }

  // If this is a vector<string>, convert the bytes of string to bytes of length
  // (i64).
  if (isa<cudaq::cc::CharspanType>(eleTy)) {
    auto arrTy = cudaq::opt::factory::genHostStringType(module);
    auto words =
        builder.create<arith::ConstantIntOp>(loc, arrTy.getSize() / 8, 64);
    size = builder.create<arith::DivSIOp>(loc, size, words);
    auto ate = builder.create<arith::ConstantIntOp>(loc, 8, 64);
    Value count = builder.create<arith::DivSIOp>(loc, size, ate);
    return {size, count};
  }

  // If this is a vector<struct<...>>, convert the bytes of struct to bytes of
  // struct with converted members.
  if (isa<cudaq::cc::StructType>(eleTy)) {
    auto vecTy = cast<cudaq::cc::PointerType>(arg.getType()).getElementType();
    auto vecEleRefTy = cast<cudaq::cc::StructType>(vecTy).getMember(0);
    auto vecEleTy = cast<cudaq::cc::PointerType>(vecEleRefTy).getElementType();
    auto i64Ty = builder.getI64Type();
    auto hostStrSize =
        builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, vecEleTy);
    Value count = builder.create<arith::DivSIOp>(loc, size, hostStrSize);
    Type packedTy = cudaq::opt::factory::genArgumentBufferType(eleTy);
    auto packSize = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, packedTy);
    size = builder.create<arith::MulIOp>(loc, count, packSize);
    return {size, count};
  }
  return {};
}

static bool isStdVectorBool(Type ty) {
  auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(ty);
  return stdvecTy &&
         (stdvecTy.getElementType() == IntegerType::get(ty.getContext(), 1));
}

/// Recursively check if \p ty contains a `std::vector<bool>`.
static bool hasStdVectorBool(Type ty) {
  if (isStdVectorBool(ty))
    return true;
  if (auto sty = dyn_cast<cudaq::cc::StdvecType>(ty))
    return hasStdVectorBool(sty.getElementType());
  if (auto sty = dyn_cast<cudaq::cc::StructType>(ty))
    for (auto mem : sty.getMembers())
      if (hasStdVectorBool(mem))
        return true;
  return false;
}

// The host-side type of a `std::vector<bool>` is distinct from the transient
// type for a `std::vector<bool>`. The former is a unique data type with a size
// of 40 bytes. The latter is identical to `std::vector<char>` (which has a size
// of 24 bytes).
static Type convertToTransientType(Type ty, ModuleOp mod) {
  if (isStdVectorBool(ty)) {
    auto *ctx = ty.getContext();
    return cudaq::opt::factory::stlVectorType(IntegerType::get(ctx, 1));
  }
  if (auto sty = dyn_cast<cudaq::cc::StdvecType>(ty))
    return cudaq::opt::factory::stlVectorType(
        convertToTransientType(sty.getElementType(), mod));
  if (auto sty = dyn_cast<cudaq::cc::StructType>(ty)) {
    SmallVector<Type> newMems;
    for (auto mem : sty.getMembers())
      newMems.push_back(convertToTransientType(mem, mod));
    auto *ctx = ty.getContext();
    return cudaq::cc::StructType::get(ctx, newMems);
  }
  return cudaq::opt::factory::convertToHostSideType(ty, mod);
}

static std::pair<Value, bool>
convertAllStdVectorBool(Location loc, OpBuilder &builder, ModuleOp module,
                        Value arg, Type ty, Value heapTracker,
                        std::optional<Value> preallocated = std::nullopt) {
  // If we are here, `ty` must be a `std::vector<bool>` or recursively contain a
  // `std::vector<bool>`.

  // Handle `std::vector<bool>`.
  if (isStdVectorBool(ty)) {
    auto stdvecTy = cast<cudaq::cc::StdvecType>(ty);
    Type stdvecHostTy =
        cudaq::opt::factory::stlVectorType(stdvecTy.getElementType());
    Value tmp = preallocated.has_value()
                    ? *preallocated
                    : builder.create<cudaq::cc::AllocaOp>(loc, stdvecHostTy);
    builder.create<func::CallOp>(loc, std::nullopt,
                                 cudaq::stdvecBoolUnpackToInitList,
                                 ArrayRef<Value>{tmp, arg, heapTracker});
    return {tmp, true};
  }

  // Handle `std::vector<T>` where `T` != `bool`.
  if (auto sty = dyn_cast<cudaq::cc::StdvecType>(ty)) {
    // arg is a std::vector<T>.
    // It's type must be ptr<struct<ptr<T>, ptr<T>, ptr<T>>>.
    auto seleTy = sty.getElementType();
    auto ptrArgTy = cast<cudaq::cc::PointerType>(arg.getType());
    auto argVecTy = cast<cudaq::cc::StructType>(ptrArgTy.getElementType());
    auto subVecPtrTy = cudaq::cc::PointerType::get(argVecTy.getMember(0));
    // Compute the pointer to the pointer to the first T element.
    auto inputRef = builder.create<cudaq::cc::ComputePtrOp>(
        loc, subVecPtrTy, arg, ArrayRef<cudaq::cc::ComputePtrArg>{0});
    auto startInput = builder.create<cudaq::cc::LoadOp>(loc, inputRef);
    auto startTy = startInput.getType();
    auto subArrTy = cudaq::cc::ArrayType::get(
        cast<cudaq::cc::PointerType>(startTy).getElementType());
    auto input = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(subArrTy), startInput);
    auto transientTy = convertToTransientType(sty, module);
    auto tmp = [&]() -> Value {
      if (preallocated)
        return builder.create<cudaq::cc::CastOp>(
            loc, cudaq::cc::PointerType::get(transientTy), *preallocated);
      return builder.create<cudaq::cc::AllocaOp>(loc, transientTy);
    }();
    Value sizeDelta = genVectorSize(loc, builder, arg);
    auto count = [&]() -> Value {
      if (cudaq::cc::isDynamicType(seleTy)) {
        auto p = genByteSizeAndElementCount(loc, builder, module, seleTy,
                                            sizeDelta, arg, sty);
        return p.second;
      }
      auto sizeEle = builder.create<cudaq::cc::SizeOfOp>(
          loc, builder.getI64Type(), seleTy);
      return builder.create<arith::DivSIOp>(loc, sizeDelta, sizeEle);
    }();
    auto transEleTy = cast<cudaq::cc::StructType>(transientTy).getMember(0);
    auto dataTy = cast<cudaq::cc::PointerType>(transEleTy).getElementType();
    auto sizeTransientTy =
        builder.create<cudaq::cc::SizeOfOp>(loc, builder.getI64Type(), dataTy);
    Value sizeInBytes =
        builder.create<arith::MulIOp>(loc, count, sizeTransientTy);

    // Create a new vector that we'll store the converted data into.
    Value byteBuffer = builder.create<cudaq::cc::AllocaOp>(
        loc, builder.getI8Type(), sizeInBytes);

    // Initialize the temporary vector.
    auto vecEleTy = cudaq::cc::PointerType::get(transEleTy);
    auto tmpBegin = builder.create<cudaq::cc::ComputePtrOp>(
        loc, vecEleTy, tmp, ArrayRef<cudaq::cc::ComputePtrArg>{0});
    auto bufferBegin =
        builder.create<cudaq::cc::CastOp>(loc, transEleTy, byteBuffer);
    builder.create<cudaq::cc::StoreOp>(loc, bufferBegin, tmpBegin);
    auto tmpEnd = builder.create<cudaq::cc::ComputePtrOp>(
        loc, vecEleTy, tmp, ArrayRef<cudaq::cc::ComputePtrArg>{1});
    auto byteBufferEnd = builder.create<cudaq::cc::ComputePtrOp>(
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), byteBuffer,
        ArrayRef<cudaq::cc::ComputePtrArg>{sizeInBytes});
    auto bufferEnd =
        builder.create<cudaq::cc::CastOp>(loc, transEleTy, byteBufferEnd);
    builder.create<cudaq::cc::StoreOp>(loc, bufferEnd, tmpEnd);
    auto tmpEnd2 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, vecEleTy, tmp, ArrayRef<cudaq::cc::ComputePtrArg>{2});
    builder.create<cudaq::cc::StoreOp>(loc, bufferEnd, tmpEnd2);

    // Loop over each element in the outer vector and initialize it to the inner
    // vector value. The data may be heap allocated.)
    auto transientEleTy = convertToTransientType(seleTy, module);
    auto transientBufferTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(transientEleTy));
    auto buffer =
        builder.create<cudaq::cc::CastOp>(loc, transientBufferTy, byteBuffer);

    cudaq::opt::factory::createInvariantLoop(
        builder, loc, count,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value i = block.getArgument(0);
          Value inp = builder.create<cudaq::cc::ComputePtrOp>(
              loc, startTy, input, ArrayRef<cudaq::cc::ComputePtrArg>{i});
          auto currentVector = builder.create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(transientEleTy), buffer,
              ArrayRef<cudaq::cc::ComputePtrArg>{i});
          convertAllStdVectorBool(loc, builder, module, inp, seleTy,
                                  heapTracker, currentVector);
        });
    return {tmp, true};
  }

  // Handle `struct { ... };`.
  if (auto sty = dyn_cast<cudaq::cc::StructType>(ty)) {
    auto bufferTy = convertToTransientType(ty, module);
    auto argPtrTy = cast<cudaq::cc::PointerType>(arg.getType());
    auto argStrTy = cast<cudaq::cc::StructType>(argPtrTy.getElementType());

    // If a struct was preallocated, use it. Otherwise, create a new struct that
    // we'll store the converted data into.
    auto buffer = [&]() -> Value {
      if (preallocated)
        return builder.create<cudaq::cc::CastOp>(
            loc, cudaq::cc::PointerType::get(bufferTy), *preallocated);
      return builder.create<cudaq::cc::AllocaOp>(loc, bufferTy);
    }();

    // Loop over each element. Replace each with the converted value.
    for (auto iter : llvm::enumerate(sty.getMembers())) {
      std::int32_t i = iter.index();
      Type memTy = iter.value();
      auto fromPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(argStrTy.getMember(i)), arg,
          ArrayRef<cudaq::cc::ComputePtrArg>{i});
      auto transientTy = convertToTransientType(memTy, module);
      Value toPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(transientTy), buffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{i});
      convertAllStdVectorBool(loc, builder, module, fromPtr, memTy, heapTracker,
                              toPtr);
    }
    return {buffer, true};
  }
  return {arg, false};
}

static std::pair<Value, bool>
unpackAnyStdVectorBool(Location loc, OpBuilder &builder, ModuleOp module,
                       Value arg, Type ty, Value heapTracker) {
  if (hasStdVectorBool(ty))
    return convertAllStdVectorBool(loc, builder, module, arg, ty, heapTracker);
  return {arg, false};
}

// Take the list of host-side arguments and device side argument types and zip
// them together logically with the position. Generates any fixup code that's
// needed, like when the device side uses a pair of arguments for a single
// logical device side argument. May drop some arguments on the floor if they
// cannot be encoded.
template <bool argsAreReferences>
static SmallVector<std::tuple<unsigned, Value, Type>>
zipArgumentsWithDeviceTypes(Location loc, OpBuilder &builder, ModuleOp module,
                            ValueRange args, TypeRange types,
                            Value heapTracker) {
  SmallVector<std::tuple<unsigned, Value, Type>> result;
  if constexpr (argsAreReferences) {
    // Simple case: the number of args must be equal to the types.
    assert(args.size() == types.size() &&
           "arguments and types must have same size");
    for (auto iter : llvm::enumerate(llvm::zip(args, types))) {
      // Remove the reference.
      Value v = std::get<Value>(iter.value());
      Type ty = std::get<Type>(iter.value());
      if (!(cudaq::cc::isDynamicType(ty) || isStateType(ty) ||
            isa<cudaq::cc::IndirectCallableType>(ty)))
        v = builder.create<cudaq::cc::LoadOp>(loc, v);
      // Python will pass a std::vector<bool> to us here. Unpack it.
      auto pear =
          unpackAnyStdVectorBool(loc, builder, module, v, ty, heapTracker);
      v = pear.first;
      result.emplace_back(iter.index(), v, ty);
    }
  } else /*constexpr*/ {
    // In this case, we *may* have logical arguments that are passed in pairs.
    auto *ctx = builder.getContext();
    auto *parent = builder.getBlock()->getParentOp();
    auto module = parent->getParentOfType<ModuleOp>();
    auto lastArg = args.end();
    auto tyIter = types.begin();
    unsigned argPos = 0;
    for (auto argIter = args.begin(); argIter != lastArg;
         ++argIter, ++tyIter, ++argPos) {
      assert(tyIter != types.end());
      Type devTy = *tyIter;

      // std::vector<bool> isn't really a std::vector<>. Use the helper
      // function to unpack it so it looks like any other vector.
      auto pear = unpackAnyStdVectorBool(loc, builder, module, *argIter, devTy,
                                         heapTracker);
      if (pear.second) {
        result.emplace_back(argPos, pear.first, devTy);
        continue;
      }

      // Check for a struct passed in a pair of arguments.
      if (isa<cudaq::cc::StructType>(devTy) &&
          !isa<cudaq::cc::PointerType>((*argIter).getType()) &&
          cudaq::opt::factory::isX86_64(module) &&
          cudaq::opt::factory::structUsesTwoArguments(devTy)) {
        auto first = *argIter++;
        auto second = *argIter;
        // TODO: Investigate if it's correct to assume the register layout
        // will match the memory layout of the small struct.
        auto pairTy = cudaq::cc::StructType::get(
            ctx, ArrayRef<Type>{first.getType(), second.getType()});
        auto tmp = builder.create<cudaq::cc::AllocaOp>(loc, pairTy);
        auto tmp1 = builder.create<cudaq::cc::CastOp>(
            loc, cudaq::cc::PointerType::get(first.getType()), tmp);
        builder.create<cudaq::cc::StoreOp>(loc, first, tmp1);
        auto tmp2 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(second.getType()), tmp,
            ArrayRef<cudaq::cc::ComputePtrArg>{1});
        builder.create<cudaq::cc::StoreOp>(loc, second, tmp2);
        auto devPtrTy = cudaq::cc::PointerType::get(devTy);
        Value devVal = builder.create<cudaq::cc::CastOp>(loc, devPtrTy, tmp);
        if (!cudaq::cc::isDynamicType(devTy))
          devVal = builder.create<cudaq::cc::LoadOp>(loc, devVal);
        result.emplace_back(argPos, devVal, devTy);
        continue;
      }

      // Is this a static struct passed as a byval pointer?
      if (isa<cudaq::cc::StructType>(devTy) &&
          isa<cudaq::cc::PointerType>((*argIter).getType()) &&
          !cudaq::cc::isDynamicType(devTy)) {
        Value devVal = builder.create<cudaq::cc::LoadOp>(loc, *argIter);
        result.emplace_back(argPos, devVal, devTy);
        continue;
      }
      result.emplace_back(argPos, *argIter, devTy);
    }
  }
  return result;
}

static Value descendThroughDynamicType(Location loc, OpBuilder &builder,
                                       ModuleOp module, Type ty, Value addend,
                                       Value arg, Value tmp) {
  auto i64Ty = builder.getI64Type();
  Value tySize =
      TypeSwitch<Type, Value>(ty)
          // A char span is dynamic, but it is not recursively dynamic. Just
          // read the length of the string out.
          .Case([&](cudaq::cc::CharspanType t) -> Value {
            return genStringLength(loc, builder, arg, module);
          })
          // A std::vector is dynamic and may be recursive dynamic as well.
          .Case([&](cudaq::cc::StdvecType t) -> Value {
            // Compute the byte span of the vector.
            Value size = genVectorSize(loc, builder, arg);
            auto eleTy = t.getElementType();
            if (!cudaq::cc::isDynamicType(eleTy))
              return size;

            // Otherwise, we have a recursively dynamic case.
            auto [bytes, count] = genByteSizeAndElementCount(
                loc, builder, module, eleTy, size, arg, t);
            assert(count && "vector must have elements");
            size = bytes;

            // At this point, arg is a known vector of elements of dynamic
            // type, so walk over the vector and recurse on each element.
            // `size` is already the proper size of the lengths of each of the
            // elements in turn.
            builder.create<cudaq::cc::StoreOp>(loc, size, tmp);
            auto ptrTy = cast<cudaq::cc::PointerType>(arg.getType());
            auto strTy = cast<cudaq::cc::StructType>(ptrTy.getElementType());
            auto memTy = cast<cudaq::cc::PointerType>(strTy.getMember(0));
            auto arrTy =
                cudaq::cc::PointerType::get(cudaq::cc::PointerType::get(
                    cudaq::cc::ArrayType::get(memTy.getElementType())));
            auto castPtr = builder.create<cudaq::cc::CastOp>(loc, arrTy, arg);
            auto castArg = builder.create<cudaq::cc::LoadOp>(loc, castPtr);
            auto castPtrTy =
                cudaq::cc::PointerType::get(memTy.getElementType());
            cudaq::opt::factory::createInvariantLoop(
                builder, loc, count,
                [&](OpBuilder &builder, Location loc, Region &, Block &block) {
                  Value i = block.getArgument(0);
                  auto ai = builder.create<cudaq::cc::ComputePtrOp>(
                      loc, castPtrTy, castArg,
                      ArrayRef<cudaq::cc::ComputePtrArg>{i});
                  auto tmpVal = builder.create<cudaq::cc::LoadOp>(loc, tmp);
                  Value innerSize = descendThroughDynamicType(
                      loc, builder, module, eleTy, tmpVal, ai, tmp);
                  builder.create<cudaq::cc::StoreOp>(loc, innerSize, tmp);
                });
            return builder.create<cudaq::cc::LoadOp>(loc, tmp);
          })
          // A struct can be dynamic if it contains dynamic members. Get the
          // static portion of the struct first, which will have length slots.
          // Then get the dynamic sizes for the dynamic members.
          .Case([&](cudaq::cc::StructType t) -> Value {
            if (cudaq::cc::isDynamicType(t)) {
              Type packedTy = cudaq::opt::factory::genArgumentBufferType(t);
              Value strSize =
                  builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, packedTy);
              for (auto iter : llvm::enumerate(t.getMembers())) {
                std::int32_t i = iter.index();
                auto m = iter.value();
                if (cudaq::cc::isDynamicType(m)) {
                  auto hostPtrTy = cast<cudaq::cc::PointerType>(arg.getType());
                  auto hostStrTy =
                      cast<cudaq::cc::StructType>(hostPtrTy.getElementType());
                  auto pm = cudaq::cc::PointerType::get(hostStrTy.getMember(i));
                  auto ai = builder.create<cudaq::cc::ComputePtrOp>(
                      loc, pm, arg, ArrayRef<cudaq::cc::ComputePtrArg>{i});
                  strSize = descendThroughDynamicType(loc, builder, module, m,
                                                      strSize, ai, tmp);
                }
              }
              return strSize;
            }
            return builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, t);
          })
          .Default([&](Type t) -> Value {
            return builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, t);
          });
  return builder.create<arith::AddIOp>(loc, tySize, addend);
}

static Value
genSizeOfDynamicMessageBuffer(Location loc, OpBuilder &builder, ModuleOp module,
                              cudaq::cc::StructType structTy,
                              ArrayRef<std::tuple<unsigned, Value, Type>> zippy,
                              Value tmp) {
  auto i64Ty = builder.getI64Type();
  Value initSize = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, structTy);
  for (auto [_, a, t] : zippy)
    if (cudaq::cc::isDynamicType(t))
      initSize =
          descendThroughDynamicType(loc, builder, module, t, initSize, a, tmp);
  return initSize;
}

static Value populateStringAddendum(Location loc, OpBuilder &builder,
                                    Value host, Value sizeSlot, Value addendum,
                                    ModuleOp module) {
  Value size = genStringLength(loc, builder, host, module);
  builder.create<cudaq::cc::StoreOp>(loc, size, sizeSlot);
  auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
  auto fromPtr = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, host);
  StringRef helperName = module->getAttr(cudaq::runtime::sizeofStringAttrName)
                             ? cudaq::runtime::getPauliWordData
                             : cudaq::runtime::bindingGetStringData;
  auto dataPtr = builder.create<func::CallOp>(loc, ptrI8Ty, helperName,
                                              ValueRange{fromPtr});
  auto notVolatile = builder.create<arith::ConstantIntOp>(loc, 0, 1);
  auto toPtr = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, addendum);
  builder.create<func::CallOp>(
      loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
      ValueRange{toPtr, dataPtr.getResult(0), size, notVolatile});
  auto ptrI8Arr = getByteAddressableType(builder);
  auto addBytes = builder.create<cudaq::cc::CastOp>(loc, ptrI8Arr, addendum);
  return builder.create<cudaq::cc::ComputePtrOp>(
      loc, ptrI8Ty, addBytes, ArrayRef<cudaq::cc::ComputePtrArg>{size});
}

// Simple case when the vector data is known to not hold dynamic data.
static Value populateVectorAddendum(Location loc, OpBuilder &builder,
                                    Value host, Value sizeSlot,
                                    Value addendum) {
  Value size = genVectorSize(loc, builder, host);
  builder.create<cudaq::cc::StoreOp>(loc, size, sizeSlot);
  auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
  auto ptrPtrI8 = getPointerToPointerType(builder);
  auto fromPtrPtr = builder.create<cudaq::cc::CastOp>(loc, ptrPtrI8, host);
  auto fromPtr = builder.create<cudaq::cc::LoadOp>(loc, fromPtrPtr);
  auto notVolatile = builder.create<arith::ConstantIntOp>(loc, 0, 1);
  auto toPtr = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, addendum);
  builder.create<func::CallOp>(loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
                               ValueRange{toPtr, fromPtr, size, notVolatile});
  auto ptrI8Arr = getByteAddressableType(builder);
  auto addBytes = builder.create<cudaq::cc::CastOp>(loc, ptrI8Arr, addendum);
  return builder.create<cudaq::cc::ComputePtrOp>(
      loc, ptrI8Ty, addBytes, ArrayRef<cudaq::cc::ComputePtrArg>{size});
}

static Value populateDynamicAddendum(Location loc, OpBuilder &builder,
                                     ModuleOp module, Type devArgTy, Value host,
                                     Value sizeSlot, Value addendum,
                                     Value addendumScratch) {
  if (isa<cudaq::cc::CharspanType>(devArgTy))
    return populateStringAddendum(loc, builder, host, sizeSlot, addendum,
                                  module);
  if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(devArgTy)) {
    auto eleTy = vecTy.getElementType();
    if (cudaq::cc::isDynamicType(eleTy)) {
      // Recursive case. Visit each dynamic element, copying it.
      Value size = genVectorSize(loc, builder, host);
      auto [bytes, count] = genByteSizeAndElementCount(
          loc, builder, module, eleTy, size, host, devArgTy);
      size = bytes;
      builder.create<cudaq::cc::StoreOp>(loc, size, sizeSlot);

      // Convert from bytes to vector length in elements.
      // Compute new addendum start.
      auto addrTy = getByteAddressableType(builder);
      auto castEnd = builder.create<cudaq::cc::CastOp>(loc, addrTy, addendum);
      Value newAddendum = builder.create<cudaq::cc::ComputePtrOp>(
          loc, addendum.getType(), castEnd,
          ArrayRef<cudaq::cc::ComputePtrArg>{size});
      builder.create<cudaq::cc::StoreOp>(loc, newAddendum, addendumScratch);
      Type dataTy = cudaq::opt::factory::genArgumentBufferType(eleTy);
      auto arrDataTy = cudaq::cc::ArrayType::get(dataTy);
      auto sizeBlockTy = cudaq::cc::PointerType::get(arrDataTy);
      auto ptrDataTy = cudaq::cc::PointerType::get(dataTy);

      // In the recursive case, the next block of addendum is a vector of
      // elements which are either sizes or contain sizes. The sizes are i64
      // and expressed in bytes. Each size will be the size of the span of the
      // element (or its subfields) at that offset.
      auto sizeBlock =
          builder.create<cudaq::cc::CastOp>(loc, sizeBlockTy, addendum);
      auto hostEleTy =
          cast<cudaq::cc::PointerType>(host.getType()).getElementType();
      auto ptrPtrBlockTy = cudaq::cc::PointerType::get(
          cast<cudaq::cc::StructType>(hostEleTy).getMember(0));

      // The host argument is a std::vector, so we want to get the address of
      // "front" out of the vector (the first pointer in the triple) and step
      // over the contiguous range of vectors in the host block. The vector of
      // vectors forms a ragged array structure in host memory.
      auto hostBeginPtrRef = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrPtrBlockTy, host, ArrayRef<cudaq::cc::ComputePtrArg>{0});
      auto hostBegin = builder.create<cudaq::cc::LoadOp>(loc, hostBeginPtrRef);
      auto hostBeginEleTy = cast<cudaq::cc::PointerType>(hostBegin.getType());
      auto hostBlockTy = cudaq::cc::PointerType::get(
          cudaq::cc::ArrayType::get(hostBeginEleTy.getElementType()));
      auto hostBlock =
          builder.create<cudaq::cc::CastOp>(loc, hostBlockTy, hostBegin);

      // Loop over each vector element in the vector (recursively).
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, count,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = block.getArgument(0);
            Value addm =
                builder.create<cudaq::cc::LoadOp>(loc, addendumScratch);
            auto subSlot = builder.create<cudaq::cc::ComputePtrOp>(
                loc, ptrDataTy, sizeBlock,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            auto subHost = builder.create<cudaq::cc::ComputePtrOp>(
                loc, hostBeginEleTy, hostBlock,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            Value newAddm =
                populateDynamicAddendum(loc, builder, module, eleTy, subHost,
                                        subSlot, addm, addendumScratch);
            builder.create<cudaq::cc::StoreOp>(loc, newAddm, addendumScratch);
          });
      return builder.create<cudaq::cc::LoadOp>(loc, addendumScratch);
    }
    return populateVectorAddendum(loc, builder, host, sizeSlot, addendum);
  }
  auto devStrTy = cast<cudaq::cc::StructType>(devArgTy);
  auto hostStrTy = cast<cudaq::cc::StructType>(
      cast<cudaq::cc::PointerType>(sizeSlot.getType()).getElementType());
  assert(devStrTy.getNumMembers() == hostStrTy.getNumMembers());
  for (auto iter : llvm::enumerate(devStrTy.getMembers())) {
    std::int32_t iterIdx = iter.index();
    auto hostPtrTy = cast<cudaq::cc::PointerType>(host.getType());
    auto hostMemTy = cast<cudaq::cc::StructType>(hostPtrTy.getElementType())
                         .getMember(iterIdx);
    auto val = builder.create<cudaq::cc::ComputePtrOp>(
        loc, cudaq::cc::PointerType::get(hostMemTy), host,
        ArrayRef<cudaq::cc::ComputePtrArg>{iterIdx});
    Type iterTy = iter.value();
    if (cudaq::cc::isDynamicType(iterTy)) {
      Value fieldInSlot = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(builder.getI64Type()), sizeSlot,
          ArrayRef<cudaq::cc::ComputePtrArg>{iterIdx});
      addendum =
          populateDynamicAddendum(loc, builder, module, iterTy, val,
                                  fieldInSlot, addendum, addendumScratch);
    } else {
      Value fieldInSlot = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(iterTy), sizeSlot,
          ArrayRef<cudaq::cc::ComputePtrArg>{iterIdx});
      auto v = builder.create<cudaq::cc::LoadOp>(loc, val);
      builder.create<cudaq::cc::StoreOp>(loc, v, fieldInSlot);
    }
  }
  return addendum;
}

static void
populateMessageBuffer(Location loc, OpBuilder &builder, ModuleOp module,
                      Value msgBufferBase,
                      ArrayRef<std::tuple<unsigned, Value, Type>> zippy,
                      Value addendum = {}, Value addendumScratch = {}) {
  auto structTy = cast<cudaq::cc::StructType>(
      cast<cudaq::cc::PointerType>(msgBufferBase.getType()).getElementType());
  // Loop over all the arguments and populate the message buffer.
  for (auto [idx, arg, devArgTy] : zippy) {
    std::int32_t i = idx;
    if (cudaq::cc::isDynamicType(devArgTy)) {
      assert(addendum && "must have addendum to encode dynamic argument(s)");
      // Get the address of the slot to be filled.
      auto memberTy = cast<cudaq::cc::StructType>(structTy).getMember(i);
      auto ptrTy = cudaq::cc::PointerType::get(memberTy);
      auto slot = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrTy, msgBufferBase, ArrayRef<cudaq::cc::ComputePtrArg>{i});
      addendum = populateDynamicAddendum(loc, builder, module, devArgTy, arg,
                                         slot, addendum, addendumScratch);
      continue;
    }

    // If the argument is a callable, skip it.
    if (isa<cudaq::cc::CallableType>(devArgTy))
      continue;
    // If the argument is an empty struct, skip it.
    if (auto strTy = dyn_cast<cudaq::cc::StructType>(devArgTy);
        strTy && strTy.isEmpty())
      continue;

    // Get the address of the slot to be filled.
    auto memberTy = cast<cudaq::cc::StructType>(structTy).getMember(i);
    auto ptrTy = cudaq::cc::PointerType::get(memberTy);
    Value slot = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTy, msgBufferBase, ArrayRef<cudaq::cc::ComputePtrArg>{i});

    // Argument is a packaged kernel. In this case, the argument is some
    // unknown kernel that may be called. The packaged argument is coming
    // from opaque C++ host code, so we need to identify what kernel it
    // references and then pass its name as a span of characters to the
    // launch kernel.
    if (isa<cudaq::cc::IndirectCallableType>(devArgTy)) {
      auto i64Ty = builder.getI64Type();
      auto kernKey = builder.create<func::CallOp>(
          loc, i64Ty, cudaq::runtime::getLinkableKernelKey, ValueRange{arg});
      builder.create<cudaq::cc::StoreOp>(loc, kernKey.getResult(0), slot);
      continue;
    }

    // Just pass the raw pointer. The buffer is supposed to be pointer-free
    // since it may be unpacked in a different address space. However, if this
    // is a simulation and things are in the same address space, we pass the
    // pointer for convenience.
    if (isa<cudaq::cc::PointerType>(devArgTy))
      arg = builder.create<cudaq::cc::CastOp>(loc, memberTy, arg);

    if (isa<cudaq::cc::StructType, cudaq::cc::ArrayType>(arg.getType()) &&
        (cudaq::cc::PointerType::get(arg.getType()) != slot.getType())) {
      slot = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(arg.getType()), slot);
    }
    builder.create<cudaq::cc::StoreOp>(loc, arg, slot);
  }
}

/// A kernel function that takes a quantum type argument (also known as a pure
/// device kernel) cannot be called directly from C++ (classical) code. It must
/// be called via other quantum code.
static bool hasLegalType(FunctionType funTy) {
  for (auto ty : funTy.getInputs())
    if (quake::isQuantumType(ty))
      return false;
  for (auto ty : funTy.getResults())
    if (quake::isQuantumType(ty))
      return false;
  return true;
}

static MutableArrayRef<BlockArgument>
dropAnyHiddenArguments(MutableArrayRef<BlockArgument> args, FunctionType funcTy,
                       bool hasThisPointer) {
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

/// Generate code to initialize the std::vector<T>, \p sret, from an initializer
/// list with data at \p data and length \p size. Use the library helper
/// routine. This function takes two !llvm.ptr arguments.
static void genStdvecBoolFromInitList(Location loc, OpBuilder &builder,
                                      Value sret, Value data, Value size) {
  auto ptrTy = cudaq::cc::PointerType::get(builder.getContext());
  auto castData = builder.create<cudaq::cc::CastOp>(loc, ptrTy, data);
  auto castSret = builder.create<cudaq::cc::CastOp>(loc, ptrTy, sret);
  builder.create<func::CallOp>(loc, std::nullopt,
                               cudaq::stdvecBoolCtorFromInitList,
                               ArrayRef<Value>{castSret, castData, size});
}

/// Generate a `std::vector<T>` (where `T != bool`) from an initializer list.
/// This is done with the assumption that `std::vector` is implemented as a
/// triple of pointers. The original content of the vector is freed and the new
/// content, which is already on the stack, is moved into the `std::vector`.
static void genStdvecTFromInitList(Location loc, OpBuilder &builder, Value sret,
                                   Value data, Value tSize, Value vecSize) {
  auto i8Ty = builder.getI8Type();
  auto stlVectorTy =
      cudaq::cc::PointerType::get(cudaq::opt::factory::stlVectorType(i8Ty));
  auto ptrTy = cudaq::cc::PointerType::get(i8Ty);
  auto castSret = builder.create<cudaq::cc::CastOp>(loc, stlVectorTy, sret);
  auto ptrPtrTy = cudaq::cc::PointerType::get(ptrTy);
  auto sret0 = builder.create<cudaq::cc::ComputePtrOp>(
      loc, ptrPtrTy, castSret, SmallVector<cudaq::cc::ComputePtrArg>{0});
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

// Alloca a pointer to a pointer and initialize it to nullptr.
static Value createEmptyHeapTracker(Location loc, OpBuilder &builder) {
  auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
  auto result = builder.create<cudaq::cc::AllocaOp>(loc, ptrI8Ty);
  auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  auto null = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, zero);
  builder.create<cudaq::cc::StoreOp>(loc, null, result);
  return result;
}

// If there are temporaries, call the helper to free them.
static void maybeFreeHeapAllocations(Location loc, OpBuilder &builder,
                                     Value heapTracker) {
  auto head = builder.create<cudaq::cc::LoadOp>(loc, heapTracker);
  auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  auto headAsInt =
      builder.create<cudaq::cc::CastOp>(loc, builder.getI64Type(), head);
  auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                           headAsInt, zero);
  // If there are no std::vector<bool> to unpack, then the heapTracker will be
  // set to `nullptr` and otherwise unused. That will allow the compiler to DCE
  // this call after constant propagation.
  builder.create<cudaq::cc::IfOp>(
      loc, TypeRange{}, cmp,
      [&](OpBuilder &builder, Location loc, Region &region) {
        region.push_back(new Block());
        auto &body = region.front();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body);
        builder.create<func::CallOp>(loc, std::nullopt,
                                     cudaq::stdvecBoolFreeTemporaryLists,
                                     ArrayRef<Value>{head});
        builder.create<cudaq::cc::ContinueOp>(loc);
      });
}

/// Fetch an argument from the comm buffer. Here, the argument is not dynamic so
/// it can be read as is out of the buffer.
static Value fetchInputValue(Location loc, OpBuilder &builder, Type devTy,
                             Value ptr) {
  assert(!cudaq::cc::isDynamicType(devTy) && "must not be a dynamic type");
  if (isa<cudaq::cc::IndirectCallableType>(devTy)) {
    // An indirect callable passes a key value which will be used to determine
    // the kernel that is being called.
    auto key = builder.create<cudaq::cc::LoadOp>(loc, ptr);
    return builder.create<cudaq::cc::CastOp>(loc, devTy, key);
  }

  if (isa<cudaq::cc::CallableType>(devTy)) {
    // A direct callable will have already been effectively inlined and this
    // argument should not be referenced.
    return builder.create<cudaq::cc::PoisonOp>(loc, devTy);
  }

  auto ptrDevTy = cudaq::cc::PointerType::get(devTy);
  if (auto strTy = dyn_cast<cudaq::cc::StructType>(devTy)) {
    // Argument is a struct.
    if (strTy.isEmpty())
      return builder.create<cudaq::cc::UndefOp>(loc, devTy);

    // Cast to avoid conflicts between layout compatible, distinct struct types.
    auto structPtr = builder.create<cudaq::cc::CastOp>(loc, ptrDevTy, ptr);
    return builder.create<cudaq::cc::LoadOp>(loc, structPtr);
  }

  // Default case: argument passed as a value inplace.
  return builder.create<cudaq::cc::LoadOp>(loc, ptr);
}

/// Helper routine to generate code to increment the trailing data pointer to
/// the next block of data (if any).
static Value incrementTrailingDataPointer(Location loc, OpBuilder &builder,
                                          Value trailingData, Value bytes) {
  auto i8Ty = builder.getI8Type();
  auto bufferTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty));
  auto buffPtr = builder.create<cudaq::cc::CastOp>(loc, bufferTy, trailingData);
  auto i8PtrTy = cudaq::cc::PointerType::get(i8Ty);
  return builder.create<cudaq::cc::ComputePtrOp>(
      loc, i8PtrTy, buffPtr, ArrayRef<cudaq::cc::ComputePtrArg>{bytes});
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
static std::pair<Value, Value> constructDynamicInputValue(Location loc,
                                                          OpBuilder &builder,
                                                          Type devTy, Value ptr,
                                                          Value trailingData) {
  assert(cudaq::cc::isDynamicType(devTy) && "must be dynamic type");
  // There are 2 cases.
  // 1. The dynamic type is a std::span of any legal device argument type.
  // 2. The dynamic type is a struct containing at least 1 std::span.
  if (auto spanTy = dyn_cast<cudaq::cc::SpanLikeType>(devTy)) {
    // ptr: a pointer to the length of the block in bytes.
    // trailingData: the block of data to decode.
    auto eleTy = spanTy.getElementType();
    auto i64Ty = builder.getI64Type();
    auto buffEleTy = cudaq::opt::factory::genArgumentBufferType(eleTy);

    // Get the size of each element in the vector and compute the vector's
    // logical length.
    auto eleSize = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, buffEleTy);
    Value bytes = builder.create<cudaq::cc::LoadOp>(loc, ptr);
    auto vecLength = builder.create<arith::DivSIOp>(loc, bytes, eleSize);

    if (cudaq::cc::isDynamicType(eleTy)) {
      // The vector is recursively dynamic.
      // Create a new block in which to place the stdvec/struct data in
      // device-side format.
      Value newVecData =
          builder.create<cudaq::cc::AllocaOp>(loc, eleTy, vecLength);
      // Compute new trailing data, skipping the current vector's data.
      auto nextTrailingData =
          incrementTrailingDataPointer(loc, builder, trailingData, bytes);

      // For each element in the vector, convert it to device-side format and
      // save the result in newVecData.
      auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
      auto packTy = cudaq::opt::factory::genArgumentBufferType(eleTy);
      Type packedArrTy =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(packTy));
      Type packedEleTy = cudaq::cc::PointerType::get(packTy);
      auto arrPtr =
          builder.create<cudaq::cc::CastOp>(loc, packedArrTy, trailingData);
      auto trailingDataVar =
          builder.create<cudaq::cc::AllocaOp>(loc, nextTrailingData.getType());
      builder.create<cudaq::cc::StoreOp>(loc, nextTrailingData,
                                         trailingDataVar);
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, vecLength,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = block.getArgument(0);
            auto nextTrailingData =
                builder.create<cudaq::cc::LoadOp>(loc, trailingDataVar);
            auto vecMemPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, packedEleTy, arrPtr,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            auto r = constructDynamicInputValue(loc, builder, eleTy, vecMemPtr,
                                                nextTrailingData);
            auto newVecPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, elePtrTy, newVecData,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            builder.create<cudaq::cc::StoreOp>(loc, r.first, newVecPtr);
            builder.create<cudaq::cc::StoreOp>(loc, r.second, trailingDataVar);
          });

      // Create the new outer stdvec span as the result.
      Value stdvecResult = builder.create<cudaq::cc::StdvecInitOp>(
          loc, spanTy, newVecData, vecLength);
      nextTrailingData =
          builder.create<cudaq::cc::LoadOp>(loc, trailingDataVar);
      return {stdvecResult, nextTrailingData};
    }

    // This vector has constant data, so just use the data in-place and
    // construct the stdvec span with it.
    auto castTrailingData = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(eleTy), trailingData);
    Value stdvecResult = builder.create<cudaq::cc::StdvecInitOp>(
        loc, spanTy, castTrailingData, vecLength);
    auto nextTrailingData =
        incrementTrailingDataPointer(loc, builder, trailingData, bytes);
    return {stdvecResult, nextTrailingData};
  }

  // Argument must be a struct.
  // The struct contains dynamic components. Extract them and build up the
  // struct value to be passed as an argument.
  // ptr: pointer to the first element of the struct or a vector length.
  // tailingData: the block of data for the first dynamic type field.
  auto strTy = cast<cudaq::cc::StructType>(devTy);
  auto ptrEleTy = cast<cudaq::cc::PointerType>(ptr.getType()).getElementType();
  auto packedTy = cast<cudaq::cc::StructType>(ptrEleTy);
  Value result = builder.create<cudaq::cc::UndefOp>(loc, strTy);
  assert(strTy.getNumMembers() == packedTy.getNumMembers());
  for (auto iter :
       llvm::enumerate(llvm::zip(strTy.getMembers(), packedTy.getMembers()))) {
    auto devMemTy = std::get<0>(iter.value());
    std::int32_t off = iter.index();
    auto packedMemTy = std::get<1>(iter.value());
    auto dataPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, cudaq::cc::PointerType::get(packedMemTy), ptr,
        ArrayRef<cudaq::cc::ComputePtrArg>{off});
    if (cudaq::cc::isDynamicType(devMemTy)) {
      auto r = constructDynamicInputValue(loc, builder, devMemTy, dataPtr,
                                          trailingData);
      result = builder.create<cudaq::cc::InsertValueOp>(loc, strTy, result,
                                                        r.first, off);
      trailingData = r.second;
      continue;
    }
    auto val = fetchInputValue(loc, builder, devMemTy, dataPtr);
    result =
        builder.create<cudaq::cc::InsertValueOp>(loc, strTy, result, val, off);
  }
  return {result, trailingData};
}

/// Translate the buffer data to a sequence of arguments suitable to the
/// actual kernel call.
///
/// \param inTy      The actual expected type of the argument.
/// \param structTy  The modified buffer type over all the arguments at the
/// current level.
static std::pair<Value, Value>
processInputValue(Location loc, OpBuilder &builder, Value trailingData,
                  Value ptrPackedStruct, Type inTy, std::int32_t off,
                  cudaq::cc::StructType packedStructTy) {
  auto packedPtr = builder.create<cudaq::cc::ComputePtrOp>(
      loc, cudaq::cc::PointerType::get(packedStructTy.getMember(off)),
      ptrPackedStruct, ArrayRef<cudaq::cc::ComputePtrArg>{off});
  if (cudaq::cc::isDynamicType(inTy))
    return constructDynamicInputValue(loc, builder, inTy, packedPtr,
                                      trailingData);
  auto val = fetchInputValue(loc, builder, inTy, packedPtr);
  return {val, trailingData};
}

/// This pass adds a `<kernel name>.thunk` function and a rewritten C++ host
/// side (mangled) stub to the code for every entry-point kernel in the module.
/// It may also generate a `<kernel name>.argsCreator` function. Finally, it
/// creates registration hooks for the CUDA-Q runtime to be able to find the
/// kernel by name and, as appropriate, the `<kernel name>.argsCreator`
/// function.
namespace {
class GenerateKernelExecution
    : public cudaq::opt::impl::GenerateKernelExecutionBase<
          GenerateKernelExecution> {
public:
  using GenerateKernelExecutionBase::GenerateKernelExecutionBase;

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
                                            ModuleOp module,
                                            FunctionType devKernelTy,
                                            cudaq::cc::StructType msgStructTy,
                                            const std::string &classNameStr,
                                            FunctionType hostFuncTy,
                                            bool hasThisPtr) {
    auto *ctx = builder.getContext();
    Type i8Ty = builder.getI8Type();
    Type ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto ptrPtrType = getPointerToPointerType(builder);
    Type i64Ty = builder.getI64Type();
    auto structPtrTy = cudaq::cc::PointerType::get(msgStructTy);
    auto passedDevArgTys = devKernelTy.getInputs().drop_front(startingArgIdx);

    SmallVector<Type> passedHostArgTys;
    for (auto ty : passedDevArgTys) {
      Type hostTy = cudaq::opt::factory::convertToHostSideType(ty, module);
      if (cudaq::cc::isDynamicType(ty))
        hostTy = cudaq::cc::PointerType::get(hostTy);
      passedHostArgTys.push_back(hostTy);
    }

    // Create the function that we'll fill.
    auto funcType = FunctionType::get(ctx, {ptrPtrType, ptrPtrType}, {i64Ty});
    auto argsCreatorFunc = builder.create<func::FuncOp>(
        loc, classNameStr + ".argsCreator", funcType);
    OpBuilder::InsertionGuard guard(builder);
    auto *entry = argsCreatorFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Convert all the arguments passed in the array of void* to appear as if
    // they had been naturally passed as C++ arguments.
    // This means, casting to the correct type (host-side) and removing the
    // outer pointer by a dereference. Each argument must be a valid reference
    // at this point, so if the dereference fails (say it is a nullptr), it is a
    // bug in the code that is calling this argsCreator.

    // Get the array of void* args.
    auto argsArray = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(ptrI8Ty)),
        entry->getArgument(0));

    // Loop over the array and cast the void* to the host-side type.
    SmallVector<Value> pseudoArgs;
    for (auto iter : llvm::enumerate(passedHostArgTys)) {
      std::int32_t i = iter.index();
      auto parg = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrPtrType, argsArray, ArrayRef<cudaq::cc::ComputePtrArg>{i});
      Type ty = iter.value();
      // parg is a pointer to a pointer as it is an element of an array of
      // pointers. Always dereference the first layer here.
      Value deref = builder.create<cudaq::cc::LoadOp>(loc, parg);
      if (!isa<cudaq::cc::PointerType>(ty))
        ty = cudaq::cc::PointerType::get(ty);
      pseudoArgs.push_back(builder.create<cudaq::cc::CastOp>(loc, ty, deref));
    }

    // Zip the arguments with the device side argument types. Recall that some
    // of the (left-most) arguments may have been dropped on the floor.
    const bool hasDynamicSignature = isDynamicSignature(devKernelTy);
    Value heapTracker = createEmptyHeapTracker(loc, builder);
    auto zippy = zipArgumentsWithDeviceTypes</*argsAreReferences=*/true>(
        loc, builder, module, pseudoArgs, passedDevArgTys, heapTracker);
    auto sizeScratch = builder.create<cudaq::cc::AllocaOp>(loc, i64Ty);
    auto messageBufferSize = [&]() -> Value {
      if (hasDynamicSignature)
        return genSizeOfDynamicMessageBuffer(loc, builder, module, msgStructTy,
                                             zippy, sizeScratch);
      return builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, msgStructTy);
    }();

    // Allocate the message buffer on the heap. It must outlive this call.
    auto buff = builder.create<func::CallOp>(loc, ptrI8Ty, "malloc",
                                             ValueRange(messageBufferSize));
    Value rawMessageBuffer = buff.getResult(0);
    Value msgBufferPrefix =
        builder.create<cudaq::cc::CastOp>(loc, structPtrTy, rawMessageBuffer);

    // Populate the message buffer with the pointer-free argument values.
    if (hasDynamicSignature) {
      auto addendumScratch = builder.create<cudaq::cc::AllocaOp>(loc, ptrI8Ty);
      Value prefixSize =
          builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, msgStructTy);
      auto arrMessageBuffer = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty)),
          rawMessageBuffer);
      // Compute the position of the addendum.
      Value addendumPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI8Ty, arrMessageBuffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
      populateMessageBuffer(loc, builder, module, msgBufferPrefix, zippy,
                            addendumPtr, addendumScratch);
    } else {
      populateMessageBuffer(loc, builder, module, msgBufferPrefix, zippy);
    }

    maybeFreeHeapAllocations(loc, builder, heapTracker);

    // Return the message buffer and its size in bytes.
    builder.create<cudaq::cc::StoreOp>(loc, rawMessageBuffer,
                                       entry->getArgument(1));
    builder.create<func::ReturnOp>(loc, ValueRange{messageBufferSize});

    // Note: the .argsCreator will have allocated space for a static result in
    // the message buffer. If the kernel returns a dynamic result, the launch
    // kernel code will have to properly return it in the appropriate context.
    return argsCreatorFunc;
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
    const std::int32_t offset = funcTy.getNumInputs();
    for (auto inp : llvm::enumerate(funcTy.getInputs())) {
      auto [a, t] = processInputValue(loc, builder, trailingData, castOp,
                                      inp.value(), inp.index(), structTy);
      trailingData = t;
      args.push_back(a);
    }
    auto call = builder.create<func::CallOp>(loc, funcTy.getResults(),
                                             funcOp.getName(), args);
    const bool hasVectorResult =
        funcTy.getNumResults() == 1 &&
        isa<cudaq::cc::SpanLikeType>(funcTy.getResult(0));
    if (hasVectorResult) {
      // If the original result was a std::vector<T>, then depending on whether
      // this is client-server or not, the thunk function packs the dynamic
      // return data into a message buffer or just returns a pointer to the
      // shared heap allocation, resp.
      auto *currentBlock = builder.getBlock();
      auto *reg = currentBlock->getParent();
      auto *thenBlock = builder.createBlock(reg);
      auto *elseBlock = builder.createBlock(reg);
      builder.setInsertionPointToEnd(currentBlock);
      auto eleTy = structTy.getMember(offset);
      auto memTy = cudaq::cc::PointerType::get(eleTy);
      auto mem = builder.create<cudaq::cc::ComputePtrOp>(
          loc, memTy, castOp, SmallVector<cudaq::cc::ComputePtrArg>{offset});
      auto resPtrTy = cudaq::cc::PointerType::get(call.getResult(0).getType());
      auto castMem = builder.create<cudaq::cc::CastOp>(loc, resPtrTy, mem);
      builder.create<cudaq::cc::StoreOp>(loc, call.getResult(0), castMem);
      builder.create<cf::CondBranchOp>(loc, isClientServer, thenBlock,
                                       elseBlock);
      builder.setInsertionPointToEnd(thenBlock);
      auto resAsArg = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(thunkTy.getResults()[0]), mem);
      auto retOffset = genComputeReturnOffset(loc, builder, funcTy, structTy);
      // createDynamicResult allocates a new buffer and packs the input values
      // and the dynamic results into this single new buffer to pass back as a
      // message.
      // NB: This code only handles one dimensional vectors of static types. It
      // will have to be changed if there is a need to return recursively
      // dynamic structures, i.e., vectors of vectors.
      auto res = builder.create<func::CallOp>(
          loc, thunkTy.getResults()[0], "__nvqpp_createDynamicResult",
          ValueRange{thunkEntry->getArgument(0), structSize, resAsArg,
                     retOffset});
      builder.create<func::ReturnOp>(loc, res.getResult(0));
      builder.setInsertionPointToEnd(elseBlock);
      // For the else case, the span was already copied to the block.
    } else {
      // FIXME: Should check for recursive vector case.
      // If the kernel returns non-dynamic results (no spans), then take those
      // values and store them in the results section of the struct. They will
      // eventually be returned to the original caller.
      if (funcTy.getNumResults()) {
        for (std::int32_t o = 0;
             o < static_cast<std::int32_t>(funcTy.getNumResults()); ++o) {
          auto eleTy = structTy.getMember(offset + o);
          auto memTy = cudaq::cc::PointerType::get(eleTy);
          auto mem = builder.create<cudaq::cc::ComputePtrOp>(
              loc, memTy, castOp,
              SmallVector<cudaq::cc::ComputePtrArg>{offset + o});
          auto resTy = call.getResult(o).getType();
          auto resPtrTy = cudaq::cc::PointerType::get(resTy);
          Value castMem = mem;
          if (resPtrTy != mem.getType())
            castMem = builder.create<cudaq::cc::CastOp>(loc, resPtrTy, mem);
          builder.create<cudaq::cc::StoreOp>(loc, call.getResult(o), castMem);
        }
      }
    }
    // zeroDynamicResult is used by models other than client-server. It assumes
    // that no messages need to be sent and that the CPU and QPU code share a
    // memory space. Therefore, making any copies can be skipped.
    auto zeroRes =
        builder.create<func::CallOp>(loc, thunkTy.getResults()[0],
                                     "__nvqpp_zeroDynamicResult", ValueRange{});
    builder.create<func::ReturnOp>(loc, zeroRes.getResult(0));
    return thunk;
  }

  /// Generate an all new entry point body, calling <i>some</i>LaunchKernel in
  /// the runtime library. Pass along the thunk, so the runtime can call the
  /// quantum circuit. These entry points may be `operator()` member functions
  /// in a class, so account for the `this` argument here.
  void genNewHostEntryPoint(Location loc, OpBuilder &builder, ModuleOp module,
                            FunctionType devFuncTy,
                            LLVM::GlobalOp kernelNameObj, func::FuncOp hostFunc,
                            bool addThisPtr, cudaq::cc::StructType structTy,
                            func::FuncOp thunkFunc) {
    auto *ctx = builder.getContext();
    auto i64Ty = builder.getI64Type();
    auto i8Ty = builder.getI8Type();
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto thunkTy = getThunkType(ctx);
    auto structPtrTy = cudaq::cc::PointerType::get(structTy);
    const std::int32_t offset = devFuncTy.getNumInputs();

    Block *hostFuncEntryBlock = hostFunc.addEntryBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(hostFuncEntryBlock);

    SmallVector<BlockArgument> blockArgs{dropAnyHiddenArguments(
        hostFuncEntryBlock->getArguments(), devFuncTy, addThisPtr)};
    SmallVector<Value> blockValues(blockArgs.size());
    std::copy(blockArgs.begin(), blockArgs.end(), blockValues.begin());
    const bool hasDynamicSignature = isDynamicSignature(devFuncTy);
    Value heapTracker = createEmptyHeapTracker(loc, builder);
    auto zippy = zipArgumentsWithDeviceTypes</*argsAreReferences=*/false>(
        loc, builder, module, blockValues, devFuncTy.getInputs(), heapTracker);
    auto sizeScratch = builder.create<cudaq::cc::AllocaOp>(loc, i64Ty);
    auto messageBufferSize = [&]() -> Value {
      if (hasDynamicSignature)
        return genSizeOfDynamicMessageBuffer(loc, builder, module, structTy,
                                             zippy, sizeScratch);
      return builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, structTy);
    }();

    Value msgBufferPrefix;
    Value castTemp;
    Value resultOffset;
    Value castLoadThunk;
    Value extendedStructSize;
    if (isCodegenPackedData(codegenKind)) {
      auto rawMessageBuffer =
          builder.create<cudaq::cc::AllocaOp>(loc, i8Ty, messageBufferSize);
      msgBufferPrefix =
          builder.create<cudaq::cc::CastOp>(loc, structPtrTy, rawMessageBuffer);

      if (hasDynamicSignature) {
        auto addendumScratch =
            builder.create<cudaq::cc::AllocaOp>(loc, ptrI8Ty);
        Value prefixSize =
            builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, structTy);
        Value addendumPtr = builder.create<cudaq::cc::ComputePtrOp>(
            loc, ptrI8Ty, rawMessageBuffer,
            ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
        populateMessageBuffer(loc, builder, module, msgBufferPrefix, zippy,
                              addendumPtr, addendumScratch);
      } else {
        populateMessageBuffer(loc, builder, module, msgBufferPrefix, zippy);
      }

      maybeFreeHeapAllocations(loc, builder, heapTracker);
      extendedStructSize = messageBufferSize;
      Value loadThunk =
          builder.create<func::ConstantOp>(loc, thunkTy, thunkFunc.getName());
      castLoadThunk =
          builder.create<cudaq::cc::FuncToPtrOp>(loc, ptrI8Ty, loadThunk);
      castTemp =
          builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, msgBufferPrefix);
      resultOffset = genComputeReturnOffset(loc, builder, devFuncTy, structTy);
    }

    Value vecArgPtrs;
    if (isCodegenArgumentGather(codegenKind)) {
      // 1) Allocate and initialize a std::vector<void*> object.
      const unsigned count = devFuncTy.getInputs().size();
      auto stdVec = builder.create<cudaq::cc::AllocaOp>(
          loc, cudaq::opt::factory::stlVectorType(ptrI8Ty));
      auto arrPtrTy = cudaq::cc::ArrayType::get(ctx, ptrI8Ty, count);
      Value buffer = builder.create<cudaq::cc::AllocaOp>(loc, arrPtrTy);
      auto buffSize = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, arrPtrTy);
      auto ptrPtrTy = cudaq::cc::PointerType::get(ptrI8Ty);
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

      // 2) Iterate over the arguments passed in and populate the vector.
      SmallVector<BlockArgument> blockArgs{dropAnyHiddenArguments(
          hostFuncEntryBlock->getArguments(), devFuncTy, addThisPtr)};
      unsigned j = 0;
      for (std::int32_t i = 0, N = blockArgs.size(); i < N; ++i, ++j) {
        auto blkArg = blockArgs[i];
        auto pos = builder.create<cudaq::cc::ComputePtrOp>(
            loc, ptrPtrTy, buffer, ArrayRef<cudaq::cc::ComputePtrArg>{i});
        if (isa<cudaq::cc::PointerType>(blkArg.getType())) {
          auto castArg =
              builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, blkArg);
          builder.create<cudaq::cc::StoreOp>(loc, castArg, pos);
          continue;
        }
        Value temp;
        if (cudaq::opt::factory::isX86_64(
                hostFunc->getParentOfType<ModuleOp>()) &&
            cudaq::opt::factory::structUsesTwoArguments(
                devFuncTy.getInput(j))) {
          temp =
              builder.create<cudaq::cc::AllocaOp>(loc, devFuncTy.getInput(j));
          auto part1 = builder.create<cudaq::cc::CastOp>(
              loc, cudaq::cc::PointerType::get(blkArg.getType()), temp);
          builder.create<cudaq::cc::StoreOp>(loc, blkArg, part1);
          auto blkArg2 = blockArgs[++i];
          auto cast2 = builder.create<cudaq::cc::CastOp>(
              loc,
              cudaq::cc::PointerType::get(
                  cudaq::cc::ArrayType::get(blkArg2.getType())),
              temp);
          auto part2 = builder.create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(blkArg2.getType()), cast2,
              ArrayRef<cudaq::cc::ComputePtrArg>{1});
          builder.create<cudaq::cc::StoreOp>(loc, blkArg2, part2);
        } else {
          temp = builder.create<cudaq::cc::AllocaOp>(loc, blkArg.getType());
          builder.create<cudaq::cc::StoreOp>(loc, blkArg, temp);
        }
        auto castTemp = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, temp);
        builder.create<cudaq::cc::StoreOp>(loc, castTemp, pos);
      }
      vecArgPtrs = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, stdVec);
    }

    // Prepare to call the `launchKernel` runtime library entry point.
    Value loadKernName = builder.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(kernelNameObj.getType()),
        kernelNameObj.getSymName());
    auto castLoadKernName =
        builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, loadKernName);

    auto hostFuncTy = hostFunc.getFunctionType();
    assert((hostFuncTy.getResults().empty() ||
            (hostFuncTy.getNumResults() == 1)) &&
           "C++ function expected to have 0 or 1 return value");
    const bool resultVal = !hostFuncTy.getResults().empty();
    const bool kernelReturnsValue =
        resultVal || cudaq::opt::factory::hasSRet(hostFunc);
    Value launchResult;
    Value launchResultToFree;
    auto decodeLaunchResults = [&](Value spanReturned) {
      if (!kernelReturnsValue)
        return;
      Type res0Ty = structTy.getMember(offset);
      auto ptrResTy = cudaq::cc::PointerType::get(res0Ty);
      auto rptr = builder.create<cudaq::cc::ExtractValueOp>(loc, ptrI8Ty,
                                                            spanReturned, 0);
      launchResultToFree = rptr;
      auto rIntPtr = builder.create<cudaq::cc::CastOp>(loc, i64Ty, rptr);
      auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
      auto cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                               rIntPtr, zero);
      auto *currentBlock = builder.getBlock();
      auto *reg = currentBlock->getParent();
      auto *thenBlock = builder.createBlock(reg);
      auto *elseBlock = builder.createBlock(reg);
      auto *endifBlock = builder.createBlock(
          reg, reg->end(), TypeRange{ptrResTy}, SmallVector<Location>(1, loc));
      builder.setInsertionPointToEnd(currentBlock);
      builder.create<cf::CondBranchOp>(loc, cmp, thenBlock, elseBlock);
      builder.setInsertionPointToEnd(thenBlock);
      // dynamic result was returned.
      // We need to free() this buffer before the end of this function.
      auto rStructPtr =
          builder.create<cudaq::cc::CastOp>(loc, structPtrTy, rptr);
      Value lRes = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrResTy, rStructPtr,
          ArrayRef<cudaq::cc::ComputePtrArg>{offset});
      builder.create<cf::BranchOp>(loc, endifBlock, ArrayRef<Value>{lRes});
      builder.setInsertionPointToEnd(elseBlock);
      // span was returned in the original buffer.
      Value mRes = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrResTy, msgBufferPrefix,
          ArrayRef<cudaq::cc::ComputePtrArg>{offset});
      builder.create<cf::BranchOp>(loc, endifBlock, ArrayRef<Value>{mRes});
      builder.setInsertionPointToEnd(endifBlock);
      launchResult = endifBlock->getArgument(0);
    };

    // Generate the call to `launchKernel`.
    switch (codegenKind) {
    case 0: {
      assert(vecArgPtrs && castLoadThunk);
      auto launch = builder.create<func::CallOp>(
          loc, cudaq::opt::factory::getDynamicBufferType(ctx),
          cudaq::runtime::launchKernelHybridFuncName,
          ArrayRef<Value>{castLoadKernName, castLoadThunk, castTemp,
                          extendedStructSize, resultOffset, vecArgPtrs});
      decodeLaunchResults(launch.getResult(0));
    } break;
    case 1: {
      assert(!vecArgPtrs && castLoadThunk);
      auto launch = builder.create<func::CallOp>(
          loc, cudaq::opt::factory::getDynamicBufferType(ctx),
          cudaq::runtime::launchKernelFuncName,
          ArrayRef<Value>{castLoadKernName, castLoadThunk, castTemp,
                          extendedStructSize, resultOffset});
      decodeLaunchResults(launch.getResult(0));
    } break;
    case 2: {
      assert(vecArgPtrs && !castLoadThunk);
      builder.create<func::CallOp>(
          loc, std::nullopt, cudaq::runtime::launchKernelStreamlinedFuncName,
          ArrayRef<Value>{castLoadKernName, vecArgPtrs});
      // For this codegen kind, we drop any results on the floor and return
      // random data in registers and/or off the stack. This maintains parity
      // with any pre-existing kernel launchers.
      SmallVector<Value> garbage;
      for (auto ty : hostFunc.getFunctionType().getResults())
        garbage.push_back(builder.create<cudaq::cc::UndefOp>(loc, ty));
      builder.create<func::ReturnOp>(loc, garbage);
      return;
    }
    default:
      hostFunc.emitOpError("codegen kind is invalid");
      return;
    }

    // If and only if this kernel returns a value, unpack and load the
    // result value(s) from the struct returned by `launchKernel` and return
    // them to our caller.
    SmallVector<Value> results;
    if (kernelReturnsValue) {
      Type res0Ty = structTy.getMember(offset);
      auto ptrResTy = cudaq::cc::PointerType::get(res0Ty);
      // Host function returns a value. Either returning by value or via an sret
      // reference.
      if (resultVal) {
        // Static values. std::vector are necessarily sret, see below.
        auto resPtr = builder.create<cudaq::cc::ComputePtrOp>(
            loc, ptrResTy, msgBufferPrefix,
            ArrayRef<cudaq::cc::ComputePtrArg>{offset});
        Type castToTy = cudaq::cc::PointerType::get(hostFuncTy.getResult(0));
        auto castResPtr = [&]() -> Value {
          if (castToTy == ptrResTy)
            return resPtr;
          return builder.create<cudaq::cc::CastOp>(loc, castToTy, resPtr);
        }();
        results.push_back(builder.create<cudaq::cc::LoadOp>(loc, castResPtr));
      } else {
        // This is an sret return. Check if device is returning a span. If it
        // is, then we will need to convert it to a std::vector here. The vector
        // is constructed in-place on the sret memory block.
        Value arg0 = hostFuncEntryBlock->getArguments().front();
        if (auto spanTy =
                dyn_cast<cudaq::cc::SpanLikeType>(devFuncTy.getResult(0))) {
          auto eleTy = spanTy.getElementType();
          auto ptrTy = cudaq::cc::PointerType::get(eleTy);
          auto gep0 = builder.create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(ptrTy), launchResult,
              SmallVector<cudaq::cc::ComputePtrArg>{0});
          auto dataPtr = builder.create<cudaq::cc::LoadOp>(loc, gep0);
          auto lenPtrTy = cudaq::cc::PointerType::get(i64Ty);
          auto gep1 = builder.create<cudaq::cc::ComputePtrOp>(
              loc, lenPtrTy, launchResult,
              SmallVector<cudaq::cc::ComputePtrArg>{1});
          auto vecLen = builder.create<cudaq::cc::LoadOp>(loc, gep1);
          if (spanTy.getElementType() == builder.getI1Type()) {
            genStdvecBoolFromInitList(loc, builder, arg0, dataPtr, vecLen);
          } else {
            Value tSize =
                builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, eleTy);
            genStdvecTFromInitList(loc, builder, arg0, dataPtr, tSize, vecLen);
          }
          // free(nullptr) is defined to be a nop in the standard.
          builder.create<func::CallOp>(loc, std::nullopt, "free",
                                       ArrayRef<Value>{launchResultToFree});
        } else {
          // Otherwise, we can just copy the aggregate into the sret memory
          // block. Uses the size of the host function's sret pointer element
          // type for the memcpy, so the device should return an (aggregate)
          // value of suitable size.
          auto resPtr = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrResTy, msgBufferPrefix,
              ArrayRef<cudaq::cc::ComputePtrArg>{offset});
          auto castMsgBuff =
              builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, resPtr);
          Type eleTy =
              cast<cudaq::cc::PointerType>(arg0.getType()).getElementType();
          Value bytes = builder.create<cudaq::cc::SizeOfOp>(loc, i64Ty, eleTy);
          auto notVolatile = builder.create<arith::ConstantIntOp>(loc, 0, 1);
          auto castArg0 = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, arg0);
          builder.create<func::CallOp>(
              loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
              ValueRange{castArg0, castMsgBuff, bytes, notVolatile});
        }
      }
    }

    // Return the result (if any).
    builder.create<func::ReturnOp>(loc, results);
  }

  /// Generate a function to be executed at load-time which will register the
  /// kernel with the runtime.
  LLVM::LLVMFuncOp registerKernelWithRuntimeForExecution(
      Location loc, OpBuilder &builder, const std::string &classNameStr,
      LLVM::GlobalOp kernelNameObj, func::FuncOp argsCreatorFunc,
      StringRef mangledName) {
    auto module = getOperation();
    auto *ctx = builder.getContext();
    auto ptrType = cudaq::cc::PointerType::get(builder.getI8Type());
    auto initFun = builder.create<LLVM::LLVMFuncOp>(
        loc, classNameStr + ".kernelRegFunc",
        LLVM::LLVMFunctionType::get(cudaq::opt::factory::getVoidType(ctx), {}));
    OpBuilder::InsertionGuard guard(builder);
    auto *initFunEntry = initFun.addEntryBlock();
    builder.setInsertionPointToStart(initFunEntry);
    auto kernRef = builder.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(kernelNameObj.getType()),
        kernelNameObj.getSymName());
    auto castKernRef = builder.create<cudaq::cc::CastOp>(loc, ptrType, kernRef);
    builder.create<func::CallOp>(loc, std::nullopt,
                                 cudaq::runtime::CudaqRegisterKernelName,
                                 ValueRange{castKernRef});

    if (isCodegenPackedData(codegenKind)) {
      // Register the argsCreator too
      auto ptrPtrType = cudaq::cc::PointerType::get(ptrType);
      auto argsCreatorFuncType = FunctionType::get(
          ctx, {ptrPtrType, ptrPtrType}, {builder.getI64Type()});
      Value loadArgsCreator = builder.create<func::ConstantOp>(
          loc, argsCreatorFuncType, argsCreatorFunc.getName());
      auto castLoadArgsCreator =
          builder.create<cudaq::cc::FuncToPtrOp>(loc, ptrType, loadArgsCreator);
      builder.create<func::CallOp>(
          loc, std::nullopt, cudaq::runtime::CudaqRegisterArgsCreator,
          ValueRange{castKernRef, castLoadArgsCreator});
    }

    // Check if this is a lambda mangled name
    auto demangledPtr = abi::__cxa_demangle(mangledName.str().c_str(), nullptr,
                                            nullptr, nullptr);
    if (demangledPtr) {
      std::string demangledName(demangledPtr);
      demangledName =
          std::regex_replace(demangledName, std::regex("::operator()(.*)"), "");
      if (demangledName.find("$_") != std::string::npos) {
        auto insertPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(module.getBody());

        // Create this global name, it is unique for any lambda
        // bc classNameStr contains the parentFunc + varName
        auto lambdaName = builder.create<LLVM::GlobalOp>(
            loc,
            cudaq::opt::factory::getStringType(ctx, demangledName.size() + 1),
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
        builder.create<LLVM::CallOp>(loc, std::nullopt,
                                     cudaq::runtime::CudaqRegisterLambdaName,
                                     ValueRange{castLambdaRef, castKernelRef});
      }
    }

    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
    return initFun;
  }

  // Load the prototypes of runtime functions that we may call into the Module.
  LogicalResult loadPrototypes() {
    ModuleOp module = getOperation();
    auto mangledNameMap =
        module->getAttrOfType<DictionaryAttr>(cudaq::runtime::mangledNameMap);
    if (!mangledNameMap || mangledNameMap.empty())
      return failure();
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    switch (codegenKind) {
    case 0:
      if (failed(irBuilder.loadIntrinsic(
              module, cudaq::runtime::launchKernelHybridFuncName)))
        return module.emitError("could not load altLaunchKernel intrinsic.");
      break;
    case 1:
      if (failed(irBuilder.loadIntrinsic(module,
                                         cudaq::runtime::launchKernelFuncName)))
        return module.emitError("could not load altLaunchKernel intrinsic.");
      break;
    case 2:
      if (failed(irBuilder.loadIntrinsic(
              module, cudaq::runtime::launchKernelStreamlinedFuncName)))
        return module.emitError("could not load altLaunchKernel intrinsic.");
      break;
    default:
      return module.emitError("invalid codegen kind value.");
    }

    if (failed(irBuilder.loadIntrinsic(
            module, cudaq::runtime::CudaqRegisterKernelName)))
      return module.emitError("could not load kernel registration API");

    if (failed(irBuilder.loadIntrinsic(module, "malloc")))
      return module.emitError("could not load malloc");
    if (failed(irBuilder.loadIntrinsic(module, "free")))
      return module.emitError("could not load free");
    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::stdvecBoolCtorFromInitList)))
      return module.emitError(std::string("could not load ") +
                              cudaq::stdvecBoolCtorFromInitList);
    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::stdvecBoolUnpackToInitList)))
      return module.emitError(std::string("could not load ") +
                              cudaq::stdvecBoolUnpackToInitList);
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::stdvecBoolFreeTemporaryLists)))
      return module.emitError(std::string("could not load ") +
                              cudaq::stdvecBoolFreeTemporaryLists);
    if (failed(irBuilder.loadIntrinsic(module, cudaq::llvmMemCopyIntrinsic)))
      return module.emitError(std::string("could not load ") +
                              cudaq::llvmMemCopyIntrinsic);
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_zeroDynamicResult")))
      return module.emitError("could not load __nvqpp_zeroDynamicResult");
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_createDynamicResult")))
      return module.emitError("could not load __nvqpp_createDynamicResult");
    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::runtime::getPauliWordSize)))
      return module.emitError(
          "could not load cudaq::pauli_word::_nvqpp_size or _nvqpp_data");
    return success();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    auto mangledNameMap =
        module->getAttrOfType<DictionaryAttr>(cudaq::runtime::mangledNameMap);
    std::error_code ec;
    llvm::ToolOutputFile out(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "Failed to open output file '" << outputFilename << "'\n";
      std::exit(ec.value());
    }

    if (failed(loadPrototypes()))
      return;

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

      if (isCodegenPackedData(codegenKind)) {
        // Generate the function that computes the return offset.
        genReturnOffsetFunction(loc, builder, funcTy, structTy, classNameStr);

        // Generate thunk, `<kernel>.thunk`, to call back to the MLIR code.
        thunk = genThunkFunction(loc, builder, classNameStr, structTy, funcTy,
                                 funcOp);

        // Generate the argsCreator function used by synthesis.
        if (startingArgIdx == 0) {
          argsCreatorFunc = genKernelArgsCreatorFunction(
              loc, builder, module, funcTy, structTy, classNameStr, hostFuncTy,
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
              loc, builder, module, funcTy, structTy_argsCreator, classNameStr,
              hostFuncTy, hasThisPtr);
        }
      }

      // Generate a new mangled function on the host side to call the
      // callback function.
      if (hostEntryNeeded)
        genNewHostEntryPoint(loc, builder, module, funcTy, kernelNameObj,
                             hostFunc, hasThisPtr, structTy, thunk);

      // Generate a function at startup to register this kernel as having
      // been processed for kernel execution.
      auto initFun = registerKernelWithRuntimeForExecution(
          loc, builder, classNameStr, kernelNameObj, argsCreatorFunc,
          mangledName);

      // Create a global with a default ctor to be run at program startup.
      // The ctor will execute the above function, which will register this
      // kernel as having been processed.
      cudaq::opt::factory::createGlobalCtorCall(
          module, FlatSymbolRefAttr::get(ctx, initFun.getName()));

      LLVM_DEBUG(llvm::dbgs() << "final module:\n" << module << '\n');
    }
    out.keep();
  }
};
} // namespace
