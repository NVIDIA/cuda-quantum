/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

/// Generate code to read the length from a host-side string object. (On the
/// device side, a string is encoded as a span.) The length of a string is the
/// number of bytes of data.
///
/// The layout of a string object depends on the standard header files being
/// used. To work around this, we use runtime callback functions which are
/// compiled using those standard headers.
///
/// This implementation does \e not support wide characters.
template <bool FromQPU>
Value genStringLength(Location loc, OpBuilder &builder, Value stringArg,
                      ModuleOp module) {
  if constexpr (FromQPU) {
    Type stringTy = stringArg.getType();
    assert(isa<cudaq::cc::CharspanType>(stringTy));
    return cudaq::cc::StdvecSizeOp::create(builder, loc, builder.getI64Type(),
                                                   stringArg);
  } else /*constexpr */ {
    Type stringTy = stringArg.getType();
    assert(isa<cudaq::cc::PointerType>(stringTy) &&
           isa<cudaq::cc::ArrayType>(
               cast<cudaq::cc::PointerType>(stringTy).getElementType()) &&
           "host side string expected");
    auto callArg = cudaq::cc::CastOp::create(builder, 
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), stringArg);
    StringRef helperName = module->getAttr(cudaq::runtime::sizeofStringAttrName)
                               ? cudaq::runtime::getPauliWordSize
                               : cudaq::runtime::bindingGetStringSize;
    auto lenRes = func::CallOp::create(builder, loc, builder.getI64Type(),
                                               helperName, ValueRange{callArg});
    return lenRes.getResult(0);
  }
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
template <bool FromQPU>
Value genVectorSize(Location loc, OpBuilder &builder, Value vecArg) {
  if constexpr (FromQPU) {
    Type vecArgTy = vecArg.getType();
    assert(isa<cudaq::cc::StdvecType>(vecArgTy));
    return cudaq::cc::StdvecSizeOp::create(builder, loc, builder.getI64Type(),
                                                   vecArg);
  } else /* constexpr */ {
    auto vecTy = cast<cudaq::cc::PointerType>(vecArg.getType());
    auto vecStructTy = cast<cudaq::cc::StructType>(vecTy.getElementType());
    assert(vecStructTy.getNumMembers() == 3 &&
           vecStructTy.getMember(0) == vecStructTy.getMember(1) &&
           vecStructTy.getMember(0) == vecStructTy.getMember(2) &&
           "host side vector expected");
    auto vecElePtrTy = cudaq::cc::PointerType::get(vecStructTy.getMember(0));

    // Get the pointer to the pointer of the end of the array
    Value endPtr = cudaq::cc::ComputePtrOp::create(builder, 
        loc, vecElePtrTy, vecArg, ArrayRef<cudaq::cc::ComputePtrArg>{1});

    // Get the pointer to the pointer of the beginning of the array
    Value beginPtr = cudaq::cc::ComputePtrOp::create(builder, 
        loc, vecElePtrTy, vecArg, ArrayRef<cudaq::cc::ComputePtrArg>{0});

    // Load to a T*
    endPtr = cudaq::cc::LoadOp::create(builder, loc, endPtr);
    beginPtr = cudaq::cc::LoadOp::create(builder, loc, beginPtr);

    // Map those pointers to integers
    Type i64Ty = builder.getI64Type();
    Value endInt = cudaq::cc::CastOp::create(builder, loc, i64Ty, endPtr);
    Value beginInt = cudaq::cc::CastOp::create(builder, loc, i64Ty, beginPtr);

    // Subtracting these will give us the size in bytes.
    return arith::SubIOp::create(builder, loc, endInt, beginInt);
  }
}

Value cudaq::opt::marshal::genComputeReturnOffset(
    Location loc, OpBuilder &builder, FunctionType funcTy,
    cudaq::cc::StructType msgStructTy) {
  if (funcTy.getNumResults() == 0)
    return arith::ConstantIntOp::create(builder, loc, NoResultOffset, 64);
  std::int32_t numKernelArgs = funcTy.getNumInputs();
  auto i64Ty = builder.getI64Type();
  return cc::OffsetOfOp::create(builder, loc, i64Ty, msgStructTy,
                                        ArrayRef<std::int32_t>{numKernelArgs});
}

void cudaq::opt::marshal::genReturnOffsetFunction(
    Location loc, OpBuilder &builder, FunctionType devKernelTy,
    cudaq::cc::StructType msgStructTy, const std::string &classNameStr) {
  auto *ctx = builder.getContext();
  auto i64Ty = builder.getI64Type();
  auto funcTy = FunctionType::get(ctx, {}, {i64Ty});
  auto returnOffsetFunc =
      func::FuncOp::create(builder, loc, classNameStr + ".returnOffset", funcTy);
  OpBuilder::InsertionGuard guard(builder);
  auto *entry = returnOffsetFunc.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  auto result = genComputeReturnOffset(loc, builder, devKernelTy, msgStructTy);
  func::ReturnOp::create(builder, loc, result);
}

static cudaq::cc::PointerType getByteAddressableType(OpBuilder &builder) {
  return cudaq::cc::PointerType::get(
      cudaq::cc::ArrayType::get(builder.getI8Type()));
}

cudaq::cc::PointerType
cudaq::opt::marshal::getPointerToPointerType(OpBuilder &builder) {
  return cc::PointerType::get(cc::PointerType::get(builder.getI8Type()));
}

bool cudaq::opt::marshal::isDynamicSignature(FunctionType devFuncTy) {
  for (auto t : devFuncTy.getInputs())
    if (cc::isDynamicType(t))
      return true;
  for (auto t : devFuncTy.getResults())
    if (cc::isDynamicType(t))
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
    auto eleSize = cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, tTy);
    Value count = arith::DivSIOp::create(builder, loc, size, eleSize);
    auto ate = arith::ConstantIntOp::create(builder, loc, 8, 64);
    size = arith::MulIOp::create(builder, loc, count, ate);
    return {size, count};
  }

  // If this is a vector<string>, convert the bytes of string to bytes of length
  // (i64).
  if (isa<cudaq::cc::CharspanType>(eleTy)) {
    auto arrTy = cudaq::opt::factory::genHostStringType(module);
    auto words =
        arith::ConstantIntOp::create(builder, loc, arrTy.getSize() / 8, 64);
    size = arith::DivSIOp::create(builder, loc, size, words);
    auto ate = arith::ConstantIntOp::create(builder, loc, 8, 64);
    Value count = arith::DivSIOp::create(builder, loc, size, ate);
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
        cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, vecEleTy);
    Value count = arith::DivSIOp::create(builder, loc, size, hostStrSize);
    Type packedTy = cudaq::opt::factory::genArgumentBufferType(eleTy);
    auto packSize = cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, packedTy);
    size = arith::MulIOp::create(builder, loc, count, packSize);
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
// of 40 bytes on libstdc++ (Linux) or 24 bytes on libc++ (macOS). The latter
// is identical to `std::vector<char>` (which has a size of 24 bytes).
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

// This code is used to unpack a std::vector<bool> (which is a specialized
// custom data structure and not like other std::vector<T>) from the host side
// into a std::vector<char>. This conversion allows us to use all the same code
// as every other std::vector<T> to marshal the argument.
static std::pair<Value, bool>
convertAllStdVectorBool(Location loc, OpBuilder &builder, ModuleOp module,
                        Value arg, Type ty, Value heapTracker,
                        std::optional<Value> preallocated = {}) {
  // If we are here, `ty` must be a `std::vector<bool>` or recursively contain a
  // `std::vector<bool>`.

  // Handle `std::vector<bool>`.
  if (isStdVectorBool(ty)) {
    auto stdvecTy = cast<cudaq::cc::StdvecType>(ty);
    Type stdvecHostTy =
        cudaq::opt::factory::stlVectorType(stdvecTy.getElementType());
    Value tmp = preallocated.has_value()
                    ? *preallocated
                    : cudaq::cc::AllocaOp::create(builder, loc, stdvecHostTy);
    func::CallOp::create(builder, loc, TypeRange{},
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
    auto inputRef = cudaq::cc::ComputePtrOp::create(builder, 
        loc, subVecPtrTy, arg, ArrayRef<cudaq::cc::ComputePtrArg>{0});
    auto startInput = cudaq::cc::LoadOp::create(builder, loc, inputRef);
    auto startTy = startInput.getType();
    auto subArrTy = cudaq::cc::ArrayType::get(
        cast<cudaq::cc::PointerType>(startTy).getElementType());
    auto input = cudaq::cc::CastOp::create(builder, 
        loc, cudaq::cc::PointerType::get(subArrTy), startInput);
    auto transientTy = convertToTransientType(sty, module);
    auto tmp = [&]() -> Value {
      if (preallocated)
        return cudaq::cc::CastOp::create(builder, 
            loc, cudaq::cc::PointerType::get(transientTy), *preallocated);
      return cudaq::cc::AllocaOp::create(builder, loc, transientTy);
    }();
    Value sizeDelta = genVectorSize</*FromQPU=*/false>(loc, builder, arg);
    auto count = [&]() -> Value {
      if (cudaq::cc::isDynamicType(seleTy)) {
        auto p = genByteSizeAndElementCount(loc, builder, module, seleTy,
                                            sizeDelta, arg, sty);
        return p.second;
      }
      auto sizeEle = cudaq::cc::SizeOfOp::create(builder, 
          loc, builder.getI64Type(), seleTy);
      return arith::DivSIOp::create(builder, loc, sizeDelta, sizeEle);
    }();
    auto transEleTy = cast<cudaq::cc::StructType>(transientTy).getMember(0);
    auto dataTy = cast<cudaq::cc::PointerType>(transEleTy).getElementType();
    auto sizeTransientTy =
        cudaq::cc::SizeOfOp::create(builder, loc, builder.getI64Type(), dataTy);
    Value sizeInBytes =
        arith::MulIOp::create(builder, loc, count, sizeTransientTy);

    // Create a new vector that we'll store the converted data into.
    Value byteBuffer = cudaq::cc::AllocaOp::create(builder, 
        loc, builder.getI8Type(), sizeInBytes);

    // Initialize the temporary vector.
    auto vecEleTy = cudaq::cc::PointerType::get(transEleTy);
    auto tmpBegin = cudaq::cc::ComputePtrOp::create(builder, 
        loc, vecEleTy, tmp, ArrayRef<cudaq::cc::ComputePtrArg>{0});
    auto bufferBegin =
        cudaq::cc::CastOp::create(builder, loc, transEleTy, byteBuffer);
    cudaq::cc::StoreOp::create(builder, loc, bufferBegin, tmpBegin);
    auto tmpEnd = cudaq::cc::ComputePtrOp::create(builder, 
        loc, vecEleTy, tmp, ArrayRef<cudaq::cc::ComputePtrArg>{1});
    auto byteBufferEnd = cudaq::cc::ComputePtrOp::create(builder, 
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), byteBuffer,
        ArrayRef<cudaq::cc::ComputePtrArg>{sizeInBytes});
    auto bufferEnd =
        cudaq::cc::CastOp::create(builder, loc, transEleTy, byteBufferEnd);
    cudaq::cc::StoreOp::create(builder, loc, bufferEnd, tmpEnd);
    auto tmpEnd2 = cudaq::cc::ComputePtrOp::create(builder, 
        loc, vecEleTy, tmp, ArrayRef<cudaq::cc::ComputePtrArg>{2});
    cudaq::cc::StoreOp::create(builder, loc, bufferEnd, tmpEnd2);

    // Loop over each element in the outer vector and initialize it to the inner
    // vector value. The data may be heap allocated.)
    auto transientEleTy = convertToTransientType(seleTy, module);
    auto transientBufferTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(transientEleTy));
    auto buffer =
        cudaq::cc::CastOp::create(builder, loc, transientBufferTy, byteBuffer);

    cudaq::opt::factory::createInvariantLoop(
        builder, loc, count,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value i = block.getArgument(0);
          Value inp = cudaq::cc::ComputePtrOp::create(builder, 
              loc, startTy, input, ArrayRef<cudaq::cc::ComputePtrArg>{i});
          auto currentVector = cudaq::cc::ComputePtrOp::create(builder, 
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
        return cudaq::cc::CastOp::create(builder, 
            loc, cudaq::cc::PointerType::get(bufferTy), *preallocated);
      return cudaq::cc::AllocaOp::create(builder, loc, bufferTy);
    }();

    // Loop over each element. Replace each with the converted value.
    for (auto iter : llvm::enumerate(sty.getMembers())) {
      std::int32_t i = iter.index();
      Type memTy = iter.value();
      auto fromPtr = cudaq::cc::ComputePtrOp::create(builder, 
          loc, cudaq::cc::PointerType::get(argStrTy.getMember(i)), arg,
          ArrayRef<cudaq::cc::ComputePtrArg>{i});
      auto transientTy = convertToTransientType(memTy, module);
      Value toPtr = cudaq::cc::ComputePtrOp::create(builder, 
          loc, cudaq::cc::PointerType::get(transientTy), buffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{i});
      convertAllStdVectorBool(loc, builder, module, fromPtr, memTy, heapTracker,
                              toPtr);
    }
    return {buffer, true};
  }
  return {arg, false};
}

std::pair<Value, bool>
cudaq::opt::marshal::unpackAnyStdVectorBool(Location loc, OpBuilder &builder,
                                            ModuleOp module, Value arg, Type ty,
                                            Value heapTracker) {
  if (hasStdVectorBool(ty))
    return convertAllStdVectorBool(loc, builder, module, arg, ty, heapTracker);
  return {arg, false};
}

// Return the size of the dynamic type, \p ty. Use recursive descent to compute
// the total addendum size needed for the argument, \p arg.
template <bool FromQPU>
Value descendThroughDynamicType(Location loc, OpBuilder &builder,
                                ModuleOp module, Type ty, Value addend,
                                Value arg, Value tmp) {
  auto i64Ty = builder.getI64Type();
  Value tySize =
      TypeSwitch<Type, Value>(ty)
          // A char span is dynamic, but it is not recursively dynamic. Just
          // read the length of the string out.
          .Case([&](cudaq::cc::CharspanType t) -> Value {
            return genStringLength<FromQPU>(loc, builder, arg, module);
          })
          // A std::vector is dynamic and may be recursive dynamic as well.
          .Case([&](cudaq::cc::StdvecType t) -> Value {
            // Compute the byte span of the vector.
            Value size = genVectorSize<FromQPU>(loc, builder, arg);
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
            cudaq::cc::StoreOp::create(builder, loc, size, tmp);
            auto ptrTy = cast<cudaq::cc::PointerType>(arg.getType());
            auto strTy = cast<cudaq::cc::StructType>(ptrTy.getElementType());
            auto memTy = cast<cudaq::cc::PointerType>(strTy.getMember(0));
            auto arrTy =
                cudaq::cc::PointerType::get(cudaq::cc::PointerType::get(
                    cudaq::cc::ArrayType::get(memTy.getElementType())));
            auto castPtr = cudaq::cc::CastOp::create(builder, loc, arrTy, arg);
            auto castArg = cudaq::cc::LoadOp::create(builder, loc, castPtr);
            auto castPtrTy =
                cudaq::cc::PointerType::get(memTy.getElementType());
            cudaq::opt::factory::createInvariantLoop(
                builder, loc, count,
                [&](OpBuilder &builder, Location loc, Region &, Block &block) {
                  Value i = block.getArgument(0);
                  auto ai = cudaq::cc::ComputePtrOp::create(builder, 
                      loc, castPtrTy, castArg,
                      ArrayRef<cudaq::cc::ComputePtrArg>{i});
                  auto tmpVal = cudaq::cc::LoadOp::create(builder, loc, tmp);
                  Value innerSize = descendThroughDynamicType<FromQPU>(
                      loc, builder, module, eleTy, tmpVal, ai, tmp);
                  cudaq::cc::StoreOp::create(builder, loc, innerSize, tmp);
                });
            return cudaq::cc::LoadOp::create(builder, loc, tmp);
          })
          // A struct can be dynamic if it contains dynamic members. Get the
          // static portion of the struct first, which will have length slots.
          // Then get the dynamic sizes for the dynamic members.
          .Case([&](cudaq::cc::StructType t) -> Value {
            if (cudaq::cc::isDynamicType(t)) {
              Type packedTy = cudaq::opt::factory::genArgumentBufferType(t);
              Value strSize =
                  cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, packedTy);
              for (auto iter : llvm::enumerate(t.getMembers())) {
                std::int32_t i = iter.index();
                auto m = iter.value();
                if (cudaq::cc::isDynamicType(m)) {
                  auto hostPtrTy = cast<cudaq::cc::PointerType>(arg.getType());
                  auto hostStrTy =
                      cast<cudaq::cc::StructType>(hostPtrTy.getElementType());
                  auto pm = cudaq::cc::PointerType::get(hostStrTy.getMember(i));
                  auto ai = cudaq::cc::ComputePtrOp::create(builder, 
                      loc, pm, arg, ArrayRef<cudaq::cc::ComputePtrArg>{i});
                  strSize = descendThroughDynamicType<FromQPU>(
                      loc, builder, module, m, strSize, ai, tmp);
                }
              }
              return strSize;
            }
            return cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, t);
          })
          .Default([&](Type t) -> Value {
            return cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, t);
          });
  return arith::AddIOp::create(builder, loc, tySize, addend);
}

template <bool FromQPU>
Value genSizeOfDynamicMessageBufferImpl(
    Location loc, OpBuilder &builder, ModuleOp module,
    cudaq::cc::StructType structTy,
    ArrayRef<std::tuple<unsigned, Value, Type>> zippy, Value tmp) {
  auto i64Ty = builder.getI64Type();
  Value initSize = cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, structTy);
  for (auto [_, a, t] : zippy)
    if (cudaq::cc::isDynamicType(t))
      initSize = descendThroughDynamicType<FromQPU>(loc, builder, module, t,
                                                    initSize, a, tmp);
  return initSize;
}

Value cudaq::opt::marshal::genSizeOfDynamicMessageBuffer(
    Location loc, OpBuilder &builder, ModuleOp module,
    cudaq::cc::StructType structTy,
    ArrayRef<std::tuple<unsigned, Value, Type>> zippy, Value tmp) {
  return genSizeOfDynamicMessageBufferImpl</*FromQPU=*/false>(
      loc, builder, module, structTy, zippy, tmp);
}

Value cudaq::opt::marshal::genSizeOfDynamicCallbackBuffer(
    Location loc, OpBuilder &builder, ModuleOp module,
    cudaq::cc::StructType structTy,
    ArrayRef<std::tuple<unsigned, Value, Type>> zippy, Value tmp) {
  return genSizeOfDynamicMessageBufferImpl</*FromQPU=*/true>(
      loc, builder, module, structTy, zippy, tmp);
}

template <bool FromQPU>
Value populateStringAddendum(Location loc, OpBuilder &builder, Value host,
                             Value sizeSlot, Value addendum, ModuleOp module) {
  Value size = genStringLength<FromQPU>(loc, builder, host, module);
  cudaq::cc::StoreOp::create(builder, loc, size, sizeSlot);
  auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
  Value dataPtr;
  if constexpr (FromQPU) {
    dataPtr = cudaq::cc::StdvecDataOp::create(builder, loc, ptrI8Ty, host);
  } else /*constexpr*/ {
    auto fromPtr = cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, host);
    StringRef helperName = module->getAttr(cudaq::runtime::sizeofStringAttrName)
                               ? cudaq::runtime::getPauliWordData
                               : cudaq::runtime::bindingGetStringData;
    auto call = func::CallOp::create(builder, loc, ptrI8Ty, helperName,
                                             ValueRange{fromPtr});
    dataPtr = call.getResult(0);
  }
  auto notVolatile = arith::ConstantIntOp::create(builder, loc, 0, 1);
  auto toPtr = cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, addendum);
  func::CallOp::create(builder, loc, TypeRange{}, cudaq::llvmMemCopyIntrinsic,
                               ValueRange{toPtr, dataPtr, size, notVolatile});
  auto ptrI8Arr = getByteAddressableType(builder);
  auto addBytes = cudaq::cc::CastOp::create(builder, loc, ptrI8Arr, addendum);
  return cudaq::cc::ComputePtrOp::create(builder, 
      loc, ptrI8Ty, addBytes, ArrayRef<cudaq::cc::ComputePtrArg>{size});
}

// Simple case when the vector data is known to not hold dynamic data.
template <bool FromQPU>
Value populateVectorAddendum(Location loc, OpBuilder &builder, Value host,
                             Value sizeSlot, Value addendum) {
  Value size = genVectorSize<FromQPU>(loc, builder, host);
  cudaq::cc::StoreOp::create(builder, loc, size, sizeSlot);
  auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
  auto ptrPtrI8 = cudaq::opt::marshal::getPointerToPointerType(builder);
  Value dataPtr = [&]() -> Value {
    if constexpr (FromQPU) {
      auto eleTy = cast<cudaq::cc::StdvecType>(host.getType()).getElementType();
      auto ptrTy = cudaq::cc::PointerType::get(eleTy);
      auto vecDataPtr =
          cudaq::cc::StdvecDataOp::create(builder, loc, ptrTy, host);
      return cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, vecDataPtr);
    } else /*constexpr*/ {
      auto fromPtrPtr = cudaq::cc::CastOp::create(builder, loc, ptrPtrI8, host);
      return cudaq::cc::LoadOp::create(builder, loc, fromPtrPtr);
    }
  }();
  auto notVolatile = arith::ConstantIntOp::create(builder, loc, 0, 1);
  auto toPtr = cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, addendum);
  func::CallOp::create(builder, loc, TypeRange{}, cudaq::llvmMemCopyIntrinsic,
                               ValueRange{toPtr, dataPtr, size, notVolatile});
  auto ptrI8Arr = getByteAddressableType(builder);
  auto addBytes = cudaq::cc::CastOp::create(builder, loc, ptrI8Arr, addendum);
  return cudaq::cc::ComputePtrOp::create(builder, 
      loc, ptrI8Ty, addBytes, ArrayRef<cudaq::cc::ComputePtrArg>{size});
}

template <bool FromQPU>
Value populateDynamicAddendum(Location loc, OpBuilder &builder, ModuleOp module,
                              Type devArgTy, Value host, Value sizeSlot,
                              Value addendum, Value addendumScratch) {
  if (isa<cudaq::cc::CharspanType>(devArgTy))
    return populateStringAddendum<FromQPU>(loc, builder, host, sizeSlot,
                                           addendum, module);
  if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(devArgTy)) {
    auto eleTy = vecTy.getElementType();
    if (cudaq::cc::isDynamicType(eleTy)) {
      // Recursive case. Visit each dynamic element, copying it.
      Value size = genVectorSize<FromQPU>(loc, builder, host);
      auto [bytes, count] = genByteSizeAndElementCount(
          loc, builder, module, eleTy, size, host, devArgTy);
      size = bytes;
      cudaq::cc::StoreOp::create(builder, loc, size, sizeSlot);

      // Convert from bytes to vector length in elements.
      // Compute new addendum start.
      auto addrTy = getByteAddressableType(builder);
      auto castEnd = cudaq::cc::CastOp::create(builder, loc, addrTy, addendum);
      Value newAddendum = cudaq::cc::ComputePtrOp::create(builder, 
          loc, addendum.getType(), castEnd,
          ArrayRef<cudaq::cc::ComputePtrArg>{size});
      cudaq::cc::StoreOp::create(builder, loc, newAddendum, addendumScratch);
      Type dataTy = cudaq::opt::factory::genArgumentBufferType(eleTy);
      auto arrDataTy = cudaq::cc::ArrayType::get(dataTy);
      auto sizeBlockTy = cudaq::cc::PointerType::get(arrDataTy);
      auto ptrDataTy = cudaq::cc::PointerType::get(dataTy);

      // In the recursive case, the next block of addendum is a vector of
      // elements which are either sizes or contain sizes. The sizes are i64
      // and expressed in bytes. Each size will be the size of the span of the
      // element (or its subfields) at that offset.
      auto sizeBlock =
          cudaq::cc::CastOp::create(builder, loc, sizeBlockTy, addendum);
      auto hostEleTy =
          cast<cudaq::cc::PointerType>(host.getType()).getElementType();
      auto ptrPtrBlockTy = cudaq::cc::PointerType::get(
          cast<cudaq::cc::StructType>(hostEleTy).getMember(0));

      // The host argument is a std::vector, so we want to get the address of
      // "front" out of the vector (the first pointer in the triple) and step
      // over the contiguous range of vectors in the host block. The vector of
      // vectors forms a ragged array structure in host memory.
      auto hostBeginPtrRef = cudaq::cc::ComputePtrOp::create(builder, 
          loc, ptrPtrBlockTy, host, ArrayRef<cudaq::cc::ComputePtrArg>{0});
      auto hostBegin = cudaq::cc::LoadOp::create(builder, loc, hostBeginPtrRef);
      auto hostBeginEleTy = cast<cudaq::cc::PointerType>(hostBegin.getType());
      auto hostBlockTy = cudaq::cc::PointerType::get(
          cudaq::cc::ArrayType::get(hostBeginEleTy.getElementType()));
      auto hostBlock =
          cudaq::cc::CastOp::create(builder, loc, hostBlockTy, hostBegin);

      // Loop over each vector element in the vector (recursively).
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, count,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = block.getArgument(0);
            Value addm =
                cudaq::cc::LoadOp::create(builder, loc, addendumScratch);
            auto subSlot = cudaq::cc::ComputePtrOp::create(builder, 
                loc, ptrDataTy, sizeBlock,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            auto subHost = cudaq::cc::ComputePtrOp::create(builder, 
                loc, hostBeginEleTy, hostBlock,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            Value newAddm = populateDynamicAddendum<FromQPU>(
                loc, builder, module, eleTy, subHost, subSlot, addm,
                addendumScratch);
            cudaq::cc::StoreOp::create(builder, loc, newAddm, addendumScratch);
          });
      return cudaq::cc::LoadOp::create(builder, loc, addendumScratch);
    }
    return populateVectorAddendum<FromQPU>(loc, builder, host, sizeSlot,
                                           addendum);
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
    auto val = cudaq::cc::ComputePtrOp::create(builder, 
        loc, cudaq::cc::PointerType::get(hostMemTy), host,
        ArrayRef<cudaq::cc::ComputePtrArg>{iterIdx});
    Type iterTy = iter.value();
    if (cudaq::cc::isDynamicType(iterTy)) {
      Value fieldInSlot = cudaq::cc::ComputePtrOp::create(builder, 
          loc, cudaq::cc::PointerType::get(builder.getI64Type()), sizeSlot,
          ArrayRef<cudaq::cc::ComputePtrArg>{iterIdx});
      addendum = populateDynamicAddendum<FromQPU>(loc, builder, module, iterTy,
                                                  val, fieldInSlot, addendum,
                                                  addendumScratch);
    } else {
      Value fieldInSlot = cudaq::cc::ComputePtrOp::create(builder, 
          loc, cudaq::cc::PointerType::get(iterTy), sizeSlot,
          ArrayRef<cudaq::cc::ComputePtrArg>{iterIdx});
      auto v = cudaq::cc::LoadOp::create(builder, loc, val);
      cudaq::cc::StoreOp::create(builder, loc, v, fieldInSlot);
    }
  }
  return addendum;
}

template <bool FromQPU>
void populateMessageBufferImpl(
    Location loc, OpBuilder &builder, ModuleOp module, Value msgBufferBase,
    ArrayRef<std::tuple<unsigned, Value, Type>> zippy, Value addendum,
    Value addendumScratch) {
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
      auto slot = cudaq::cc::ComputePtrOp::create(builder, 
          loc, ptrTy, msgBufferBase, ArrayRef<cudaq::cc::ComputePtrArg>{i});
      addendum = populateDynamicAddendum<FromQPU>(
          loc, builder, module, devArgTy, arg, slot, addendum, addendumScratch);
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
    Value slot = cudaq::cc::ComputePtrOp::create(builder, 
        loc, ptrTy, msgBufferBase, ArrayRef<cudaq::cc::ComputePtrArg>{i});

    // Argument is a packaged kernel. In this case, the argument is some
    // unknown kernel that may be called. The packaged argument is coming
    // from opaque C++ host code, so we need to identify what kernel it
    // references and then pass its name as a span of characters to the
    // launch kernel.
    if (isa<cudaq::cc::IndirectCallableType>(devArgTy)) {
      auto i64Ty = builder.getI64Type();
      auto kernKey = func::CallOp::create(builder, 
          loc, i64Ty, cudaq::runtime::getLinkableKernelKey, ValueRange{arg});
      cudaq::cc::StoreOp::create(builder, loc, kernKey.getResult(0), slot);
      continue;
    }

    // Just pass the raw pointer. The buffer is supposed to be pointer-free
    // since it may be unpacked in a different address space. However, if this
    // is a simulation and things are in the same address space, we pass the
    // pointer for convenience.
    if (isa<cudaq::cc::PointerType>(devArgTy))
      arg = cudaq::cc::CastOp::create(builder, loc, memberTy, arg);

    if (isa<cudaq::cc::StructType, cudaq::cc::ArrayType>(arg.getType()) &&
        (cudaq::cc::PointerType::get(arg.getType()) != slot.getType())) {
      slot = cudaq::cc::CastOp::create(builder, 
          loc, cudaq::cc::PointerType::get(arg.getType()), slot);
    }
    cudaq::cc::StoreOp::create(builder, loc, arg, slot);
  }
}

void cudaq::opt::marshal::populateMessageBuffer(
    Location loc, OpBuilder &builder, ModuleOp module, Value msgBufferBase,
    ArrayRef<std::tuple<unsigned, Value, Type>> zippy, Value addendum,
    Value addendumScratch) {
  populateMessageBufferImpl</*FromQPU=*/false>(
      loc, builder, module, msgBufferBase, zippy, addendum, addendumScratch);
}

void cudaq::opt::marshal::populateCallbackBuffer(
    Location loc, OpBuilder &builder, ModuleOp module, Value msgBufferBase,
    ArrayRef<std::tuple<unsigned, Value, Type>> zippy, Value addendum,
    Value addendumScratch) {
  populateMessageBufferImpl</*FromQPU=*/true>(
      loc, builder, module, msgBufferBase, zippy, addendum, addendumScratch);
}

bool cudaq::opt::marshal::hasLegalType(FunctionType funTy) {
  for (auto ty : funTy.getInputs())
    if (quake::isQuantumType(ty))
      return false;
  for (auto ty : funTy.getResults())
    if (quake::isQuantumType(ty))
      return false;
  return true;
}

MutableArrayRef<BlockArgument>
cudaq::opt::marshal::dropAnyHiddenArguments(MutableArrayRef<BlockArgument> args,
                                            FunctionType funcTy,
                                            bool hasThisPointer) {
  const bool hiddenSRet = opt::factory::hasHiddenSRet(funcTy);
  const unsigned count = cc::numberOfHiddenArgs(hasThisPointer, hiddenSRet);
  if (count > 0 && args.size() >= count &&
      std::all_of(args.begin(), args.begin() + count,
                  [](auto i) { return isa<cc::PointerType>(i.getType()); }))
    return args.drop_front(count);
  return args;
}

std::pair<bool, func::FuncOp> cudaq::opt::marshal::lookupHostEntryPointFunc(
    StringRef mangledEntryPointName, ModuleOp module, func::FuncOp funcOp) {
  if (mangledEntryPointName == "BuilderKernel.EntryPoint" ||
      mangledEntryPointName.contains("_PyKernelEntryPointRewrite") ||
      funcOp.empty()) {
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

void cudaq::opt::marshal::genStdvecBoolFromInitList(Location loc,
                                                    OpBuilder &builder,
                                                    Value sret, Value data,
                                                    Value size) {
  auto ptrTy = cc::PointerType::get(builder.getContext());
  auto castData = cc::CastOp::create(builder, loc, ptrTy, data);
  auto castSret = cc::CastOp::create(builder, loc, ptrTy, sret);
  func::CallOp::create(builder, loc, TypeRange{}, stdvecBoolCtorFromInitList,
                               ArrayRef<Value>{castSret, castData, size});
}

void cudaq::opt::marshal::genStdvecTFromInitList(Location loc,
                                                 OpBuilder &builder, Value sret,
                                                 Value data, Value tSize,
                                                 Value vecSize) {
  auto i8Ty = builder.getI8Type();
  auto stlVectorTy = cc::PointerType::get(opt::factory::stlVectorType(i8Ty));
  auto ptrTy = cc::PointerType::get(i8Ty);
  auto castSret = cc::CastOp::create(builder, loc, stlVectorTy, sret);
  auto ptrPtrTy = cc::PointerType::get(ptrTy);
  auto sret0 = cc::ComputePtrOp::create(builder, 
      loc, ptrPtrTy, castSret, SmallVector<cc::ComputePtrArg>{0});
  auto arrI8Ty = cc::ArrayType::get(i8Ty);
  auto ptrArrTy = cc::PointerType::get(arrI8Ty);
  auto buffPtr0 = cc::CastOp::create(builder, loc, ptrTy, data);
  cc::StoreOp::create(builder, loc, buffPtr0, sret0);
  auto sret1 = cc::ComputePtrOp::create(builder, 
      loc, ptrPtrTy, castSret, SmallVector<cc::ComputePtrArg>{1});
  Value byteLen = arith::MulIOp::create(builder, loc, tSize, vecSize);
  auto buffPtr = cc::CastOp::create(builder, loc, ptrArrTy, data);
  auto endPtr = cc::ComputePtrOp::create(builder, 
      loc, ptrTy, buffPtr, SmallVector<cc::ComputePtrArg>{byteLen});
  cc::StoreOp::create(builder, loc, endPtr, sret1);
  auto sret2 = cc::ComputePtrOp::create(builder, 
      loc, ptrPtrTy, castSret, SmallVector<cc::ComputePtrArg>{2});
  cc::StoreOp::create(builder, loc, endPtr, sret2);
}

Value cudaq::opt::marshal::createEmptyHeapTracker(Location loc,
                                                  OpBuilder &builder) {
  auto ptrI8Ty = cc::PointerType::get(builder.getI8Type());
  auto result = cc::AllocaOp::create(builder, loc, ptrI8Ty);
  auto zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
  auto null = cc::CastOp::create(builder, loc, ptrI8Ty, zero);
  cc::StoreOp::create(builder, loc, null, result);
  return result;
}

void cudaq::opt::marshal::maybeFreeHeapAllocations(Location loc,
                                                   OpBuilder &builder,
                                                   Value heapTracker) {
  auto head = cc::LoadOp::create(builder, loc, heapTracker);
  auto zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
  auto headAsInt = cc::CastOp::create(builder, loc, builder.getI64Type(), head);
  auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                           headAsInt, zero);
  // If there are no std::vector<bool> to unpack, then the heapTracker will be
  // set to `nullptr` and otherwise unused. That will allow the compiler to DCE
  // this call after constant propagation.
  cc::IfOp::create(builder, 
      loc, TypeRange{}, cmp,
      [&](OpBuilder &builder, Location loc, Region &region) {
        region.push_back(new Block());
        auto &body = region.front();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body);
        func::CallOp::create(builder, loc, TypeRange{},
                                     stdvecBoolFreeTemporaryLists,
                                     ArrayRef<Value>{head});
        cc::ContinueOp::create(builder, loc);
      });
}

/// Fetch an argument from the comm buffer. Here, the argument is not dynamic so
/// it can be read as is out of the buffer.
template <bool FromQPU>
Value fetchInputValue(Location loc, OpBuilder &builder, Type devTy, Value ptr) {
  assert(!cudaq::cc::isDynamicType(devTy) && "must not be a dynamic type");
  if (isa<cudaq::cc::IndirectCallableType>(devTy)) {
    // An indirect callable passes a key value which will be used to determine
    // the kernel that is being called.
    auto key = cudaq::cc::LoadOp::create(builder, loc, ptr);
    return cudaq::cc::CastOp::create(builder, loc, devTy, key);
  }

  if (isa<cudaq::cc::CallableType>(devTy)) {
    // A direct callable will have already been effectively inlined and this
    // argument should not be referenced.
    return cudaq::cc::PoisonOp::create(builder, loc, devTy);
  }

  auto ptrDevTy = cudaq::cc::PointerType::get(devTy);
  if (auto strTy = dyn_cast<cudaq::cc::StructType>(devTy)) {
    // Argument is a struct.
    if (strTy.isEmpty())
      return cudaq::cc::UndefOp::create(builder, loc, devTy);

    // Cast to avoid conflicts between layout compatible, distinct struct types.
    auto structPtr = cudaq::cc::CastOp::create(builder, loc, ptrDevTy, ptr);
    if constexpr (FromQPU) {
      return structPtr;
    } else {
      return cudaq::cc::LoadOp::create(builder, loc, structPtr);
    }
  }

  // Default case: argument passed as a value inplace.
  return cudaq::cc::LoadOp::create(builder, loc, ptr);
}

/// Helper routine to generate code to increment the trailing data pointer to
/// the next block of data (if any).
static Value incrementTrailingDataPointer(Location loc, OpBuilder &builder,
                                          Value trailingData, Value bytes) {
  auto i8Ty = builder.getI8Type();
  auto bufferTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty));
  auto buffPtr = cudaq::cc::CastOp::create(builder, loc, bufferTy, trailingData);
  auto i8PtrTy = cudaq::cc::PointerType::get(i8Ty);
  return cudaq::cc::ComputePtrOp::create(builder, 
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
///    {{{'a'}, {'b', 'c'}, {'z'}}, {{'d', 'e', 'f'}}};
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
template <bool FromQPU>
std::pair<Value, Value>
constructDynamicInputValue(Location loc, OpBuilder &builder, Type devTy,
                           Value ptr, Value trailingData) {
  assert(cudaq::cc::isDynamicType(devTy) && "must be dynamic type");
  if constexpr (FromQPU) {
    if (auto charSpanTy = dyn_cast<cudaq::cc::CharspanType>(devTy)) {
      // From host, so construct the stdvec span with it.
      auto eleTy = charSpanTy.getElementType();
      auto castTrailingData = cudaq::cc::CastOp::create(builder, 
          loc, cudaq::cc::PointerType::get(eleTy), trailingData);
      Value vecLength = cudaq::cc::LoadOp::create(builder, loc, ptr);
      auto result = cudaq::cc::StdvecInitOp::create(builder, 
          loc, charSpanTy, castTrailingData, vecLength);
      auto nextTrailingData =
          incrementTrailingDataPointer(loc, builder, trailingData, vecLength);
      return {result, nextTrailingData};
    }
  }
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
    auto eleSize = cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, buffEleTy);
    Value bytes = cudaq::cc::LoadOp::create(builder, loc, ptr);
    auto vecLength = arith::DivSIOp::create(builder, loc, bytes, eleSize);

    if (cudaq::cc::isDynamicType(eleTy)) {
      // The vector is recursively dynamic.
      // Create a new block in which to place the stdvec/struct data. The
      // trailing data is in device-side format (pointer-free spans).
      Type toTy = [&]() {
        // From QPU, so we want to unpack the data into vectors (of vectors).
        if constexpr (FromQPU) {
          auto module = vecLength->getParentOfType<ModuleOp>();
          return cudaq::opt::factory::convertToHostSideType(eleTy, module);
        } else {
          // From host, so we want to unpack the data into spans (of spans).
          return eleTy;
        }
      }();
      Value newVecData =
          cudaq::cc::AllocaOp::create(builder, loc, toTy, vecLength);
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
          cudaq::cc::CastOp::create(builder, loc, packedArrTy, trailingData);
      auto trailingDataVar =
          cudaq::cc::AllocaOp::create(builder, loc, nextTrailingData.getType());
      cudaq::cc::StoreOp::create(builder, loc, nextTrailingData,
                                         trailingDataVar);
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, vecLength,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = block.getArgument(0);
            auto nextTrailingData =
                cudaq::cc::LoadOp::create(builder, loc, trailingDataVar);
            auto vecMemPtr = cudaq::cc::ComputePtrOp::create(builder, 
                loc, packedEleTy, arrPtr,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            auto r = constructDynamicInputValue<FromQPU>(
                loc, builder, eleTy, vecMemPtr, nextTrailingData);
            auto newVecPtr = cudaq::cc::ComputePtrOp::create(builder, 
                loc, elePtrTy, newVecData,
                ArrayRef<cudaq::cc::ComputePtrArg>{i});
            cudaq::cc::StoreOp::create(builder, loc, r.first, newVecPtr);
            cudaq::cc::StoreOp::create(builder, loc, r.second, trailingDataVar);
          });

      // Create the new outer stdvec span as the result.
      Value stdvecResult = cudaq::cc::StdvecInitOp::create(builder, 
          loc, spanTy, newVecData, vecLength);
      nextTrailingData =
          cudaq::cc::LoadOp::create(builder, loc, trailingDataVar);
      return {stdvecResult, nextTrailingData};
    }

    // This vector has constant data, so just use the data in-place.
    Value result;
    if constexpr (FromQPU) {
      // From QPU, so construct a std::vector from the span.
      auto ptrTy = cudaq::cc::PointerType::get(eleTy);
      auto *ctx = builder.getContext();
      auto vecTy =
          cudaq::cc::StructType::get(ctx, ArrayRef<Type>{ptrTy, ptrTy, ptrTy});
      Value vecVar = cudaq::cc::UndefOp::create(builder, loc, vecTy);
      Value castData =
          cudaq::cc::CastOp::create(builder, loc, ptrTy, trailingData);
      vecVar = cudaq::cc::InsertValueOp::create(builder, loc, vecTy, vecVar,
                                                        castData, 0);
      auto ptrArrTy =
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(eleTy));
      auto castTrailingData =
          cudaq::cc::CastOp::create(builder, loc, ptrArrTy, trailingData);
      Value castEnd = cudaq::cc::ComputePtrOp::create(builder, 
          loc, ptrTy, castTrailingData,
          ArrayRef<cudaq::cc::ComputePtrArg>{bytes});
      vecVar = cudaq::cc::InsertValueOp::create(builder, loc, vecTy, vecVar,
                                                        castEnd, 1);
      result = cudaq::cc::InsertValueOp::create(builder, loc, vecTy, vecVar,
                                                        castEnd, 2);
    } else /*constexpr*/ {
      // From host, so construct the stdvec span with it.
      auto castTrailingData = cudaq::cc::CastOp::create(builder, 
          loc, cudaq::cc::PointerType::get(eleTy), trailingData);
      result = cudaq::cc::StdvecInitOp::create(builder, 
          loc, spanTy, castTrailingData, vecLength);
    }
    auto nextTrailingData =
        incrementTrailingDataPointer(loc, builder, trailingData, bytes);
    return {result, nextTrailingData};
  }

  // Argument must be a struct.
  // The struct contains dynamic components. Extract them and build up the
  // struct value to be passed as an argument.
  // ptr: pointer to the first element of the struct or a vector length.
  // tailingData: the block of data for the first dynamic type field.
  auto strTy = cast<cudaq::cc::StructType>(devTy);
  auto ptrEleTy = cast<cudaq::cc::PointerType>(ptr.getType()).getElementType();
  auto packedTy = cast<cudaq::cc::StructType>(ptrEleTy);
  Value result = cudaq::cc::UndefOp::create(builder, loc, strTy);
  assert(strTy.getNumMembers() == packedTy.getNumMembers());
  for (auto iter :
       llvm::enumerate(llvm::zip(strTy.getMembers(), packedTy.getMembers()))) {
    auto devMemTy = std::get<0>(iter.value());
    std::int32_t off = iter.index();
    auto packedMemTy = std::get<1>(iter.value());
    auto dataPtr = cudaq::cc::ComputePtrOp::create(builder, 
        loc, cudaq::cc::PointerType::get(packedMemTy), ptr,
        ArrayRef<cudaq::cc::ComputePtrArg>{off});
    if (cudaq::cc::isDynamicType(devMemTy)) {
      auto r = constructDynamicInputValue<FromQPU>(loc, builder, devMemTy,
                                                   dataPtr, trailingData);
      result = cudaq::cc::InsertValueOp::create(builder, loc, strTy, result,
                                                        r.first, off);
      trailingData = r.second;
      continue;
    }
    auto val = fetchInputValue<FromQPU>(loc, builder, devMemTy, dataPtr);
    result =
        cudaq::cc::InsertValueOp::create(builder, loc, strTy, result, val, off);
  }
  return {result, trailingData};
}

template <bool FromQPU>
std::pair<Value, Value>
processInputValueImpl(Location loc, OpBuilder &builder, Value trailingData,
                      Value ptrPackedStruct, Type inTy, std::int32_t off,
                      cudaq::cc::StructType packedStructTy) {
  auto packedPtr = cudaq::cc::ComputePtrOp::create(builder, 
      loc, cudaq::cc::PointerType::get(packedStructTy.getMember(off)),
      ptrPackedStruct, ArrayRef<cudaq::cc::ComputePtrArg>{off});
  if (cudaq::cc::isDynamicType(inTy)) {
    if constexpr (FromQPU) {
      auto dynamo = constructDynamicInputValue<FromQPU>(
          loc, builder, inTy, packedPtr, trailingData);
      if (isa<cudaq::cc::StdvecType>(inTy)) {
        Value retVal = dynamo.first;
        Value tmp = cudaq::cc::AllocaOp::create(builder, loc, retVal.getType());
        cudaq::cc::StoreOp::create(builder, loc, retVal, tmp);
        return {tmp, dynamo.second};
      }
      if (isa<cudaq::cc::CharspanType>(inTy)) {
        auto module = packedPtr->getParentOfType<ModuleOp>();
        auto arrTy = cudaq::opt::factory::genHostStringType(module);
        Value retVal = dynamo.first;
        Value tmp = cudaq::cc::AllocaOp::create(builder, loc, arrTy);
        auto ptrTy = cudaq::cc::PointerType::get(builder.getI8Type());
        Value castTmp = cudaq::cc::CastOp::create(builder, loc, ptrTy, tmp);
        Value len = cudaq::cc::StdvecSizeOp::create(builder, 
            loc, builder.getI64Type(), dynamo.first);
        Value data =
            cudaq::cc::StdvecDataOp::create(builder, loc, ptrTy, dynamo.first);
        func::CallOp::create(builder, loc, TypeRange{},
                                     cudaq::runtime::bindingInitializeString,
                                     ArrayRef<Value>{castTmp, data, len});
        return {tmp, dynamo.second};
      }
      return dynamo;
    } else /*constexpr*/ {
      return constructDynamicInputValue<FromQPU>(loc, builder, inTy, packedPtr,
                                                 trailingData);
    }
  }
  auto val = fetchInputValue<FromQPU>(loc, builder, inTy, packedPtr);
  return {val, trailingData};
}

std::pair<Value, Value> cudaq::opt::marshal::processInputValue(
    Location loc, OpBuilder &builder, Value trailingData, Value ptrPackedStruct,
    Type inTy, std::int32_t off, cc::StructType packedStructTy) {
  return processInputValueImpl</*FromQPU=*/false>(
      loc, builder, trailingData, ptrPackedStruct, inTy, off, packedStructTy);
}

std::pair<Value, Value> cudaq::opt::marshal::processCallbackInputValue(
    Location loc, OpBuilder &builder, Value trailingData, Value ptrPackedStruct,
    Type inTy, std::int32_t off, cc::StructType packedStructTy) {
  return processInputValueImpl</*FromQPU=*/true>(
      loc, builder, trailingData, ptrPackedStruct, inTy, off, packedStructTy);
}
