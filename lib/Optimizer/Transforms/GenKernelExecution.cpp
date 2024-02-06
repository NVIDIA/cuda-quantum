/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
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

#define DEBUG_TYPE "quake-kernel-exec"

using namespace mlir;

namespace {
// Define some constant function name strings.
static constexpr const char cudaqRegisterLambdaName[] =
    "cudaqRegisterLambdaName";
static constexpr const char cudaqRegisterArgsCreator[] =
    "cudaqRegisterArgsCreator";
static constexpr const char cudaqRegisterKernelName[] =
    "cudaqRegisterKernelName";

static constexpr std::size_t NoResultOffset = ~0u >> 1;

class GenerateKernelExecution
    : public cudaq::opt::impl::GenerateKernelExecutionBase<
          GenerateKernelExecution> {
public:
  using GenerateKernelExecutionBase::GenerateKernelExecutionBase;

  /// Build an LLVM struct type with all the arguments and then all the results.
  /// If the type is a std::vector, then add an i64 to the struct for the
  /// length. The actual data values will be appended to the end of the
  /// dynamically sized struct.
  ///
  /// A kernel signature of
  /// ```c++
  /// i32_t operator() (i16_t, std::vector<double>, double);
  /// ```
  /// will generate the llvm struct
  /// ```llvm
  /// { i16, i64, double, i32 }
  /// ```
  /// where the values of the vector argument are pass-by-value and appended to
  /// the end of the struct as a sequence of \i n double values.
  cudaq::cc::StructType buildStructType(const std::string &name,
                                        FunctionType funcTy) {
    auto *ctx = funcTy.getContext();
    SmallVector<Type> eleTys;
    auto i64Ty = IntegerType::get(ctx, 64);
    // Add all argument types, translating std::vector to a length or pointer
    // and length.
    auto pushType = [&](const bool isOutput, Type ty) {
      if (isa<cudaq::cc::CallableType>(ty)) {
        eleTys.push_back(cudaq::cc::PointerType::get(ctx));
      } else if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(ty)) {
        if (isOutput)
          eleTys.push_back(cudaq::cc::PointerType::get(vecTy.getElementType()));
        eleTys.push_back(i64Ty);
      } else if (auto strTy = dyn_cast<cudaq::cc::StructType>(ty);
                 strTy && strTy.getMembers().empty()) {
        eleTys.push_back(i64Ty);
      } else {
        eleTys.push_back(ty);
      }
    };

    for (auto inTy : funcTy.getInputs())
      pushType(false, inTy);
    for (auto outTy : funcTy.getResults())
      pushType(true, outTy);
    return cudaq::cc::StructType::get(ctx, eleTys);
  }

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
  /// <code>
  ///   struct vector {
  ///     T* begin;
  ///     T* end;
  ///     T* allocated_end;
  ///   };
  /// </code>
  /// The first two elements are pointers to the beginning and end of the data
  /// in the vector, respectively. This data is kept in a contiguous memory
  /// range. The following implementation follows what Clang CodeGen produces
  /// for `std::vector<T>::size()` without the final `sdiv` op that divides the
  /// `sizeof(data[N])` by the `sizeof(T)`. The result is the total required
  /// memory size for the vector data itself in \e bytes.
  Value getVectorSize(OpBuilder &builder, Location loc,
                      cudaq::cc::PointerType ptrTy, Value arg) {
    // Create the i64 type
    Type i64Ty = builder.getI64Type();

    // We're given ptr<struct<...>>, get that struct type (struct<T*,T*,T*>)
    auto inpStructTy = cast<cudaq::cc::StructType>(ptrTy.getElementType());

    // For the following GEP calls, we'll expect them to return T**
    auto ptrTtype = cudaq::cc::PointerType::get(inpStructTy.getMembers()[0]);

    // Get the pointer to the pointer of the end of the array
    Value endPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTtype, arg, SmallVector<cudaq::cc::ComputePtrArg>{0, 1});

    // Get the pointer to the pointer of the beginning of the array
    Value beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTtype, arg, SmallVector<cudaq::cc::ComputePtrArg>{0, 0});

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
  Value convertLengthBytesToLengthI64(OpBuilder &builder, Location loc,
                                      Value length) {
    auto eight = builder.create<arith::ConstantIntOp>(loc, 8, 64);
    return builder.create<arith::DivSIOp>(loc, length, eight);
  }

  /// This computes a vector's size and handles recursive vector types. This
  /// first value returned is the size of the top level (outermost) vector in
  /// bytes. The second value is the recursive size of all the vectors within
  /// the outer vector.
  std::pair<Value, Value>
  computeRecursiveVectorSize(OpBuilder &builder, Location loc, Value cpuArg,
                             cudaq::cc::PointerType cpuVecTy,
                             cudaq::cc::StdvecType stdvecTy) {
    Value topLevelSize;
    Value recursiveSize;
    auto eleTy = stdvecTy.getElementType();
    if (auto sTy = dyn_cast<cudaq::cc::StdvecType>(eleTy)) {
      // Convert size of vectors to i64s.
      topLevelSize = computeCpuVectorLengthInBytes(
          builder, loc, cpuArg, stdvecTy.getElementType(), cpuVecTy);
      auto nested = fetchCpuVectorFront(builder, loc, cpuArg, cpuVecTy);
      auto tmp = builder.create<cudaq::cc::AllocaOp>(loc, builder.getI64Type());
      builder.create<cudaq::cc::StoreOp>(loc, topLevelSize, tmp);
      // Convert bytes to units of i64. (Divide by 8)
      auto topLevelCount =
          convertLengthBytesToLengthI64(builder, loc, topLevelSize);
      // Now walk the vectors recursively.
      auto topLevelIndex = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), topLevelCount);
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, topLevelIndex,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = builder.create<arith::IndexCastOp>(
                loc, builder.getI64Type(), block.getArgument(0));
            auto sub = builder.create<cudaq::cc::ComputePtrOp>(loc, cpuVecTy,
                                                               nested, i);
            auto p =
                computeRecursiveVectorSize(builder, loc, sub, cpuVecTy, sTy);
            auto subSz = builder.create<cudaq::cc::LoadOp>(loc, tmp);
            auto sum = builder.create<arith::AddIOp>(loc, p.second, subSz);
            builder.create<cudaq::cc::StoreOp>(loc, sum, tmp);
          });
      recursiveSize = builder.create<cudaq::cc::LoadOp>(loc, tmp);
    } else {
      // Non-recusive case. Just compute the size of the top-level vector<T>.
      topLevelSize = getVectorSize(builder, loc, cpuVecTy, cpuArg);
      recursiveSize = topLevelSize;
    }
    return {topLevelSize, recursiveSize};
  }

  /// Copy a vector's data, which must be \p bytes in length, from \p cpuArg to
  /// \p outputBuffer. The cpuArg must have a pointer type that is compatible
  /// with the triple pointer std::vector base implementation.
  Value copyVectorData(OpBuilder &builder, Location loc, Value bytes,
                       Value cpuArg, Value outputBuffer) {
    auto notVolatile = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    auto inStructTy = cast<cudaq::cc::StructType>(
        cast<cudaq::cc::PointerType>(cpuArg.getType()).getElementType());
    auto beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, cudaq::cc::PointerType::get(inStructTy.getMembers()[0]), cpuArg,
        SmallVector<cudaq::cc::ComputePtrArg>{0, 0});
    auto fromBuff = builder.create<cudaq::cc::LoadOp>(loc, beginPtr);
    auto vecFromBuff = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), fromBuff);
    builder.create<func::CallOp>(
        loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
        SmallVector<Value>{outputBuffer, vecFromBuff, bytes, notVolatile});
    // Increment outputBuffer by size bytes.
    return builder.create<cudaq::cc::ComputePtrOp>(
        loc, outputBuffer.getType(), outputBuffer, SmallVector<Value>{bytes});
  }

  /// Creates a function that can take a block of pointers to argument values
  /// and using the compiler's knowledge of a kernel encodes those argument
  /// values into a message buffer. The message buffer is a pointer-free block
  /// of memory allocated on the heap on the host. Once the argument values are
  /// packed into the message buffer, they can be passed to altLaunchKernel or
  /// the corresponding thunk function.
  ///
  /// The created function takes two arguments: a pointer to the argument values
  /// to be encoded and a pointer to a pointer into which the message buffer
  /// value will be written for return. This function returns to size of the
  /// message buffer. (Message buffers are at least the size of \p structTy but
  /// may be extended.)
  func::FuncOp genKernelArgsCreatorFunction(Location loc, OpBuilder &builder,
                                            const std::string &classNameStr,
                                            cudaq::cc::StructType structTy,
                                            FunctionType funcTy) {
    auto structPtrTy = cudaq::cc::PointerType::get(structTy);
    // Local types and values we'll need
    auto *ctx = builder.getContext();
    Type ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
    auto ptrPtrType = cudaq::cc::PointerType::get(ptrI8Ty);
    Type i64Ty = builder.getI64Type();

    // Create the function that we'll fill
    auto funcType = FunctionType::get(ctx, {ptrPtrType, ptrPtrType}, {i64Ty});
    auto argsCreatorFunc = builder.create<func::FuncOp>(
        loc, classNameStr + ".argsCreator", funcType);
    OpBuilder::InsertionGuard guard(builder);
    auto *entry = argsCreatorFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Get the original function args
    auto kernelArgTypes = funcTy.getInputs();

    // Init the struct
    Value stVal = builder.create<cudaq::cc::UndefOp>(loc, structTy);

    // Get the variadic void* args
    auto variadicArgs = entry->getArgument(0);

    // Initialize the counter for extra size.
    Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    Value extraBytes = zero;

    // Loop over the struct elements
    bool hasTrailingData = false;
    for (auto structElementTypeIter : llvm::enumerate(structTy.getMembers())) {
      std::int64_t idx = structElementTypeIter.index();

      // Don't do anything with return args.
      if (idx == static_cast<std::int64_t>(kernelArgTypes.size()))
        break;

      // Get the corresponding cudaq kernel arg type
      auto correspondingKernelArgType = kernelArgTypes[idx];

      // The struct element type for a vector, this type is a i64, the size.
      auto currEleTy = structElementTypeIter.value();

      // Get the pointer out of the void** variadic args - > void* -> TYPE*
      Value argPtrPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrPtrType, variadicArgs,
          SmallVector<cudaq::cc::ComputePtrArg>{
              static_cast<std::int32_t>(idx)});
      Value argPtr = builder.create<cudaq::cc::LoadOp>(loc, ptrI8Ty, argPtrPtr);
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});

      if (auto stdvecTy =
              dyn_cast<cudaq::cc::StdvecType>(correspondingKernelArgType)) {
        // If this is a vector argument, then we will add data to the message
        // buffer's addendum (unless the vector is length 0).
        auto ptrInTy = cudaq::cc::PointerType::get(
            cudaq::opt::factory::stlVectorType(stdvecTy.getElementType()));
        Value arg = builder.create<cudaq::cc::CastOp>(loc, ptrInTy, argPtr);
        // Store the size of the vector.
        auto [topLevelSize, recursiveSize] =
            computeRecursiveVectorSize(builder, loc, arg, ptrInTy, stdvecTy);
        stVal = builder.create<cudaq::cc::InsertValueOp>(
            loc, stVal.getType(), stVal, topLevelSize, off);
        extraBytes =
            builder.create<arith::AddIOp>(loc, extraBytes, recursiveSize);
        hasTrailingData = true;
        continue;
      }
      argPtr = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(currEleTy), argPtr);
      // cast to the struct element type, void* -> TYPE *
      Value loadedVal =
          builder.create<cudaq::cc::LoadOp>(loc, currEleTy, argPtr);
      stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                       stVal, loadedVal, off);
    }

    // Compute the struct size
    auto nullSt = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, zero);
    auto computedOffset = builder.create<cudaq::cc::ComputePtrOp>(
        loc, structPtrTy, nullSt, SmallVector<cudaq::cc::ComputePtrArg>{1});
    Value structSize =
        builder.create<cudaq::cc::CastOp>(loc, i64Ty, computedOffset);

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
      Value vecToBuffer = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI8Ty, buff, SmallVector<Value>{structSize});
      for (auto iter : llvm::enumerate(structTy.getMembers())) {
        std::int64_t idx = iter.index();
        if (idx == static_cast<std::int64_t>(kernelArgTypes.size()))
          break;
        // Get the corresponding cudaq kernel arg type
        auto correspondingKernelArgType = kernelArgTypes[idx];
        if (auto stdvecTy =
                dyn_cast<cudaq::cc::StdvecType>(correspondingKernelArgType)) {
          auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
          auto bytes = builder.create<cudaq::cc::ExtractValueOp>(
              loc, builder.getI64Type(), stVal, off);
          Value argPtrPtr = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrPtrType, variadicArgs,
              SmallVector<cudaq::cc::ComputePtrArg>{
                  static_cast<std::int32_t>(idx)});
          auto ptrInTy = cudaq::cc::PointerType::get(
              cudaq::opt::factory::stlVectorType(stdvecTy.getElementType()));
          Value arg =
              builder.create<cudaq::cc::LoadOp>(loc, ptrI8Ty, argPtrPtr);
          arg = builder.create<cudaq::cc::CastOp>(loc, ptrInTy, arg);
          vecToBuffer = encodeVectorData(builder, loc, bytes, stdvecTy, arg,
                                         vecToBuffer, ptrInTy);
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
              SmallVector<cudaq::cc::ComputePtrArg>{0, i});
          builder.create<cudaq::cc::StoreOp>(loc, retOp.getOperands()[i], mem);
        }
      } else if (auto stdvecTy =
                     dyn_cast<cudaq::cc::StdvecType>(funcTy.getResult(0))) {
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
        auto ptrTy = cudaq::cc::PointerType::get(eleTy);
        auto data = builder.create<cudaq::cc::StdvecDataOp>(loc, ptrTy, stdvec);
        auto mem0 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(ptrTy), cast,
            SmallVector<cudaq::cc::ComputePtrArg>{0, 0});
        builder.create<cudaq::cc::StoreOp>(loc, data, mem0);
        auto i64Ty = builder.getI64Type();
        auto size = builder.create<cudaq::cc::StdvecSizeOp>(loc, i64Ty, stdvec);
        auto mem1 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(i64Ty), cast,
            SmallVector<cudaq::cc::ComputePtrArg>{0, 1});
        builder.create<cudaq::cc::StoreOp>(loc, size, mem1);
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
  std::pair<Value, Value> unpackStdVector(OpBuilder &builder, Location loc,
                                          cudaq::cc::StdvecType stdvecTy,
                                          Value vecSize, Value trailingData) {
    // Convert the pointer-free std::vector<T> to a span structure to be
    // passed. A span structure is a pointer and a size (in element
    // units). Note that this structure may be recursive.
    auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
    Type eleTy = stdvecTy.getElementType();
    auto innerStdvecTy = dyn_cast<cudaq::cc::StdvecType>(eleTy);
    std::size_t eleSize = [&]() -> std::size_t {
      if (isa<quake::PauliWordType>(eleTy))
        return 8;
      return innerStdvecTy ? /*(i64Type/8)*/ 8 : dataLayout->getTypeSize(eleTy);
    }();
    auto eleSizeVal = [&]() -> Value {
      if (eleSize)
        return builder.create<arith::ConstantIntOp>(loc, eleSize, 64);
      // FIXME: should also handle ArrayType here.
      assert(isa<cudaq::cc::StructType>(eleTy) && "handle non-StructType");
      auto strTy = cast<cudaq::cc::StructType>(eleTy);
      auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
      auto arrTy = cudaq::cc::ArrayType::get(strTy);
      auto ptrTy = cudaq::cc::PointerType::get(strTy);
      auto ptrArrTy = cudaq::cc::PointerType::get(arrTy);
      Value nullVal = builder.create<cudaq::cc::CastOp>(loc, ptrArrTy, zero);
      Value sizePtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrTy, nullVal, SmallVector<cudaq::cc::ComputePtrArg>{0, 1});
      auto i64Ty = builder.getI64Type();
      return builder.create<cudaq::cc::CastOp>(loc, i64Ty, sizePtr);
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
      auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
      auto innerVec =
          builder.create<cudaq::cc::CastOp>(loc, ptrI64Ty, trailingData);
      trailingData = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI8Ty, trailingData, vecSize);
      builder.create<cudaq::cc::StoreOp>(loc, trailingData, currentEnd);
      // Loop over each subvector in the vector and recursively unpack it into
      // the vecTmp variable. Leaf vectors do not need a fresh variable. This
      // effectively translates all the size/offset information for all the
      // subvectors into temps.
      Value vecLengthIndex = builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), vecLength);
      cudaq::opt::factory::createInvariantLoop(
          builder, loc, vecLengthIndex,
          [&](OpBuilder &builder, Location loc, Region &, Block &block) {
            Value i = builder.create<arith::IndexCastOp>(loc, i64Ty,
                                                         block.getArgument(0));
            auto innerPtr = builder.create<cudaq::cc::ComputePtrOp>(
                loc, cudaq::cc::PointerType::get(i64Ty), innerVec,
                SmallVector<cudaq::cc::ComputePtrArg>{i});
            Value innerVecSize =
                builder.create<cudaq::cc::LoadOp>(loc, innerPtr);
            Value tmp = builder.create<cudaq::cc::LoadOp>(loc, currentEnd);
            auto unpackPair =
                unpackStdVector(builder, loc, innerStdvecTy, innerVecSize, tmp);
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
    trailingData = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrI8Ty, trailingData, vecSize);
    return {stdVecResult, trailingData};
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
    auto ptrArrayStructTy = cudaq::opt::factory::getIndexedObjectType(structTy);
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    auto nullSt =
        builder.create<cudaq::cc::CastOp>(loc, ptrArrayStructTy, zero);
    auto computedOffset = builder.create<cudaq::cc::ComputePtrOp>(
        loc, structPtrTy, nullSt, SmallVector<cudaq::cc::ComputePtrArg>{1});
    Value structSize =
        builder.create<cudaq::cc::CastOp>(loc, i64Ty, computedOffset);

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
      auto eleTy = structTy.getMembers()[offset];
      auto mem = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(eleTy), castOp,
          SmallVector<cudaq::cc::ComputePtrArg>{0, offset});
      auto sretPtrTy = cudaq::cc::PointerType::get(
          cudaq::opt::factory::getSRetElementType(funcTy));
      auto sretMem = builder.create<cudaq::cc::CastOp>(loc, sretPtrTy, mem);
      args.push_back(sretMem);

      // Rewrite the original kernel's signature and return op(s).
      updateQPUKernelAsSRet(builder, funcOp, newFuncTy);
    }
    for (auto inp : llvm::enumerate(funcTy.getInputs())) {
      Type inTy = inp.value();
      std::int64_t idx = inp.index();
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      if (isa<cudaq::cc::CallableType, cudaq::cc::StructType>(inTy)) {
        auto undef = builder.create<cudaq::cc::UndefOp>(loc, inTy);
        args.push_back(undef);
      } else if (auto stdVecTy = dyn_cast<cudaq::cc::StdvecType>(inTy)) {
        Value vecSize =
            builder.create<cudaq::cc::ExtractValueOp>(loc, i64Ty, val, off);
        auto unpackPair =
            unpackStdVector(builder, loc, stdVecTy, vecSize, trailingData);
        trailingData = unpackPair.second;
        args.push_back(unpackPair.first);
      } else {
        args.push_back(
            builder.create<cudaq::cc::ExtractValueOp>(loc, inTy, val, off));
      }
    }
    auto call = builder.create<func::CallOp>(loc, newFuncTy.getResults(),
                                             funcOp.getName(), args);
    // If and only if the kernel returns non-sret results, then take those
    // values and store them in the results section of the struct. They will
    // eventually be returned to the original caller.
    if (!hiddenSRet && funcTy.getNumResults() == 1) {
      auto eleTy = structTy.getMembers()[offset];
      auto mem = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(eleTy), castOp,
          SmallVector<cudaq::cc::ComputePtrArg>{0, offset});
      builder.create<cudaq::cc::StoreOp>(loc, call.getResult(0), mem);
    }

    // If the original result was a std::vector<T>, then depending on whether
    // this is client-server or not, the thunk function packs the dynamic return
    // data into a message buffer or just returns a pointer to the shared heap
    // allocation, resp.
    bool hasVectorResult = funcTy.getNumResults() == 1 &&
                           isa<cudaq::cc::StdvecType>(funcTy.getResult(0));
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
          loc, cudaq::cc::PointerType::get(structTy.getMembers()[offset]),
          castOp, SmallVector<cudaq::cc::ComputePtrArg>{0, offset});
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
        loc, ptrPtrTy, castSret, SmallVector<cudaq::cc::ComputePtrArg>{0, 0});
    Value vecPtr = builder.create<cudaq::cc::LoadOp>(loc, ptrTy, sret0);
    builder.create<func::CallOp>(loc, std::nullopt, "free", ValueRange{vecPtr});
    auto arrI8Ty = cudaq::cc::ArrayType::get(i8Ty);
    auto ptrArrTy = cudaq::cc::PointerType::get(arrI8Ty);
    auto buffPtr0 = builder.create<cudaq::cc::CastOp>(loc, ptrTy, data);
    builder.create<cudaq::cc::StoreOp>(loc, buffPtr0, sret0);
    auto sret1 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrPtrTy, castSret, SmallVector<cudaq::cc::ComputePtrArg>{0, 1});
    Value byteLen = builder.create<arith::MulIOp>(loc, tSize, vecSize);
    auto buffPtr = builder.create<cudaq::cc::CastOp>(loc, ptrArrTy, data);
    auto endPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTy, buffPtr, SmallVector<cudaq::cc::ComputePtrArg>{0, byteLen});
    builder.create<cudaq::cc::StoreOp>(loc, endPtr, sret1);
    auto sret2 = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrPtrTy, castSret, SmallVector<cudaq::cc::ComputePtrArg>{0, 2});
    builder.create<cudaq::cc::StoreOp>(loc, endPtr, sret2);
  }

  static MutableArrayRef<BlockArgument>
  dropAnyHiddenArguments(MutableArrayRef<BlockArgument> args,
                         FunctionType funcTy, bool hasThisPointer) {
    unsigned count = hasThisPointer ? 1 : 0;
    if (cudaq::opt::factory::hasHiddenSRet(funcTy))
      ++count;
    if (count > 0 && args.size() >= count &&
        std::all_of(args.begin(), args.begin() + count, [](auto i) {
          return isa<cudaq::cc::PointerType>(i.getType());
        }))
      return args.drop_front(count);
    return args;
  }

  // Return the vector's length, computed on the CPU side, in bytes.
  Value computeCpuVectorLengthInBytes(OpBuilder &builder, Location loc,
                                      Value cpuArg, Type eleTy,
                                      cudaq::cc::PointerType cpuVecTy) {
    auto rawSize = getVectorSize(builder, loc, cpuVecTy, cpuArg);
    if (isa<cudaq::cc::StdvecType>(eleTy)) {
      auto three = builder.create<arith::ConstantIntOp>(loc, 3, 64);
      return builder.create<arith::DivSIOp>(loc, rawSize, three);
    }
    return rawSize;
  }

  Value fetchCpuVectorFront(OpBuilder &builder, Location loc, Value cpuArg,
                            cudaq::cc::PointerType cpuVecTy) {
    auto inpStructTy = cast<cudaq::cc::StructType>(cpuVecTy.getElementType());
    auto ptrTtype = cudaq::cc::PointerType::get(inpStructTy.getMembers()[0]);
    auto beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTtype, cpuArg, SmallVector<cudaq::cc::ComputePtrArg>{0, 0});
    auto ptrArrSTy = cudaq::opt::factory::getIndexedObjectType(inpStructTy);
    auto vecPtr = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(ptrArrSTy), beginPtr);
    return builder.create<cudaq::cc::LoadOp>(loc, vecPtr);
  }

  Value recursiveVectorDataCopy(OpBuilder &builder, Location loc, Value cpuArg,
                                Value buffPtr, cudaq::cc::StdvecType stdvecTy,
                                cudaq::cc::PointerType cpuVecTy) {
    auto vecLen =
        computeCpuVectorLengthInBytes(builder, loc, cpuArg, stdvecTy, cpuVecTy);
    auto nested = fetchCpuVectorFront(builder, loc, cpuArg, cpuVecTy);
    auto vecLogicalLen = convertLengthBytesToLengthI64(builder, loc, vecLen);
    auto vecLenIndex = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), vecLogicalLen);
    auto buffPtrTy = buffPtr.getType();
    auto tmp = builder.create<cudaq::cc::AllocaOp>(loc, buffPtrTy);
    auto newEnd = builder.create<cudaq::cc::ComputePtrOp>(
        loc, buffPtrTy, buffPtr, SmallVector<cudaq::cc::ComputePtrArg>{vecLen});
    builder.create<cudaq::cc::StoreOp>(loc, newEnd, tmp);
    auto i64Ty = builder.getI64Type();
    auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
    auto vecBasePtr = builder.create<cudaq::cc::CastOp>(loc, ptrI64Ty, buffPtr);
    auto nestedArr = builder.create<cudaq::cc::CastOp>(loc, cpuVecTy, nested);
    cudaq::opt::factory::createInvariantLoop(
        builder, loc, vecLenIndex,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value i = builder.create<arith::IndexCastOp>(
              loc, builder.getI64Type(), block.getArgument(0));
          auto currBuffPtr = builder.create<cudaq::cc::ComputePtrOp>(
              loc, ptrI64Ty, vecBasePtr,
              SmallVector<cudaq::cc::ComputePtrArg>{i});
          auto cpuSubVec = builder.create<cudaq::cc::ComputePtrOp>(
              loc, cpuVecTy, nestedArr, i);
          Value buff = builder.create<cudaq::cc::LoadOp>(loc, tmp);
          // Compute and save the byte size.
          auto vecSz = computeCpuVectorLengthInBytes(
              builder, loc, cpuSubVec, stdvecTy.getElementType(), cpuVecTy);
          builder.create<cudaq::cc::StoreOp>(loc, vecSz, currBuffPtr);
          // Recursively copy vector data.
          auto endBuff = encodeVectorData(builder, loc, vecSz, stdvecTy,
                                          cpuSubVec, buff, cpuVecTy);
          builder.create<cudaq::cc::StoreOp>(loc, endBuff, tmp);
        });
    return builder.create<cudaq::cc::LoadOp>(loc, tmp);
  }

  /// Recursively encode a `std::vector` into a buffer's addendum. The data is
  /// read from \p cpuArg. The data is \p bytes size long if this is a leaf
  /// vector, otherwise the size is computed on-the-fly during the encoding of
  /// the ragged array.
  Value encodeVectorData(OpBuilder &builder, Location loc, Value bytes,
                         cudaq::cc::StdvecType stdvecTy, Value cpuArg,
                         Value bufferAddendum, cudaq::cc::PointerType ptrInTy) {
    auto eleTy = stdvecTy.getElementType();
    if (auto subVecTy = dyn_cast<cudaq::cc::StdvecType>(eleTy))
      return recursiveVectorDataCopy(builder, loc, cpuArg, bufferAddendum,
                                     subVecTy, ptrInTy);
    return copyVectorData(builder, loc, bytes, cpuArg, bufferAddendum);
  }

  /// Generate an all new entry point body, calling launchKernel in the runtime
  /// library. Pass along the thunk, so the runtime can call the quantum
  /// circuit. These entry points are `operator()` member functions in a class,
  /// so account for the `this` argument here.
  void genNewHostEntryPoint(Location loc, OpBuilder &builder,
                            StringAttr mangledAttr, FunctionType funcTy,
                            Type structTy, LLVM::GlobalOp kernName,
                            func::FuncOp thunk, ModuleOp module,
                            bool addThisPtr) {
    auto *ctx = builder.getContext();
    auto i64Ty = builder.getI64Type();
    auto offset = funcTy.getNumInputs();
    auto thunkTy = getThunkType(ctx);
    auto structPtrTy = cudaq::cc::PointerType::get(structTy);
    FunctionType newFuncTy;
    if (auto *decl = module.lookupSymbol(mangledAttr.getValue())) {
      auto func = dyn_cast<func::FuncOp>(decl);
      if (func && func.empty()) {
        // Do not add any hidden arguments like a `this` pointer.
        newFuncTy = func.getFunctionType();
        func.erase();
      } else {
        decl->emitOpError("object preventing generation of host entry point");
        return;
      }
    } else {
      newFuncTy = cudaq::opt::factory::toCpuSideFuncType(funcTy, addThisPtr);
    }
    auto rewriteEntry =
        builder.create<func::FuncOp>(loc, mangledAttr.getValue(), newFuncTy);
    const bool hiddenSRet = cudaq::opt::factory::hasHiddenSRet(funcTy);
    if (hiddenSRet) {
      // The first argument should be a pointer type if this function has a
      // hidden sret.
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(
              rewriteEntry.getFunctionType().getInput(0))) {
        auto eleTy = cudaq::opt::factory::getSRetElementType(funcTy);
        rewriteEntry.setArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName(),
                                TypeAttr::get(eleTy));
      }
    }

    OpBuilder::InsertionGuard guard(builder);
    auto *rewriteEntryBlock = rewriteEntry.addEntryBlock();
    builder.setInsertionPointToStart(rewriteEntryBlock);
    Value stVal = builder.create<cudaq::cc::UndefOp>(loc, structTy);

    // Process all the arguments for the original call, ignoring the `this`
    // pointer.
    auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
    Value extraBytes = zero;
    bool hasTrailingData = false;
    for (auto inp : llvm::enumerate(dropAnyHiddenArguments(
             rewriteEntryBlock->getArguments(), funcTy, addThisPtr))) {
      Value arg = inp.value();
      Type inTy = arg.getType();
      std::int64_t idx = inp.index();
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      if (isa<cudaq::cc::CallableType, cudaq::cc::StructType>(inTy)) {
        /* do nothing */
      } else if (cudaq::opt::factory::isStdVecArg(inTy)) {
        // Per the CUDAQ spec, `[const] std::vector<T>&` must be passed.
        auto ptrInTy = cast<cudaq::cc::PointerType>(inTy);
        // FIXME: call the `size` member function. For expediency, assume this
        // is an std::vector and the size is the scaled delta between the
        // first two pointers. Use the unscaled size for now.
        auto stdvecTy = cast<cudaq::cc::StdvecType>(funcTy.getInput(idx));
        auto [topLevelSize, recursiveSize] =
            computeRecursiveVectorSize(builder, loc, arg, ptrInTy, stdvecTy);
        stVal = builder.create<cudaq::cc::InsertValueOp>(
            loc, stVal.getType(), stVal, topLevelSize, off);
        extraBytes =
            builder.create<arith::AddIOp>(loc, extraBytes, recursiveSize);
        hasTrailingData = true;
      } else if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(inTy)) {
        // do nothing - we can only encode pointers to std::vector<T>.
      } else {
        stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                         stVal, arg, off);
      }
    }

    // Compute the struct size without the trailing bytes, structSize, and with
    // the trailing bytes, extendedStructSize.
    auto nullSt = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, zero);
    auto computedOffset = builder.create<cudaq::cc::ComputePtrOp>(
        loc, structPtrTy, nullSt, SmallVector<cudaq::cc::ComputePtrArg>{1});
    Value structSize =
        builder.create<cudaq::cc::CastOp>(loc, i64Ty, computedOffset);
    Value extendedStructSize =
        builder.create<arith::AddIOp>(loc, structSize, extraBytes);

    // Allocate our struct to save the argument to.
    auto i8Ty = builder.getI8Type();
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto buff =
        builder.create<cudaq::cc::AllocaOp>(loc, i8Ty, extendedStructSize);

    auto temp = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, buff);

    // Store the arguments to the argument section.
    builder.create<cudaq::cc::StoreOp>(loc, stVal, temp);

    // Append the vector data to the end of the struct.
    if (hasTrailingData) {
      Value vecToBuffer = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrI8Ty, buff, SmallVector<Value>{structSize});
      // Ignore any hidden `this` argument.
      for (auto inp : llvm::enumerate(dropAnyHiddenArguments(
               rewriteEntryBlock->getArguments(), funcTy, addThisPtr))) {
        Value arg = inp.value();
        Type inTy = arg.getType();
        std::int64_t idx = inp.index();
        auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
        if (cudaq::opt::factory::isStdVecArg(inTy)) {
          auto bytes = builder.create<cudaq::cc::ExtractValueOp>(
              loc, builder.getI64Type(), stVal, off);
          auto stdvecTy = cast<cudaq::cc::StdvecType>(funcTy.getInput(idx));
          auto ptrInTy = cast<cudaq::cc::PointerType>(inTy);
          vecToBuffer = encodeVectorData(builder, loc, bytes, stdvecTy, arg,
                                         vecToBuffer, ptrInTy);
        }
      }
    }

    // Prepare to call the `launchKernel` runtime library entry point.
    Value loadKernName = builder.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(kernName.getType()),
        kernName.getSymName());
    Value loadThunk =
        builder.create<func::ConstantOp>(loc, thunkTy, thunk.getName());
    auto castLoadKernName =
        builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, loadKernName);
    auto castLoadThunk =
        builder.create<cudaq::cc::FuncToPtrOp>(loc, ptrI8Ty, loadThunk);
    auto castTemp = builder.create<cudaq::cc::CastOp>(loc, ptrI8Ty, temp);

    auto resultOffset = [&]() -> Value {
      if (funcTy.getNumResults() == 0)
        return builder.create<arith::ConstantIntOp>(loc, NoResultOffset, 64);
      int offset = funcTy.getNumInputs();
      auto gep = builder.create<cudaq::cc::ComputePtrOp>(
          loc, structPtrTy, nullSt,
          SmallVector<cudaq::cc::ComputePtrArg>{0, offset});
      return builder.create<cudaq::cc::CastOp>(loc, i64Ty, gep);
    }();

    // Generate the call to `launchKernel`.
    builder.create<func::CallOp>(
        loc, std::nullopt, cudaq::runtime::launchKernelFuncName,
        ArrayRef<Value>{castLoadKernName, castLoadThunk, castTemp,
                        extendedStructSize, resultOffset});

    // If and only if this kernel returns a value, unpack and load the
    // result value(s) from the struct returned by `launchKernel` and return
    // them to our caller.
    SmallVector<Value> results;
    for (auto res : llvm::enumerate(funcTy.getResults())) {
      int off = res.index() + offset;
      if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(res.value())) {
        auto eleTy = vecTy.getElementType();
        auto ptrTy = cudaq::cc::PointerType::get(eleTy);
        auto gep0 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(ptrTy), temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off});
        auto dataPtr = builder.create<cudaq::cc::LoadOp>(loc, gep0);
        auto lenPtrTy = cudaq::cc::PointerType::get(builder.getI64Type());
        auto gep1 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, lenPtrTy, temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off + 1});
        auto vecLen = builder.create<cudaq::cc::LoadOp>(loc, gep1);
        if (vecTy.getElementType() == builder.getI1Type()) {
          genStdvecBoolFromInitList(loc, builder,
                                    rewriteEntryBlock->getArguments().front(),
                                    dataPtr, vecLen);
        } else {
          auto size = (eleTy.getIntOrFloatBitWidth() + 7) / 8;
          Value tSize = builder.create<arith::ConstantIntOp>(loc, size, 64);
          genStdvecTFromInitList(loc, builder,
                                 rewriteEntryBlock->getArguments().front(),
                                 dataPtr, tSize, vecLen);
        }
        offset++;
      } else {
        auto gep = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(res.value()), temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off});
        Value loadVal = builder.create<cudaq::cc::LoadOp>(loc, gep);
        if (hiddenSRet) {
          auto castPtr = builder.create<cudaq::cc::CastOp>(
              loc, temp.getType(), rewriteEntryBlock->getArguments().front());
          auto outPtr = builder.create<cudaq::cc::ComputePtrOp>(
              loc, cudaq::cc::PointerType::get(res.value()), castPtr,
              SmallVector<cudaq::cc::ComputePtrArg>{0, off});
          builder.create<cudaq::cc::StoreOp>(loc, loadVal, outPtr);
        } else {
          results.push_back(loadVal);
        }
      }
    }
    builder.create<func::ReturnOp>(loc, results);
  }

  // An entry function that takes a quantum type argument cannot be called
  // directly from C++ (classical) code. It must be called via other quantum
  // code.
  bool hasLegalType(FunctionType funTy) {
    for (auto ty : funTy.getInputs())
      if (quake::isaQuantumType(ty))
        return false;
    for (auto ty : funTy.getResults())
      if (quake::isaQuantumType(ty))
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
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::runtime::launchKernelFuncName))) {
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

    // Gather a work list of functions
    SmallVector<func::FuncOp> workList;
    for (auto &op : *module.getBody())
      if (auto funcOp = dyn_cast<func::FuncOp>(op))
        if (funcOp.getName().startswith(cudaq::runtime::cudaqGenPrefixName) &&
            hasLegalType(funcOp.getFunctionType()))
          workList.push_back(funcOp);

    LLVM_DEBUG(llvm::dbgs()
               << workList.size() << " kernel entry functions to process\n");
    for (auto funcOp : workList) {
      auto loc = funcOp.getLoc();
      [[maybe_unused]] auto className =
          funcOp.getName().drop_front(cudaq::runtime::cudaqGenPrefixLength);
      LLVM_DEBUG(llvm::dbgs() << "processing function " << className << '\n');
      auto classNameStr = className.str();

      // Create a constant with the name of the kernel as a C string.
      auto kernName = builder.create<LLVM::GlobalOp>(
          loc, cudaq::opt::factory::getStringType(ctx, className.size() + 1),
          /*isConstant=*/true, LLVM::Linkage::External,
          classNameStr + ".kernelName",
          builder.getStringAttr(classNameStr + '\0'), /*alignment=*/0);

      // Create a new struct type to pass arguments and results.
      auto funcTy = funcOp.getFunctionType();
      auto structTy = buildStructType(classNameStr, funcTy);

      // Generate thunk, `<kernel>.thunk`, to call back to the MLIR code.
      auto thunk = genThunkFunction(loc, builder, classNameStr, structTy,
                                    funcTy, funcOp);

      auto argsCreatorFunc = genKernelArgsCreatorFunction(
          loc, builder, classNameStr, structTy, funcTy);

      if (!mangledNameMap.contains(funcOp.getName()))
        continue;
      auto mangledAttr = mangledNameMap.getAs<StringAttr>(funcOp.getName());
      assert(mangledAttr && "funcOp must appear in mangled name map");
      auto thunkNameStr = thunk.getName().str();

      // Generate a new mangled function on the host side to call the
      // callback function.
      genNewHostEntryPoint(loc, builder, mangledAttr, funcTy, structTy,
                           kernName, thunk, module,
                           !funcOp->hasAttr("no_this"));

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
            loc, cudaq::opt::factory::getPointerType(kernName.getType()),
            kernName.getSymName());
        auto castKernRef =
            builder.create<cudaq::cc::CastOp>(loc, ptrType, kernRef);
        builder.create<func::CallOp>(loc, std::nullopt, cudaqRegisterKernelName,
                                     ValueRange{castKernRef});

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

        // Check if this is a lambda mangled name
        auto demangledPtr = abi::__cxa_demangle(
            mangledAttr.getValue().str().c_str(), nullptr, nullptr, nullptr);
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
