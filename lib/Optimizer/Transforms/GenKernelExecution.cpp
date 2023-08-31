/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/CUDAQBuilder.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "clang/Basic/Version.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Transforms/Passes.h"
#include <cxxabi.h>
#include <regex>

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
    : public cudaq::opt::GenerateKernelExecutionBase<GenerateKernelExecution> {
public:
  GenerateKernelExecution() = default;

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
    // Add all argument types, translating std::vector to a length.
    for (auto inTy : funcTy.getInputs()) {
      if (inTy.isa<cudaq::cc::CallableType, cudaq::cc::StructType>())
        eleTys.push_back(IntegerType::get(ctx, 64));
      else if (inTy.isa<cudaq::cc::StdvecType, quake::VeqType>())
        eleTys.push_back(IntegerType::get(ctx, 64));
      else
        eleTys.push_back(inTy);
    }
    // Add all result types, translating std::vector to a length.
    for (auto outTy : funcTy.getResults()) {
      if (outTy.isa<cudaq::cc::CallableType, cudaq::cc::StructType>()) {
        eleTys.push_back(IntegerType::get(ctx, 64));
      } else if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(outTy)) {
        eleTys.push_back(cudaq::cc::PointerType::get(vecTy.getElementType()));
        eleTys.push_back(IntegerType::get(ctx, 64));
      } else {
        eleTys.push_back(outTy);
      }
    }
    return cudaq::cc::StructType::get(ctx, eleTys);
  }

  FunctionType getThunkType(MLIRContext *ctx) {
    auto ptrTy = cudaq::cc::PointerType::get(IntegerType::get(ctx, 8));
    return FunctionType::get(
        ctx, {ptrTy, IntegerType::get(ctx, 1)},
        {cudaq::cc::StructType::get(
            ctx, ArrayRef<Type>{ptrTy, IntegerType::get(ctx, 64)})});
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
    auto inpStructTy = ptrTy.getElementType().cast<cudaq::cc::StructType>();

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

  Value copyVectorData(OpBuilder &builder, Location loc, Value stVal, Value arg,
                       Value vecToBuffer, DenseI64ArrayAttr off) {
    auto falseAttr = IntegerAttr::get(builder.getI1Type(), 0);
    auto notVolatile =
        builder.create<arith::ConstantOp>(loc, builder.getI1Type(), falseAttr);
    // memcpy from arg->begin to vecToBuffer, size bytes.
    auto bytes = builder.create<cudaq::cc::ExtractValueOp>(
        loc, builder.getI64Type(), stVal, off);
    auto inStructTy = arg.getType()
                          .cast<cudaq::cc::PointerType>()
                          .getElementType()
                          .cast<cudaq::cc::StructType>();
    auto beginPtr = builder.create<cudaq::cc::ComputePtrOp>(
        loc, cudaq::cc::PointerType::get(inStructTy.getMembers()[0]), arg,
        SmallVector<cudaq::cc::ComputePtrArg>{0, 0});
    auto fromBuff = builder.create<cudaq::cc::LoadOp>(loc, beginPtr);
    auto vecFromBuff = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), fromBuff);
    builder.create<func::CallOp>(
        loc, std::nullopt, llvmMemCopyIntrinsic,
        SmallVector<Value>{vecToBuffer, vecFromBuff, bytes, notVolatile});
    // Increment vecToBuffer by size bytes.
    return builder.create<cudaq::cc::ComputePtrOp>(
        loc, vecToBuffer.getType(), vecToBuffer, SmallVector<Value>{bytes});
  }

  // Create a function that takes void**, void* and returns void
  func::FuncOp genKernelArgsCreatorFunction(Location loc, OpBuilder &builder,
                                            const std::string &classNameStr,
                                            Type structPtrTy,
                                            FunctionType funcTy) {
    // Local types and values we'll need
    auto *ctx = builder.getContext();
    auto ptrType = cudaq::cc::PointerType::get(builder.getI8Type());
    auto ptrPtrType = cudaq::cc::PointerType::get(ptrType);
    Type i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
    Type i64Ty = builder.getI64Type();
    Attribute zeroAttr = builder.getI64IntegerAttr(0);

    // Create the function that we'll fill
    auto funcType = FunctionType::get(ctx, {ptrPtrType, ptrPtrType}, {i64Ty});
    auto argsCreatorFunc = builder.create<func::FuncOp>(
        loc, classNameStr + ".argsCreator", funcType);
    auto insPt = builder.saveInsertionPoint();
    auto *entry = argsCreatorFunc.addEntryBlock();
    builder.setInsertionPointToStart(entry);

    // Get the struct type
    auto structType = structPtrTy.cast<cudaq::cc::PointerType>()
                          .getElementType()
                          .cast<cudaq::cc::StructType>();
    // Get the original function args
    auto kernelArgTypes = funcTy.getInputs();

    // Init the struct
    Value stVal = builder.create<cudaq::cc::UndefOp>(loc, structType);

    // Get the variadic void* args
    auto variadicArgs = entry->getArgument(0);

    // We want to keep track of all the vector args we see
    SmallVector<std::pair<std::int64_t, Value>> vectorArgIndices;

    // Initialize the counter for extra size.
    Value zero = builder.create<arith::ConstantOp>(loc, i64Ty, zeroAttr);
    Value extraBytes = zero;

    // Loop over the struct elements
    for (auto structElementTypeIter :
         llvm::enumerate(structType.getMembers())) {
      std::size_t idx = structElementTypeIter.index();

      // Don't do anything with return args.
      if (idx == kernelArgTypes.size())
        break;

      auto off = DenseI64ArrayAttr::get(
          ctx, ArrayRef<std::int64_t>{static_cast<std::int64_t>(idx)});

      // Get the corresponding cudaq kernel arg type
      auto correspondingKernelArgType = kernelArgTypes[idx];

      // The struct element type
      // for a vector, this type is a i64, the size.
      auto type = structElementTypeIter.value();

      // Get the pointer out of the void** variadic args - > void* -> TYPE*
      Value argPtrPtr = builder.create<cudaq::cc::ComputePtrOp>(
          loc, ptrPtrType, variadicArgs,
          SmallVector<cudaq::cc::ComputePtrArg>{static_cast<int32_t>(idx)});
      Value argPtr = builder.create<cudaq::cc::LoadOp>(loc, ptrType, argPtrPtr);

      // Is this a vecType, storing it to a bool so we can check
      // in multiple parts of the following code
      auto vecType =
          dyn_cast<cudaq::cc::StdvecType>(correspondingKernelArgType);
      bool isVecType = vecType != nullptr;
      Value loadedVal;
      if (isVecType) {
        auto vecEleTy = vecType.getElementType();
        auto vecElePtrTy = cudaq::cc::PointerType::get(vecEleTy);
        SmallVector<Type> vecStructEleTys = {vecElePtrTy, vecElePtrTy,
                                             vecElePtrTy};
        type = cudaq::cc::PointerType::get(
            cudaq::cc::StructType::get(ctx, vecStructEleTys));
        loadedVal = builder.create<cudaq::cc::CastOp>(loc, type, argPtr);
      } else {
        argPtr = builder.create<cudaq::cc::CastOp>(
            loc, cudaq::cc::PointerType::get(type), argPtr);
        // cast to the struct element type, void* -> TYPE *
        loadedVal = builder.create<cudaq::cc::LoadOp>(loc, type, argPtr);
      }

      if (isVecType) {
        // Store the index and the vec value
        vectorArgIndices.push_back(std::make_pair(idx, loadedVal));
        // compute the extra size needed for the vector data
        loadedVal = getVectorSize(
            builder, loc, type.cast<cudaq::cc::PointerType>(), loadedVal);
        extraBytes = builder.create<arith::AddIOp>(loc, extraBytes, loadedVal);
      }

      stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                       stVal, loadedVal, off);
    }

    // Compute the struct size
    auto nullSt = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, zero);
    auto computedOffset = builder.create<cudaq::cc::ComputePtrOp>(
        loc, structPtrTy, nullSt, SmallVector<cudaq::cc::ComputePtrArg>{1});
    Value structSize =
        builder.create<cudaq::cc::CastOp>(loc, i64Ty, computedOffset);

    // If no vector args, handle this simple case and drop out
    if (vectorArgIndices.empty()) {
      Value buff = builder
                       .create<func::CallOp>(loc, i8PtrTy, "malloc",
                                             ValueRange(structSize))
                       .getResult(0);

      Value casted = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, buff);
      builder.create<cudaq::cc::StoreOp>(loc, stVal, casted);
      builder.create<cudaq::cc::StoreOp>(loc, buff, entry->getArgument(1));
      builder.create<func::ReturnOp>(loc, ValueRange{structSize});
      builder.restoreInsertionPoint(insPt);
      return argsCreatorFunc;
    }

    // Here we do have vector args
    Value extendedStructSize =
        builder.create<arith::AddIOp>(loc, structSize, extraBytes);
    Value buff = builder
                     .create<func::CallOp>(loc, i8PtrTy, "malloc",
                                           ValueRange(extendedStructSize))
                     .getResult(0);

    auto casted = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, buff);
    builder.create<cudaq::cc::StoreOp>(loc, stVal, casted);
    Value vecToBuffer = builder.create<cudaq::cc::ComputePtrOp>(
        loc, i8PtrTy, buff, SmallVector<Value>{structSize});

    for (auto [idx, vecVal] : vectorArgIndices) {
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      vecToBuffer =
          copyVectorData(builder, loc, stVal, vecVal, vecToBuffer, off);
    }

    builder.create<cudaq::cc::StoreOp>(loc, buff, entry->getArgument(1));
    builder.create<func::ReturnOp>(loc, ValueRange{extendedStructSize});
    builder.restoreInsertionPoint(insPt);
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
                                Type structPtrTy, FunctionType funcTy,
                                func::FuncOp funcOp) {
    auto *ctx = builder.getContext();
    auto thunkTy = getThunkType(ctx);
    auto thunk =
        builder.create<func::FuncOp>(loc, classNameStr + ".thunk", thunkTy);
    auto insPt = builder.saveInsertionPoint();
    auto *thunkEntry = thunk.addEntryBlock();
    builder.setInsertionPointToStart(thunkEntry);
    auto cast = builder.create<cudaq::cc::CastOp>(loc, structPtrTy,
                                                  thunkEntry->getArgument(0));
    auto isClientServer = thunkEntry->getArgument(1);
    Value val = builder.create<cudaq::cc::LoadOp>(loc, cast);
    auto i64Ty = builder.getIntegerType(64);

    // Compute the struct size without the trailing bytes, structSize.
    auto zeroAttr = builder.getI64IntegerAttr(0);
    auto zero = builder.create<arith::ConstantOp>(loc, i64Ty, zeroAttr);
    auto nullSt = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, zero);
    auto computedOffset = builder.create<cudaq::cc::ComputePtrOp>(
        loc, structPtrTy, nullSt, SmallVector<cudaq::cc::ComputePtrArg>{1});
    Value structSize =
        builder.create<cudaq::cc::CastOp>(loc, i64Ty, computedOffset);

    // Compute location of trailing bytes.
    auto i8PtrTy = cudaq::cc::PointerType::get(builder.getI8Type());
    Value trailingData = builder.create<cudaq::cc::ComputePtrOp>(
        loc, i8PtrTy, thunkEntry->getArgument(0), structSize);

    // Unpack the arguments in the struct and build the argument list for
    // the call to the kernel code.
    SmallVector<Value> args;
    for (auto inp : llvm::enumerate(funcTy.getInputs())) {
      Type inTy = inp.value();
      std::int64_t idx = inp.index();
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      if (inTy.isa<cudaq::cc::CallableType, cudaq::cc::StructType>()) {
        auto undef = builder.create<cudaq::cc::UndefOp>(loc, inTy);
        args.push_back(undef);
      } else if (inTy.isa<cudaq::cc::StdvecType, quake::VeqType>()) {
        Type eleTy = IntegerType::get(ctx, /*FIXME sizeof a pointer?*/ 64);
        if (auto memrefTy = dyn_cast<cudaq::cc::StdvecType>(inTy))
          eleTy = memrefTy.getElementType();
        auto stdvecTy = cudaq::cc::StdvecType::get(ctx, eleTy);
        // Must divide by byte, 8 bits.
        auto eleSize = eleTy.getIntOrFloatBitWidth() / 8;
        Value vecSize =
            builder.create<cudaq::cc::ExtractValueOp>(loc, i64Ty, val, off);
        auto eleSizeAttr = builder.getI64IntegerAttr(eleSize);
        auto eleSizeVal =
            builder.create<arith::ConstantOp>(loc, i64Ty, eleSizeAttr);
        auto vecLength =
            builder.create<arith::DivSIOp>(loc, vecSize, eleSizeVal);
        // The data is at trailingData and is valid for vecLength of eleTy.
        auto castData = builder.create<cudaq::cc::CastOp>(
            loc, cudaq::cc::PointerType::get(eleTy), trailingData);
        args.push_back(builder.create<cudaq::cc::StdvecInitOp>(
            loc, stdvecTy, castData, vecLength));
        trailingData = builder.create<cudaq::cc::ComputePtrOp>(
            loc, i8PtrTy, trailingData, vecSize);
      } else {
        args.push_back(
            builder.create<cudaq::cc::ExtractValueOp>(loc, inTy, val, off));
      }
    }
    auto call = builder.create<func::CallOp>(loc, funcTy.getResults(),
                                             funcOp.getName(), args);
    auto offset = funcTy.getNumInputs();
    bool hasVectorResult = false;
    // If and only if the kernel returns results, then take those values and
    // store them in the results section of the struct. They will eventually
    // be returned to the original caller.
    for (auto res : llvm::enumerate(funcTy.getResults())) {
      int off = res.index() + offset;
      if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(res.value())) {
        auto callResult = call.getResult(res.index());
        auto ptrTy = cudaq::cc::PointerType::get(vecTy.getElementType());
        auto gep0 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(ptrTy), cast,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off});
        auto pointer =
            builder.create<cudaq::cc::StdvecDataOp>(loc, ptrTy, callResult);
        builder.create<cudaq::cc::StoreOp>(loc, pointer, gep0);
        auto lenTy = builder.getI64Type();
        auto gep1 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(lenTy), cast,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off + 1});
        auto length =
            builder.create<cudaq::cc::StdvecSizeOp>(loc, lenTy, callResult);
        builder.create<cudaq::cc::StoreOp>(loc, length, gep1);
        offset++;
        hasVectorResult = true;
      } else {
        auto gep = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(res.value()), cast,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off});
        builder.create<cudaq::cc::StoreOp>(loc, call.getResult(res.index()),
                                           gep);
      }
    }
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
      auto structTy = structPtrTy.cast<cudaq::cc::PointerType>()
                          .getElementType()
                          .cast<cudaq::cc::StructType>();
      auto gepRes = builder.create<cudaq::cc::ComputePtrOp>(
          loc, cudaq::cc::PointerType::get(structTy.getMembers()[offset]), cast,
          SmallVector<cudaq::cc::ComputePtrArg>{0, offset});
      auto gepRes2 = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(thunkTy.getResults()[0]), gepRes);
      auto res = builder.create<func::CallOp>(
          loc, thunkTy.getResults()[0], "__nvqpp_createDynamicResult",
          ValueRange{thunkEntry->getArgument(0), structSize, gepRes2});
      builder.create<func::ReturnOp>(loc, res.getResult(0));
      builder.setInsertionPointToEnd(elseBlock);
    }
    auto zeroRes =
        builder.create<func::CallOp>(loc, thunkTy.getResults()[0],
                                     "__nvqpp_zeroDynamicResult", ValueRange{});
    builder.create<func::ReturnOp>(loc, zeroRes.getResult(0));
    builder.restoreInsertionPoint(insPt);
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
    builder.create<func::CallOp>(loc, std::nullopt, stdvecBoolCtorFromInitList,
                                 ArrayRef<Value>{castSret, castData, size});
  }

  static MutableArrayRef<BlockArgument>
  dropAnyHiddenArguments(MutableArrayRef<BlockArgument> args,
                         FunctionType funcTy) {
    if (!args.empty() && isa<cudaq::cc::PointerType>(args[0].getType()))
      return args.drop_front(cudaq::opt::factory::hasHiddenSRet(funcTy) ? 2
                                                                        : 1);
    return args;
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
    auto i64Ty = builder.getIntegerType(64);
    auto zeroAttr = builder.getI64IntegerAttr(0);
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
    auto insPt = builder.saveInsertionPoint();
    auto *rewriteEntryBlock = rewriteEntry.addEntryBlock();
    builder.setInsertionPointToStart(rewriteEntryBlock);
    Value stVal = builder.create<cudaq::cc::UndefOp>(loc, structTy);

    // Process all the arguments for the original call, ignoring the `this`
    // pointer.
    auto zero = builder.create<arith::ConstantOp>(loc, i64Ty, zeroAttr);
    Value extraBytes = zero;
    bool hasTrailingData = false;
    for (auto inp : llvm::enumerate(dropAnyHiddenArguments(
             rewriteEntryBlock->getArguments(), funcTy))) {
      Value arg = inp.value();
      Type inTy = arg.getType();
      std::int64_t idx = inp.index();
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      if (inTy.isa<cudaq::cc::CallableType, cudaq::cc::StructType>()) {
        /* do nothing */
      } else if (cudaq::opt::factory::isStdVecArg(inTy)) {
        auto ptrTy = dyn_cast<cudaq::cc::PointerType>(inTy);
        // FIXME: call the `size` member function. For expediency, assume this
        // is an std::vector and the size is the scaled delta between the
        // first two pointers. Use the unscaled size for now.
        auto sizeBytes = getVectorSize(builder, loc, ptrTy, arg);
        stVal = builder.create<cudaq::cc::InsertValueOp>(loc, stVal.getType(),
                                                         stVal, sizeBytes, off);
        extraBytes = builder.create<arith::AddIOp>(loc, extraBytes, sizeBytes);
        hasTrailingData = true;
      } else if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(inTy)) {
        /*do nothing*/
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
    auto i8PtrTy = cudaq::cc::PointerType::get(i8Ty);
    auto buff =
        builder.create<cudaq::cc::AllocaOp>(loc, i8Ty, extendedStructSize);

    auto temp = builder.create<cudaq::cc::CastOp>(loc, structPtrTy, buff);

    // Store the arguments to the argument section.
    builder.create<cudaq::cc::StoreOp>(loc, stVal, temp);

    // Append the vector data to the end of the struct.
    if (hasTrailingData) {
      Value vecToBuffer = builder.create<cudaq::cc::ComputePtrOp>(
          loc, i8PtrTy, buff, SmallVector<Value>{structSize});
      for (auto inp :
           llvm::enumerate(rewriteEntryBlock->getArguments().drop_front(1))) {
        Value arg = inp.value();
        Type inTy = arg.getType();
        std::int64_t idx = inp.index();
        auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
        if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(inTy)) {
          // memcpy from arg->begin to vecToBuffer, size bytes.
          vecToBuffer =
              copyVectorData(builder, loc, stVal, arg, vecToBuffer, off);
        }
      }
    }

    // Prepare to call the `launchKernel` runtime library entry point.
    Value loadKernName = builder.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(kernName.getType()),
        kernName.getSymName());
    Value loadThunk =
        builder.create<func::ConstantOp>(loc, thunkTy, thunk.getName());
    auto castLoadKernName = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), loadKernName);
    auto castLoadThunk = builder.create<cudaq::cc::FuncToPtrOp>(
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), loadThunk);
    auto castTemp = builder.create<cudaq::cc::CastOp>(
        loc, cudaq::cc::PointerType::get(builder.getI8Type()), temp);

    auto resultOffset = [&]() -> Value {
      if (funcTy.getNumResults() == 0) {
        auto notZeroAttr = builder.getI64IntegerAttr(NoResultOffset);
        return builder.create<arith::ConstantOp>(loc, i64Ty, notZeroAttr);
      }
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
        auto ptrTy = cudaq::cc::PointerType::get(vecTy.getElementType());
        auto gep0 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(ptrTy), temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off});
        auto dataPtr = builder.create<cudaq::cc::LoadOp>(loc, gep0);
        auto lenPtrTy = cudaq::cc::PointerType::get(builder.getI64Type());
        auto gep1 = builder.create<cudaq::cc::ComputePtrOp>(
            loc, lenPtrTy, temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off + 1});
        auto vecLen = builder.create<cudaq::cc::LoadOp>(loc, gep1);
        if (vecTy.getElementType() == builder.getI1Type())
          genStdvecBoolFromInitList(loc, builder,
                                    rewriteEntryBlock->getArguments().front(),
                                    dataPtr, vecLen);
        else
          TODO_loc(loc, "return is std::vector<T> where T != bool");
        offset++;
      } else {
        auto gep = builder.create<cudaq::cc::ComputePtrOp>(
            loc, cudaq::cc::PointerType::get(res.value()), temp,
            SmallVector<cudaq::cc::ComputePtrArg>{0, off});
        results.push_back(builder.create<cudaq::cc::LoadOp>(loc, gep));
      }
    }
    builder.create<func::ReturnOp>(loc, results);
    builder.restoreInsertionPoint(insPt);
  }

  // An entry function that takes a quantum type argument cannot be called
  // directly from C++ (classical) code. It must be called via other quantum
  // code.
  bool hasLegalType(FunctionType funTy) {
    for (auto ty : funTy.getInputs())
      if (ty.isa<quake::RefType, quake::VeqType>())
        return false;
    for (auto ty : funTy.getResults())
      if (ty.isa<quake::RefType, quake::VeqType>())
        return false;
    return true;
  }

  void runOnOperation() override {
    auto module = getOperation();
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
    if (failed(irBuilder.loadIntrinsic(module, stdvecBoolCtorFromInitList))) {
      module.emitError(std::string("could not load ") +
                       stdvecBoolCtorFromInitList);
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, llvmMemCopyIntrinsic))) {
      module.emitError(std::string("could not load ") + llvmMemCopyIntrinsic);
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
      auto structPtrTy = cudaq::cc::PointerType::get(structTy);

      // Generate thunk, `<kernel>.thunk`, to call back to the MLIR code.
      auto thunk = genThunkFunction(loc, builder, classNameStr, structPtrTy,
                                    funcTy, funcOp);

      auto argsCreatorFunc = genKernelArgsCreatorFunction(
          loc, builder, classNameStr, structPtrTy, funcTy);

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
      auto insPt = builder.saveInsertionPoint();
      auto *initFunEntry = initFun.addEntryBlock();
      builder.setInsertionPointToStart(initFunEntry);
      auto kernRef = builder.create<LLVM::AddressOfOp>(
          loc, cudaq::opt::factory::getPointerType(kernName.getType()),
          kernName.getSymName());
      auto castKernRef = builder.create<cudaq::cc::CastOp>(
          loc, cudaq::cc::PointerType::get(builder.getI8Type()), kernRef);
      builder.create<func::CallOp>(loc, std::nullopt, cudaqRegisterKernelName,
                                   ValueRange{castKernRef});

      // Register the argsCreator too
      auto ptrPtrType = cudaq::cc::PointerType::get(ptrType);
      auto argsCreatorFuncType = FunctionType::get(
          ctx, {ptrPtrType, ptrPtrType}, {builder.getI64Type()});
      Value loadArgsCreator = builder.create<func::ConstantOp>(
          loc, argsCreatorFuncType, argsCreatorFunc.getName());
      auto castLoadArgsCreator =
          builder.create<cudaq::cc::FuncToPtrOp>(loc, ptrType, loadArgsCreator);
      builder.create<func::CallOp>(
          loc, std::nullopt, cudaqRegisterArgsCreator,
          ValueRange{castKernRef, castLoadArgsCreator});

      // Check if this is a lambda mangled name
      auto demangledPtr = abi::__cxa_demangle(
          mangledAttr.getValue().str().c_str(), nullptr, nullptr, nullptr);
      if (demangledPtr) {
        std::string demangledName(demangledPtr);
        demangledName = std::regex_replace(demangledName,
                                           std::regex("::operator()(.*)"), "");
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
          builder.create<LLVM::CallOp>(
              loc, std::nullopt, cudaqRegisterLambdaName,
              ValueRange{castLambdaRef, castKernelRef});
        }
      }

      builder.create<LLVM::ReturnOp>(loc, ValueRange{});
      builder.restoreInsertionPoint(insPt);

      // Create a global with a default ctor to be run at program startup.
      // The ctor will execute the above function, which will register this
      // kernel as having been processed.
      cudaq::opt::factory::createGlobalCtorCall(
          module, FlatSymbolRefAttr::get(ctx, initFun.getName()));
      LLVM_DEBUG(llvm::dbgs() << module << '\n');
    }
    out.keep();
  }
};
} // namespace

std::unique_ptr<Pass> cudaq::opt::createGenerateKernelExecution() {
  return std::make_unique<GenerateKernelExecution>();
}
