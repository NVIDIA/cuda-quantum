/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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
  LLVM::LLVMStructType buildStructType(const std::string &name,
                                       FunctionType funcTy) {
    auto *ctx = funcTy.getContext();
    SmallVector<Type> eleTys;
    // Add all argument types, translating std::vector to a length.
    for (auto inTy : funcTy.getInputs()) {
      if (inTy.isa<cudaq::cc::LambdaType, LLVM::LLVMStructType>())
        eleTys.push_back(IntegerType::get(ctx, 64));
      else if (inTy.isa<cudaq::cc::StdvecType, quake::VeqType>())
        eleTys.push_back(IntegerType::get(ctx, 64));
      else
        eleTys.push_back(inTy);
    }
    // Add all result types, translating std::vector to a length.
    for (auto outTy : funcTy.getResults()) {
      if (outTy.isa<cudaq::cc::LambdaType, LLVM::LLVMStructType>()) {
        eleTys.push_back(IntegerType::get(ctx, 64));
      } else if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(outTy)) {
        eleTys.push_back(
            cudaq::opt::factory::getPointerType(vecTy.getElementType()));
        eleTys.push_back(IntegerType::get(ctx, 64));
      } else {
        eleTys.push_back(outTy);
      }
    }
    return LLVM::LLVMStructType::getLiteral(ctx, eleTys);
  }

  // FIXME: We should get the underlying structure of a std::vector from the
  // AST. For expediency, we just construct the expected type directly here.
  LLVM::LLVMStructType stlVectorType(Type eleTy) {
    MLIRContext *ctx = eleTy.getContext();
    auto elePtrTy = cudaq::opt::factory::getPointerType(eleTy);
    SmallVector<Type> eleTys = {elePtrTy, elePtrTy, elePtrTy};
    return LLVM::LLVMStructType::getLiteral(ctx, eleTys);
  }

  bool hasHiddenSRet(FunctionType funcTy) {
    return funcTy.getNumResults() == 1 &&
           funcTy.getResult(0).isa<cudaq::cc::StdvecType>();
  }

  FunctionType toLLVMFuncType(FunctionType funcTy) {
    auto *ctx = funcTy.getContext();
    // In the default case, there is always a default "this" argument to the
    // kernel entry function. The CUDA Quantum language spec doesn't allow the
    // kernel object to contain data members (yet), so we can ignore this
    // pointer.
    auto ptrTy = cudaq::opt::factory::getPointerType(ctx);
    SmallVector<Type> inputTys = {ptrTy};
    bool hasSRet = false;
    if (hasHiddenSRet(funcTy)) {
      // When the kernel is returning a std::vector<T> result, the result is
      // returned via a sret argument in the first position. When this argument
      // is added, the this pointer becomes the second argument. Both are opaque
      // pointers at this point.
      inputTys.push_back(ptrTy);
      hasSRet = true;
    }

    // Add all the explicit (not hidden) arguments after the hidden ones.
    for (auto inTy : funcTy.getInputs()) {
      if (auto memrefTy = dyn_cast<cudaq::cc::StdvecType>(inTy))
        inputTys.push_back(cudaq::opt::factory::getPointerType(
            stlVectorType(memrefTy.getElementType())));
      else if (auto memrefTy = dyn_cast<quake::VeqType>(inTy))
        inputTys.push_back(cudaq::opt::factory::getPointerType(stlVectorType(
            IntegerType::get(ctx, /*FIXME sizeof a pointer?*/ 64))));
      else
        inputTys.push_back(inTy);
    }

    // Handle the result type. We only add a result type when there is a result
    // and it hasn't been converted to a hidden sret argument.
    if (funcTy.getNumResults() == 0 || hasSRet)
      return FunctionType::get(ctx, inputTys, {});
    assert(funcTy.getNumResults() == 1);
    return FunctionType::get(ctx, inputTys, funcTy.getResults());
  }

  FunctionType getThunkType(MLIRContext *ctx) {
    auto ptrTy = cudaq::opt::factory::getPointerType(ctx);
    return FunctionType::get(
        ctx, {ptrTy, IntegerType::get(ctx, 1)},
        {LLVM::LLVMStructType::getLiteral(
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
                      LLVM::LLVMPointerType ptrTy, Value arg) {
    // Create the i64 type
    Type i64Ty = builder.getI64Type();

    // We're given ptr<struct<...>>, get that struct type (struct<T*,T*,T*>)
    auto inpStructTy = ptrTy.getElementType().cast<LLVM::LLVMStructType>();

    // For the following GEP calls, we'll expect them to return T**
    auto ptrTtype =
        cudaq::opt::factory::getPointerType(inpStructTy.getBody()[0]);

    // Get the pointer to the pointer of the end of the array
    Value endPtr = builder.create<LLVM::GEPOp>(loc, ptrTtype, arg,
                                               SmallVector<LLVM::GEPArg>{0, 1});

    // Get the pointer to the pointer of the beginning of the array
    Value beginPtr = builder.create<LLVM::GEPOp>(
        loc, ptrTtype, arg, SmallVector<LLVM::GEPArg>{0, 0});

    // Load to a T*
    endPtr =
        builder.create<LLVM::LoadOp>(loc, inpStructTy.getBody()[1], endPtr, 8);
    beginPtr = builder.create<LLVM::LoadOp>(loc, inpStructTy.getBody()[0],
                                            beginPtr, 8);

    // Map those pointers to integers
    Value endInt = builder.create<LLVM::PtrToIntOp>(loc, i64Ty, endPtr);
    Value beginInt = builder.create<LLVM::PtrToIntOp>(loc, i64Ty, beginPtr);

    // Subtracting these will give us the size in bytes.
    return builder.create<LLVM::SubOp>(loc, endInt, beginInt);
  }

  Value copyVectorData(OpBuilder &builder, Location loc, Value stVal, Value arg,
                       Value vecToBuffer, DenseI64ArrayAttr off) {
    auto ctx = builder.getContext();
    auto falseAttr = IntegerAttr::get(IntegerType::get(ctx, 1), 0);
    auto notVolatile = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(ctx, 1), falseAttr);
    // memcpy from arg->begin to vecToBuffer, size bytes.
    auto bytes = builder.create<LLVM::ExtractValueOp>(loc, stVal, off);
    auto inStructTy = arg.getType()
                          .cast<LLVM::LLVMPointerType>()
                          .getElementType()
                          .cast<LLVM::LLVMStructType>();
    auto beginPtr = builder.create<LLVM::GEPOp>(
        loc, cudaq::opt::factory::getPointerType(inStructTy.getBody()[0]), arg,
        SmallVector<LLVM::GEPArg>{0, 0});
    auto fromBuff = builder.create<LLVM::LoadOp>(loc, beginPtr);
    auto vecFromBuff = builder.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(ctx), fromBuff);
    builder.create<func::CallOp>(
        loc, std::nullopt, llvmMemCopyIntrinsic,
        SmallVector<Value>{vecToBuffer, vecFromBuff, bytes, notVolatile});
    // Increment vecToBuffer by size bytes.
    return builder.create<LLVM::GEPOp>(loc, vecToBuffer.getType(), vecToBuffer,
                                       SmallVector<Value>{bytes});
  }

  // Create a function that takes void**, void* and returns void
  func::FuncOp genKernelArgsCreatorFunction(Location loc, OpBuilder &builder,
                                            const std::string &classNameStr,
                                            Type structPtrTy,
                                            FunctionType funcTy) {
    // Local types and values we'll need
    auto *ctx = builder.getContext();
    auto ptrType = cudaq::opt::factory::getPointerType(ctx);
    auto ptrPtrType = LLVM::LLVMPointerType::get(ptrType);
    Type i8PtrTy =
        cudaq::opt::factory::getPointerType(builder.getIntegerType(8));
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
    auto structType = structPtrTy.cast<LLVM::LLVMPointerType>()
                          .getElementType()
                          .cast<LLVM::LLVMStructType>();
    // Get the original function args
    auto kernelArgTypes = funcTy.getInputs();

    // Init the struct
    Value stVal = builder.create<LLVM::UndefOp>(loc, structType);

    // Get the variadic void* args
    auto variadicArgs = entry->getArgument(0);

    // We want to keep track of all the vector args we see
    SmallVector<std::pair<std::int64_t, Value>> vectorArgIndices;

    // Initialize the counter for extra size.
    Value zero = builder.create<LLVM::ConstantOp>(loc, i64Ty, zeroAttr);
    Value extraBytes = zero;

    // Loop over the struct elements
    for (auto structElementTypeIter : llvm::enumerate(structType.getBody())) {
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
      Value argPtrPtr =
          builder.create<LLVM::GEPOp>(loc, ptrPtrType, variadicArgs,
                                      SmallVector<LLVM::GEPArg>{(int32_t)idx});
      Value argPtr = builder.create<LLVM::LoadOp>(loc, ptrType, argPtrPtr);

      // Is this a vecType, storing it to a bool so we can check
      // in multiple parts of the following code
      auto vecType =
          dyn_cast<cudaq::cc::StdvecType>(correspondingKernelArgType);
      bool isVecType = vecType != nullptr;
      Value loadedVal;
      if (isVecType) {
        auto vecEleTy = vecType.getElementType();
        auto vecElePtrTy = cudaq::opt::factory::getPointerType(vecEleTy);
        SmallVector<Type> vecStructEleTys = {vecElePtrTy, vecElePtrTy,
                                             vecElePtrTy};
        type = cudaq::opt::factory::getPointerType(
            LLVM::LLVMStructType::getLiteral(ctx, vecStructEleTys));
        loadedVal = builder.create<LLVM::BitcastOp>(loc, type, argPtr);
      } else {
        argPtr = builder.create<LLVM::BitcastOp>(
            loc, LLVM::LLVMPointerType::get(type), argPtr);
        // cast to the struct element type, void* -> TYPE *
        loadedVal = builder.create<LLVM::LoadOp>(loc, type, argPtr);
      }

      if (isVecType) {
        // Store the index and the vec value
        vectorArgIndices.push_back(std::make_pair(idx, loadedVal));
        // compute the extra size needed for the vector data
        loadedVal = getVectorSize(
            builder, loc, type.cast<LLVM::LLVMPointerType>(), loadedVal);
        extraBytes = builder.create<LLVM::AddOp>(loc, extraBytes, loadedVal);
      }

      stVal = builder.create<LLVM::InsertValueOp>(loc, stVal, loadedVal, off);
    }

    // Compute the struct size
    auto nullSt = builder.create<LLVM::IntToPtrOp>(loc, structPtrTy, zero);
    auto computedOffset = builder.create<LLVM::GEPOp>(
        loc, structPtrTy, nullSt, SmallVector<LLVM::GEPArg>{1});
    Value structSize =
        builder.create<LLVM::PtrToIntOp>(loc, i64Ty, computedOffset);

    // If no vector args, handle this simple case and drop out
    if (vectorArgIndices.empty()) {
      Value buff = builder
                       .create<func::CallOp>(loc, i8PtrTy, "malloc",
                                             ValueRange(structSize))
                       .getResult(0);

      Value casted = builder.create<LLVM::BitcastOp>(loc, structPtrTy, buff);
      builder.create<LLVM::StoreOp>(loc, stVal, casted);
      builder.create<LLVM::StoreOp>(loc, buff, entry->getArgument(1));
      builder.create<func::ReturnOp>(loc, ValueRange{structSize});
      builder.restoreInsertionPoint(insPt);
      return argsCreatorFunc;
    }

    // Here we do have vector args
    Value extendedStructSize =
        builder.create<LLVM::AddOp>(loc, structSize, extraBytes);
    Value buff = builder
                     .create<func::CallOp>(loc, i8PtrTy, "malloc",
                                           ValueRange(extendedStructSize))
                     .getResult(0);

    auto casted = builder.create<LLVM::BitcastOp>(loc, structPtrTy, buff);
    builder.create<LLVM::StoreOp>(loc, stVal, casted);
    Value vecToBuffer = builder.create<LLVM::GEPOp>(
        loc, i8PtrTy, buff, SmallVector<Value>{structSize});

    for (auto [idx, vecVal] : vectorArgIndices) {
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      vecToBuffer =
          copyVectorData(builder, loc, stVal, vecVal, vecToBuffer, off);
    }

    builder.create<LLVM::StoreOp>(loc, buff, entry->getArgument(1));
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
    auto cast = builder.create<LLVM::BitcastOp>(loc, structPtrTy,
                                                thunkEntry->getArgument(0));
    auto isClientServer = thunkEntry->getArgument(1);
    Value val = builder.create<LLVM::LoadOp>(loc, cast);
    auto i64Ty = builder.getIntegerType(64);

    // Compute the struct size without the trailing bytes, structSize.
    auto zeroAttr = builder.getI64IntegerAttr(0);
    auto zero = builder.create<LLVM::ConstantOp>(loc, i64Ty, zeroAttr);
    auto nullSt = builder.create<LLVM::IntToPtrOp>(loc, structPtrTy, zero);
    auto computedOffset = builder.create<LLVM::GEPOp>(
        loc, structPtrTy, nullSt, SmallVector<LLVM::GEPArg>{1});
    Value structSize =
        builder.create<LLVM::PtrToIntOp>(loc, i64Ty, computedOffset);

    // Compute location of trailing bytes.
    auto i8PtrTy =
        cudaq::opt::factory::getPointerType(builder.getIntegerType(8));
    Value trailingData = builder.create<LLVM::GEPOp>(
        loc, i8PtrTy, thunkEntry->getArgument(0), structSize);

    // Unpack the arguments in the struct and build the argument list for
    // the call to the kernel code.
    SmallVector<Value> args;
    for (auto inp : llvm::enumerate(funcTy.getInputs())) {
      Type inTy = inp.value();
      std::int64_t idx = inp.index();
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      if (inTy.isa<cudaq::cc::LambdaType, LLVM::LLVMStructType>()) {
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
            builder.create<LLVM::ExtractValueOp>(loc, i64Ty, val, off);
        auto eleSizeAttr = builder.getI64IntegerAttr(eleSize);
        auto eleSizeVal =
            builder.create<LLVM::ConstantOp>(loc, i64Ty, eleSizeAttr);
        auto vecLength = builder.create<LLVM::SDivOp>(loc, vecSize, eleSizeVal);
        // The data is at trailingData and is valid for vecLength of eleTy.
        auto castData = builder.create<LLVM::BitcastOp>(
            loc, cudaq::opt::factory::getPointerType(eleTy), trailingData);
        args.push_back(builder.create<cudaq::cc::StdvecInitOp>(
            loc, stdvecTy, castData, vecLength));
        trailingData =
            builder.create<LLVM::GEPOp>(loc, i8PtrTy, trailingData, vecSize);
      } else {
        args.push_back(
            builder.create<LLVM::ExtractValueOp>(loc, inTy, val, off));
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
        auto ptrTy =
            cudaq::opt::factory::getPointerType(vecTy.getElementType());
        auto gep0 = builder.create<LLVM::GEPOp>(
            loc, cudaq::opt::factory::getPointerType(ptrTy), cast,
            SmallVector<LLVM::GEPArg>{0, off});
        auto pointer =
            builder.create<cudaq::cc::StdvecDataOp>(loc, ptrTy, callResult);
        builder.create<LLVM::StoreOp>(loc, pointer, gep0);
        auto lenTy = builder.getI64Type();
        auto gep1 = builder.create<LLVM::GEPOp>(
            loc, cudaq::opt::factory::getPointerType(lenTy), cast,
            SmallVector<LLVM::GEPArg>{0, off + 1});
        auto length =
            builder.create<cudaq::cc::StdvecSizeOp>(loc, lenTy, callResult);
        builder.create<LLVM::StoreOp>(loc, length, gep1);
        offset++;
        hasVectorResult = true;
      } else {
        auto gep = builder.create<LLVM::GEPOp>(
            loc, cudaq::opt::factory::getPointerType(res.value()), cast,
            SmallVector<LLVM::GEPArg>{0, off});
        builder.create<LLVM::StoreOp>(loc, call.getResult(res.index()), gep);
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
      auto structTy = structPtrTy.cast<LLVM::LLVMPointerType>()
                          .getElementType()
                          .cast<LLVM::LLVMStructType>();
      auto gepRes = builder.create<LLVM::GEPOp>(
          loc, cudaq::opt::factory::getPointerType(structTy.getBody()[offset]),
          cast, SmallVector<LLVM::GEPArg>{0, offset});
      auto gepRes2 = builder.create<LLVM::BitcastOp>(
          loc, cudaq::opt::factory::getPointerType(thunkTy.getResults()[0]),
          gepRes);
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
  /// helper routine.
  void genStdvecBoolFromInitList(Location loc, OpBuilder &builder, Value sret,
                                 Value data, Value size) {
    auto ptrTy = cudaq::opt::factory::getPointerType(builder.getContext());
    auto castData = builder.create<LLVM::BitcastOp>(loc, ptrTy, data);
    builder.create<LLVM::CallOp>(loc, std::nullopt, stdvecBoolCtorFromInitList,
                                 ArrayRef<Value>{sret, castData, size});
  }

  /// Generate an all new entry point body, calling launchKernel in the runtime
  /// library. Pass along the thunk, so the runtime can call the quantum
  /// circuit. These entry points are `operator()` member functions in a class,
  /// so account for the `this` argument here.
  void genNewHostEntryPoint(Location loc, OpBuilder &builder,
                            StringAttr mangledAttr, FunctionType funcTy,
                            Type structTy, LLVM::GlobalOp kernName,
                            func::FuncOp thunk) {
    auto *ctx = builder.getContext();
    auto i64Ty = builder.getIntegerType(64);
    auto zeroAttr = builder.getI64IntegerAttr(0);
    auto offset = funcTy.getNumInputs();
    auto thunkTy = getThunkType(ctx);
    auto structPtrTy = cudaq::opt::factory::getPointerType(structTy);
    auto rewriteEntry = builder.create<func::FuncOp>(
        loc, mangledAttr.getValue(), toLLVMFuncType(funcTy));
    auto insPt = builder.saveInsertionPoint();
    auto *rewriteEntryBlock = rewriteEntry.addEntryBlock();
    builder.setInsertionPointToStart(rewriteEntryBlock);
    Value stVal = builder.create<LLVM::UndefOp>(loc, structTy);

    // Process all the arguments for the original call, ignoring the `this`
    // pointer.
    auto zero = builder.create<LLVM::ConstantOp>(loc, i64Ty, zeroAttr);
    Value extraBytes = zero;
    bool hasTrailingData = false;
    for (auto inp :
         llvm::enumerate(rewriteEntryBlock->getArguments().drop_front(
             hasHiddenSRet(funcTy) ? 2 : 1))) {
      Value arg = inp.value();
      Type inTy = arg.getType();
      std::int64_t idx = inp.index();
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
      if (inTy.isa<cudaq::cc::LambdaType, LLVM::LLVMStructType>()) {
        /* do nothing */
      } else if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(inTy)) {
        // FIXME: for now assume this is a std::vector<`eleTy`>
        // FIXME: call the `size` member function. For expediency, assume this
        // is an std::vector and the size is the scaled delta between the
        // first two pointers. Use the unscaled size for now.
        auto sizeBytes = getVectorSize(builder, loc, ptrTy, arg);
        stVal = builder.create<LLVM::InsertValueOp>(loc, stVal, sizeBytes, off);
        extraBytes = builder.create<LLVM::AddOp>(loc, extraBytes, sizeBytes);
        hasTrailingData = true;
      } else {
        stVal = builder.create<LLVM::InsertValueOp>(loc, stVal, arg, off);
      }
    }

    // Compute the struct size without the trailing bytes, structSize, and with
    // the trailing bytes, extendedStructSize.
    auto nullSt = builder.create<LLVM::IntToPtrOp>(loc, structPtrTy, zero);
    auto computedOffset = builder.create<LLVM::GEPOp>(
        loc, structPtrTy, nullSt, SmallVector<LLVM::GEPArg>{1});
    Value structSize =
        builder.create<LLVM::PtrToIntOp>(loc, i64Ty, computedOffset);
    Value extendedStructSize =
        builder.create<LLVM::AddOp>(loc, structSize, extraBytes);

    // Allocate our struct to save the argument to.
    auto i8PtrTy = cudaq::opt::factory::getPointerType(builder.getI8Type());
    auto buff =
        builder.create<LLVM::AllocaOp>(loc, i8PtrTy, extendedStructSize);

    auto temp = builder.create<LLVM::BitcastOp>(loc, structPtrTy, buff);

    // Store the arguments to the argument section.
    builder.create<LLVM::StoreOp>(loc, stVal, temp);

    // Append the vector data to the end of the struct.
    if (hasTrailingData) {
      Value vecToBuffer = builder.create<LLVM::GEPOp>(
          loc, i8PtrTy, buff, SmallVector<Value>{structSize});
      for (auto inp :
           llvm::enumerate(rewriteEntryBlock->getArguments().drop_front(1))) {
        Value arg = inp.value();
        Type inTy = arg.getType();
        std::int64_t idx = inp.index();
        auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{idx});
        if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(inTy)) {
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
    auto castLoadKernName = builder.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(ctx), loadKernName);
    auto castLoadThunk = builder.create<cudaq::cc::FuncToPtrOp>(
        loc, cudaq::opt::factory::getPointerType(ctx), loadThunk);
    auto castTemp = builder.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(ctx), temp);

    auto resultOffset = [&]() -> Value {
      if (funcTy.getNumResults() == 0) {
        auto notZeroAttr = builder.getI64IntegerAttr(NoResultOffset);
        return builder.create<LLVM::ConstantOp>(loc, i64Ty, notZeroAttr);
      }
      int offset = funcTy.getNumInputs();
      auto gep = builder.create<LLVM::GEPOp>(
          loc, structPtrTy, nullSt, SmallVector<LLVM::GEPArg>{0, offset});
      return builder.create<LLVM::PtrToIntOp>(loc, i64Ty, gep);
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
        auto ptrTy =
            cudaq::opt::factory::getPointerType(vecTy.getElementType());
        auto gep0 = builder.create<LLVM::GEPOp>(
            loc, cudaq::opt::factory::getPointerType(ptrTy), temp,
            SmallVector<LLVM::GEPArg>{0, off});
        auto dataPtr = builder.create<LLVM::LoadOp>(loc, gep0);
        auto lenPtrTy =
            cudaq::opt::factory::getPointerType(builder.getI64Type());
        auto gep1 = builder.create<LLVM::GEPOp>(
            loc, lenPtrTy, temp, SmallVector<LLVM::GEPArg>{0, off + 1});
        auto vecLen = builder.create<LLVM::LoadOp>(loc, gep1);
        if (vecTy.getElementType() == builder.getI1Type())
          genStdvecBoolFromInitList(loc, builder,
                                    rewriteEntryBlock->getArguments().front(),
                                    dataPtr, vecLen);
        else
          TODO_loc(loc, "return is std::vector<T> where T != bool");
        offset++;
      } else {
        auto gep = builder.create<LLVM::GEPOp>(
            loc, cudaq::opt::factory::getPointerType(res.value()), temp,
            SmallVector<LLVM::GEPArg>{0, off});
        results.push_back(builder.create<LLVM::LoadOp>(loc, gep));
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
    builder.create<LLVM::LLVMFuncOp>(
        loc, cudaqRegisterKernelName,
        LLVM::LLVMFunctionType::get(
            cudaq::opt::factory::getVoidType(ctx),
            {cudaq::opt::factory::getPointerType(ctx)}));
    builder.create<LLVM::LLVMFuncOp>(
        loc, cudaqRegisterArgsCreator,
        LLVM::LLVMFunctionType::get(
            cudaq::opt::factory::getVoidType(ctx),
            {cudaq::opt::factory::getPointerType(ctx),
             cudaq::opt::factory::getPointerType(ctx)}));

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
      auto structPtrTy = cudaq::opt::factory::getPointerType(structTy);

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
                           kernName, thunk);

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
      auto castKernRef = builder.create<LLVM::BitcastOp>(
          loc, cudaq::opt::factory::getPointerType(ctx), kernRef);
      builder.create<LLVM::CallOp>(loc, std::nullopt, cudaqRegisterKernelName,
                                   ValueRange{castKernRef});

      // Register the argsCreator too
      auto ptrType = cudaq::opt::factory::getPointerType(ctx);
      auto ptrPtrType = LLVM::LLVMPointerType::get(ptrType);
      auto argsCreatorFuncType = FunctionType::get(
          ctx, {ptrPtrType, ptrPtrType}, {builder.getI64Type()});
      Value loadArgsCreator = builder.create<func::ConstantOp>(
          loc, argsCreatorFuncType, argsCreatorFunc.getName());
      auto castLoadArgsCreator = builder.create<cudaq::cc::FuncToPtrOp>(
          loc, cudaq::opt::factory::getPointerType(ctx), loadArgsCreator);
      builder.create<LLVM::CallOp>(
          loc, std::nullopt, cudaqRegisterArgsCreator,
          ValueRange{castKernRef, castLoadArgsCreator});

      // -------------
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

          auto castlambdaRef = builder.create<LLVM::BitcastOp>(
              loc, cudaq::opt::factory::getPointerType(ctx), lambdaRef);
          builder.create<LLVM::CallOp>(loc, std::nullopt,
                                       cudaqRegisterLambdaName,
                                       ValueRange{castlambdaRef, castKernRef});
        }
      }
      // -------------

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
