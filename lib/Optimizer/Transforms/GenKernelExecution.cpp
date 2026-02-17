/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
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
#include <cstdlib>
#include <cxxabi.h>
#include <regex>

namespace cudaq::opt {
#define GEN_PASS_DEF_GENERATEKERNELEXECUTION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "kernel-execution"

using namespace mlir;

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
      if (!(cudaq::cc::isDynamicType(ty) ||
            cudaq::opt::marshal::isStateType(ty) ||
            isa<cudaq::cc::IndirectCallableType>(ty)))
        v = cudaq::cc::LoadOp::create(builder, loc, v);
      // Python will pass a std::vector<bool> to us here. Unpack it.
      auto pear = cudaq::opt::marshal::unpackAnyStdVectorBool(
          loc, builder, module, v, ty, heapTracker);
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
      auto pear = cudaq::opt::marshal::unpackAnyStdVectorBool(
          loc, builder, module, *argIter, devTy, heapTracker);
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
        auto tmp = cudaq::cc::AllocaOp::create(builder, loc, pairTy);
        auto tmp1 = cudaq::cc::CastOp::create(
            builder, loc, cudaq::cc::PointerType::get(first.getType()), tmp);
        cudaq::cc::StoreOp::create(builder, loc, first, tmp1);
        auto tmp2 = cudaq::cc::ComputePtrOp::create(
            builder, loc, cudaq::cc::PointerType::get(second.getType()), tmp,
            ArrayRef<cudaq::cc::ComputePtrArg>{1});
        cudaq::cc::StoreOp::create(builder, loc, second, tmp2);
        auto devPtrTy = cudaq::cc::PointerType::get(devTy);
        Value devVal = cudaq::cc::CastOp::create(builder, loc, devPtrTy, tmp);
        if (!cudaq::cc::isDynamicType(devTy))
          devVal = cudaq::cc::LoadOp::create(builder, loc, devVal);
        result.emplace_back(argPos, devVal, devTy);
        continue;
      }

      // Is this a static struct passed as a byval pointer?
      if (isa<cudaq::cc::StructType>(devTy) &&
          isa<cudaq::cc::PointerType>((*argIter).getType()) &&
          !cudaq::cc::isDynamicType(devTy)) {
        Value devVal = cudaq::cc::LoadOp::create(builder, loc, *argIter);
        result.emplace_back(argPos, devVal, devTy);
        continue;
      }
      result.emplace_back(argPos, *argIter, devTy);
    }
  }
  return result;
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
    auto ptrPtrType = cudaq::opt::marshal::getPointerToPointerType(builder);
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
    auto argsCreatorFunc = func::FuncOp::create(
        builder, loc, classNameStr + ".argsCreator", funcType);
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
    auto argsArray = cudaq::cc::CastOp::create(
        builder, loc,
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(ptrI8Ty)),
        entry->getArgument(0));

    // Loop over the array and cast the void* to the host-side type.
    SmallVector<Value> pseudoArgs;
    for (auto iter : llvm::enumerate(passedHostArgTys)) {
      std::int32_t i = iter.index();
      auto parg = cudaq::cc::ComputePtrOp::create(
          builder, loc, ptrPtrType, argsArray,
          ArrayRef<cudaq::cc::ComputePtrArg>{i});
      Type ty = iter.value();
      // parg is a pointer to a pointer as it is an element of an array of
      // pointers. Always dereference the first layer here.
      Value deref = cudaq::cc::LoadOp::create(builder, loc, parg);
      if (!isa<cudaq::cc::PointerType>(ty))
        ty = cudaq::cc::PointerType::get(ty);
      pseudoArgs.push_back(cudaq::cc::CastOp::create(builder, loc, ty, deref));
    }

    // Zip the arguments with the device side argument types. Recall that some
    // of the (left-most) arguments may have been dropped on the floor.
    const bool hasDynamicSignature =
        cudaq::opt::marshal::isDynamicSignature(devKernelTy);
    Value heapTracker =
        cudaq::opt::marshal::createEmptyHeapTracker(loc, builder);
    auto zippy = zipArgumentsWithDeviceTypes</*argsAreReferences=*/true>(
        loc, builder, module, pseudoArgs, passedDevArgTys, heapTracker);
    auto sizeScratch = cudaq::cc::AllocaOp::create(builder, loc, i64Ty);
    auto messageBufferSize = [&]() -> Value {
      if (hasDynamicSignature)
        return cudaq::opt::marshal::genSizeOfDynamicMessageBuffer(
            loc, builder, module, msgStructTy, zippy, sizeScratch);
      return cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, msgStructTy);
    }();

    // Allocate the message buffer on the heap. It must outlive this call.
    auto buff = func::CallOp::create(builder, loc, ptrI8Ty, "malloc",
                                     ValueRange(messageBufferSize));
    Value rawMessageBuffer = buff.getResult(0);
    Value msgBufferPrefix =
        cudaq::cc::CastOp::create(builder, loc, structPtrTy, rawMessageBuffer);

    // Populate the message buffer with the pointer-free argument values.
    if (hasDynamicSignature) {
      auto addendumScratch = cudaq::cc::AllocaOp::create(builder, loc, ptrI8Ty);
      Value prefixSize =
          cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, msgStructTy);
      auto arrMessageBuffer = cudaq::cc::CastOp::create(
          builder, loc,
          cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty)),
          rawMessageBuffer);
      // Compute the position of the addendum.
      Value addendumPtr = cudaq::cc::ComputePtrOp::create(
          builder, loc, ptrI8Ty, arrMessageBuffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
      cudaq::opt::marshal::populateMessageBuffer(loc, builder, module,
                                                 msgBufferPrefix, zippy,
                                                 addendumPtr, addendumScratch);
    } else {
      cudaq::opt::marshal::populateMessageBuffer(loc, builder, module,
                                                 msgBufferPrefix, zippy);
    }

    cudaq::opt::marshal::maybeFreeHeapAllocations(loc, builder, heapTracker);

    // Return the message buffer and its size in bytes.
    cudaq::cc::StoreOp::create(builder, loc, rawMessageBuffer,
                               entry->getArgument(1));
    func::ReturnOp::create(builder, loc, ValueRange{messageBufferSize});

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
    auto thunkTy = cudaq::opt::marshal::getThunkType(ctx);
    auto thunk =
        func::FuncOp::create(builder, loc, classNameStr + ".thunk", thunkTy);
    OpBuilder::InsertionGuard guard(builder);
    auto *thunkEntry = thunk.addEntryBlock();
    builder.setInsertionPointToStart(thunkEntry);
    auto castOp = cudaq::cc::CastOp::create(builder, loc, structPtrTy,
                                            thunkEntry->getArgument(0));
    auto isClientServer = thunkEntry->getArgument(1);
    auto i64Ty = builder.getI64Type();

    // Compute the struct size without the trailing bytes, structSize.
    Value structSize =
        cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, structTy);

    // Compute location of trailing bytes.
    auto bufferPtrTy =
        cudaq::opt::factory::getIndexedObjectType(builder.getI8Type());
    Value extendedBuffer = cudaq::cc::CastOp::create(
        builder, loc, bufferPtrTy, thunkEntry->getArgument(0));
    auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
    Value trailingData = cudaq::cc::ComputePtrOp::create(
        builder, loc, ptrI8Ty, extendedBuffer, structSize);

    // Unpack the arguments in the struct and build the argument list for
    // the call to the kernel code.
    SmallVector<Value> args;
    const std::int32_t offset = funcTy.getNumInputs();
    if (positNullary) {
      for (auto inp : funcOp.getFunctionType().getInputs())
        args.push_back(cudaq::cc::UndefOp::create(builder, loc, inp));
    } else {
      for (auto inp : llvm::enumerate(funcTy.getInputs())) {
        auto [a, t] = cudaq::opt::marshal::processInputValue(
            loc, builder, trailingData, castOp, inp.value(), inp.index(),
            structTy);
        trailingData = t;
        args.push_back(a);
      }
    }
    auto call = cudaq::cc::NoInlineCallOp::create(
        builder, loc, funcTy.getResults(), funcOp.getName(), args, ArrayAttr(),
        ArrayAttr());
    // After the kernel call, clean up any `Array` allocations during kernel
    // executions.
    func::CallOp::create(builder, loc, TypeRange{},
                         cudaq::runtime::cleanupArrays, ValueRange{});
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
      auto mem = cudaq::cc::ComputePtrOp::create(
          builder, loc, memTy, castOp,
          SmallVector<cudaq::cc::ComputePtrArg>{offset});
      auto resPtrTy = cudaq::cc::PointerType::get(call.getResult(0).getType());
      auto castMem = cudaq::cc::CastOp::create(builder, loc, resPtrTy, mem);
      cudaq::cc::StoreOp::create(builder, loc, call.getResult(0), castMem);
      cf::CondBranchOp::create(builder, loc, isClientServer, thenBlock,
                               elseBlock);
      builder.setInsertionPointToEnd(thenBlock);
      auto resAsArg = cudaq::cc::CastOp::create(
          builder, loc, cudaq::cc::PointerType::get(thunkTy.getResults()[0]),
          mem);
      auto retOffset = cudaq::opt::marshal::genComputeReturnOffset(
          loc, builder, funcTy, structTy);
      // createDynamicResult allocates a new buffer and packs the input values
      // and the dynamic results into this single new buffer to pass back as a
      // message.
      // NB: This code only handles one dimensional vectors of static types. It
      // will have to be changed if there is a need to return recursively
      // dynamic structures, i.e., vectors of vectors.
      auto res = func::CallOp::create(
          builder, loc, thunkTy.getResults()[0], "__nvqpp_createDynamicResult",
          ValueRange{thunkEntry->getArgument(0), structSize, resAsArg,
                     retOffset});
      func::ReturnOp::create(builder, loc, res.getResult(0));
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
          auto mem = cudaq::cc::ComputePtrOp::create(
              builder, loc, memTy, castOp,
              SmallVector<cudaq::cc::ComputePtrArg>{offset + o});
          auto resTy = call.getResult(o).getType();
          auto resPtrTy = cudaq::cc::PointerType::get(resTy);
          Value castMem = mem;
          if (resPtrTy != mem.getType())
            castMem = cudaq::cc::CastOp::create(builder, loc, resPtrTy, mem);
          cudaq::cc::StoreOp::create(builder, loc, call.getResult(o), castMem);
        }
      }
    }
    // zeroDynamicResult is used by models other than client-server. It assumes
    // that no messages need to be sent and that the CPU and QPU code share a
    // memory space. Therefore, making any copies can be skipped.
    auto zeroRes =
        func::CallOp::create(builder, loc, thunkTy.getResults()[0],
                             "__nvqpp_zeroDynamicResult", ValueRange{});
    func::ReturnOp::create(builder, loc, zeroRes.getResult(0));
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
    auto thunkTy = cudaq::opt::marshal::getThunkType(ctx);
    auto structPtrTy = cudaq::cc::PointerType::get(structTy);
    const std::int32_t offset = devFuncTy.getNumInputs();

    Block *hostFuncEntryBlock = hostFunc.addEntryBlock();
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(hostFuncEntryBlock);

    SmallVector<BlockArgument> blockArgs{
        cudaq::opt::marshal::dropAnyHiddenArguments(
            hostFuncEntryBlock->getArguments(), devFuncTy, addThisPtr)};
    SmallVector<Value> blockValues(blockArgs.size());
    std::copy(blockArgs.begin(), blockArgs.end(), blockValues.begin());
    const bool hasDynamicSignature =
        cudaq::opt::marshal::isDynamicSignature(devFuncTy);
    Value heapTracker =
        cudaq::opt::marshal::createEmptyHeapTracker(loc, builder);
    auto zippy = zipArgumentsWithDeviceTypes</*argsAreReferences=*/false>(
        loc, builder, module, blockValues, devFuncTy.getInputs(), heapTracker);
    auto sizeScratch = cudaq::cc::AllocaOp::create(builder, loc, i64Ty);
    auto messageBufferSize = [&]() -> Value {
      if (hasDynamicSignature)
        return cudaq::opt::marshal::genSizeOfDynamicMessageBuffer(
            loc, builder, module, structTy, zippy, sizeScratch);
      return cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, structTy);
    }();

    Value msgBufferPrefix;
    Value castTemp;
    Value resultOffset;
    Value castLoadThunk;
    Value extendedStructSize;
    if (cudaq::opt::marshal::isCodegenPackedData(codegenKind)) {
      auto rawMessageBuffer =
          cudaq::cc::AllocaOp::create(builder, loc, i8Ty, messageBufferSize);
      msgBufferPrefix = cudaq::cc::CastOp::create(builder, loc, structPtrTy,
                                                  rawMessageBuffer);

      if (hasDynamicSignature) {
        auto addendumScratch =
            cudaq::cc::AllocaOp::create(builder, loc, ptrI8Ty);
        Value prefixSize =
            cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, structTy);
        Value addendumPtr = cudaq::cc::ComputePtrOp::create(
            builder, loc, ptrI8Ty, rawMessageBuffer,
            ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
        cudaq::opt::marshal::populateMessageBuffer(
            loc, builder, module, msgBufferPrefix, zippy, addendumPtr,
            addendumScratch);
      } else {
        cudaq::opt::marshal::populateMessageBuffer(loc, builder, module,
                                                   msgBufferPrefix, zippy);
      }

      cudaq::opt::marshal::maybeFreeHeapAllocations(loc, builder, heapTracker);
      extendedStructSize = messageBufferSize;
      Value loadThunk =
          func::ConstantOp::create(builder, loc, thunkTy, thunkFunc.getName());
      castLoadThunk =
          cudaq::cc::FuncToPtrOp::create(builder, loc, ptrI8Ty, loadThunk);
      castTemp =
          cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, msgBufferPrefix);
      resultOffset = cudaq::opt::marshal::genComputeReturnOffset(
          loc, builder, devFuncTy, structTy);
    }

    Value vecArgPtrs;
    if (cudaq::opt::marshal::isCodegenArgumentGather(codegenKind)) {
      // 1) Allocate and initialize a std::vector<void*> object.
      const unsigned count = devFuncTy.getInputs().size();
      auto stdVec = cudaq::cc::AllocaOp::create(
          builder, loc, cudaq::opt::factory::stlVectorType(ptrI8Ty));
      auto arrPtrTy = cudaq::cc::ArrayType::get(ctx, ptrI8Ty, count);
      Value buffer = cudaq::cc::AllocaOp::create(builder, loc, arrPtrTy);
      auto buffSize =
          cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, arrPtrTy);
      auto ptrPtrTy = cudaq::cc::PointerType::get(ptrI8Ty);
      auto cast1 = cudaq::cc::CastOp::create(builder, loc, ptrPtrTy, buffer);
      auto ptr3Ty = cudaq::cc::PointerType::get(ptrPtrTy);
      auto stdVec0 = cudaq::cc::CastOp::create(builder, loc, ptr3Ty, stdVec);
      cudaq::cc::StoreOp::create(builder, loc, cast1, stdVec0);
      auto cast2 = cudaq::cc::CastOp::create(builder, loc, i64Ty, buffer);
      auto endBuff = arith::AddIOp::create(builder, loc, cast2, buffSize);
      auto cast3 = cudaq::cc::CastOp::create(builder, loc, ptrPtrTy, endBuff);
      auto stdVec1 = cudaq::cc::ComputePtrOp::create(
          builder, loc, ptr3Ty, stdVec, ArrayRef<cudaq::cc::ComputePtrArg>{1});
      cudaq::cc::StoreOp::create(builder, loc, cast3, stdVec1);
      auto stdVec2 = cudaq::cc::ComputePtrOp::create(
          builder, loc, ptr3Ty, stdVec, ArrayRef<cudaq::cc::ComputePtrArg>{2});
      cudaq::cc::StoreOp::create(builder, loc, cast3, stdVec2);

      // 2) Iterate over the arguments passed in and populate the vector.
      SmallVector<BlockArgument> blockArgs{
          cudaq::opt::marshal::dropAnyHiddenArguments(
              hostFuncEntryBlock->getArguments(), devFuncTy, addThisPtr)};
      unsigned j = 0;
      for (std::int32_t i = 0, N = blockArgs.size(); i < N; ++i, ++j) {
        auto blkArg = blockArgs[i];
        auto pos = cudaq::cc::ComputePtrOp::create(
            builder, loc, ptrPtrTy, buffer,
            ArrayRef<cudaq::cc::ComputePtrArg>{i});
        if (isa<cudaq::cc::PointerType>(blkArg.getType())) {
          auto castArg =
              cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, blkArg);
          cudaq::cc::StoreOp::create(builder, loc, castArg, pos);
          continue;
        }
        Value temp;
        if (cudaq::opt::factory::isX86_64(
                hostFunc->getParentOfType<ModuleOp>()) &&
            cudaq::opt::factory::structUsesTwoArguments(
                devFuncTy.getInput(j))) {
          temp =
              cudaq::cc::AllocaOp::create(builder, loc, devFuncTy.getInput(j));
          auto part1 = cudaq::cc::CastOp::create(
              builder, loc, cudaq::cc::PointerType::get(blkArg.getType()),
              temp);
          cudaq::cc::StoreOp::create(builder, loc, blkArg, part1);
          auto blkArg2 = blockArgs[++i];
          auto cast2 = cudaq::cc::CastOp::create(
              builder, loc,
              cudaq::cc::PointerType::get(
                  cudaq::cc::ArrayType::get(blkArg2.getType())),
              temp);
          auto part2 = cudaq::cc::ComputePtrOp::create(
              builder, loc, cudaq::cc::PointerType::get(blkArg2.getType()),
              cast2, ArrayRef<cudaq::cc::ComputePtrArg>{1});
          cudaq::cc::StoreOp::create(builder, loc, blkArg2, part2);
        } else if (isa<cudaq::cc::CallableType>(blkArg.getType())) {
          // In C++, callables are already resolved. There is nothing to pass.
          temp = arith::ConstantIntOp::create(builder, loc, 0, 64);
        } else {
          temp = cudaq::cc::AllocaOp::create(builder, loc, blkArg.getType());
          cudaq::cc::StoreOp::create(builder, loc, blkArg, temp);
        }
        auto castTemp = cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, temp);
        cudaq::cc::StoreOp::create(builder, loc, castTemp, pos);
      }
      vecArgPtrs = cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, stdVec);
    }

    // Prepare to call the `launchKernel` runtime library entry point.
    Value loadKernName = LLVM::AddressOfOp::create(
        builder, loc,
        cudaq::opt::factory::getPointerType(kernelNameObj.getType()),
        kernelNameObj.getSymName());
    auto castLoadKernName =
        cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, loadKernName);

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
      auto rptr = cudaq::cc::ExtractValueOp::create(builder, loc, ptrI8Ty,
                                                    spanReturned, 0);
      launchResultToFree = rptr;
      auto rIntPtr = cudaq::cc::CastOp::create(builder, loc, i64Ty, rptr);
      auto zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
      auto cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                                       rIntPtr, zero);
      auto *currentBlock = builder.getBlock();
      auto *reg = currentBlock->getParent();
      auto *thenBlock = builder.createBlock(reg);
      auto *elseBlock = builder.createBlock(reg);
      auto *endifBlock = builder.createBlock(
          reg, reg->end(), TypeRange{ptrResTy}, SmallVector<Location>(1, loc));
      builder.setInsertionPointToEnd(currentBlock);
      cf::CondBranchOp::create(builder, loc, cmp, thenBlock, elseBlock);
      builder.setInsertionPointToEnd(thenBlock);
      // dynamic result was returned.
      // We need to free() this buffer before the end of this function.
      auto rStructPtr =
          cudaq::cc::CastOp::create(builder, loc, structPtrTy, rptr);
      Value lRes = cudaq::cc::ComputePtrOp::create(
          builder, loc, ptrResTy, rStructPtr,
          ArrayRef<cudaq::cc::ComputePtrArg>{offset});
      cf::BranchOp::create(builder, loc, endifBlock, ArrayRef<Value>{lRes});
      builder.setInsertionPointToEnd(elseBlock);
      // span was returned in the original buffer.
      Value mRes = cudaq::cc::ComputePtrOp::create(
          builder, loc, ptrResTy, msgBufferPrefix,
          ArrayRef<cudaq::cc::ComputePtrArg>{offset});
      cf::BranchOp::create(builder, loc, endifBlock, ArrayRef<Value>{mRes});
      builder.setInsertionPointToEnd(endifBlock);
      launchResult = endifBlock->getArgument(0);
    };

    // Generate the call to `launchKernel`.
    switch (codegenKind) {
    case 0: {
      assert(vecArgPtrs && castLoadThunk);
      auto launch = func::CallOp::create(
          builder, loc, cudaq::opt::factory::getDynamicBufferType(ctx),
          cudaq::runtime::launchKernelHybridFuncName,
          ArrayRef<Value>{castLoadKernName, castLoadThunk, castTemp,
                          extendedStructSize, resultOffset, vecArgPtrs});
      decodeLaunchResults(launch.getResult(0));
    } break;
    case 1: {
      assert(!vecArgPtrs && castLoadThunk);
      auto launch = func::CallOp::create(
          builder, loc, cudaq::opt::factory::getDynamicBufferType(ctx),
          cudaq::runtime::launchKernelFuncName,
          ArrayRef<Value>{castLoadKernName, castLoadThunk, castTemp,
                          extendedStructSize, resultOffset});
      decodeLaunchResults(launch.getResult(0));
    } break;
    case 2: {
      assert(vecArgPtrs && !castLoadThunk);
      func::CallOp::create(builder, loc, TypeRange{},
                           cudaq::runtime::launchKernelStreamlinedFuncName,
                           ArrayRef<Value>{castLoadKernName, vecArgPtrs});
      // For this codegen kind, we drop any results on the floor and return
      // random data in registers and/or off the stack. This maintains parity
      // with any pre-existing kernel launchers.
      SmallVector<Value> garbage;
      for (auto ty : hostFunc.getFunctionType().getResults())
        garbage.push_back(cudaq::cc::UndefOp::create(builder, loc, ty));
      func::ReturnOp::create(builder, loc, garbage);
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
        auto resPtr = cudaq::cc::ComputePtrOp::create(
            builder, loc, ptrResTy, msgBufferPrefix,
            ArrayRef<cudaq::cc::ComputePtrArg>{offset});
        Type castToTy = cudaq::cc::PointerType::get(hostFuncTy.getResult(0));
        auto castResPtr = [&]() -> Value {
          if (castToTy == ptrResTy)
            return resPtr;
          return cudaq::cc::CastOp::create(builder, loc, castToTy, resPtr);
        }();
        results.push_back(cudaq::cc::LoadOp::create(builder, loc, castResPtr));
      } else {
        // This is an sret return. Check if device is returning a span. If it
        // is, then we will need to convert it to a std::vector here. The vector
        // is constructed in-place on the sret memory block.
        Value arg0 = hostFuncEntryBlock->getArguments().front();
        if (auto spanTy =
                dyn_cast<cudaq::cc::SpanLikeType>(devFuncTy.getResult(0))) {
          auto eleTy = spanTy.getElementType();
          auto ptrTy = cudaq::cc::PointerType::get(eleTy);
          auto gep0 = cudaq::cc::ComputePtrOp::create(
              builder, loc, cudaq::cc::PointerType::get(ptrTy), launchResult,
              SmallVector<cudaq::cc::ComputePtrArg>{0});
          auto dataPtr = cudaq::cc::LoadOp::create(builder, loc, gep0);
          auto lenPtrTy = cudaq::cc::PointerType::get(i64Ty);
          auto gep1 = cudaq::cc::ComputePtrOp::create(
              builder, loc, lenPtrTy, launchResult,
              SmallVector<cudaq::cc::ComputePtrArg>{1});
          auto vecLen = cudaq::cc::LoadOp::create(builder, loc, gep1);
          if (spanTy.getElementType() == builder.getI1Type()) {
            cudaq::opt::marshal::genStdvecBoolFromInitList(loc, builder, arg0,
                                                           dataPtr, vecLen);
          } else {
            Value tSize =
                cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, eleTy);
            cudaq::opt::marshal::genStdvecTFromInitList(loc, builder, arg0,
                                                        dataPtr, tSize, vecLen);
          }
          // free(nullptr) is defined to be a nop in the standard.
          func::CallOp::create(builder, loc, TypeRange{}, "free",
                               ArrayRef<Value>{launchResultToFree});
        } else {
          // Otherwise, we can just copy the aggregate into the sret memory
          // block. Uses the size of the host function's sret pointer element
          // type for the memcpy, so the device should return an (aggregate)
          // value of suitable size.
          auto resPtr = cudaq::cc::ComputePtrOp::create(
              builder, loc, ptrResTy, msgBufferPrefix,
              ArrayRef<cudaq::cc::ComputePtrArg>{offset});
          auto castMsgBuff =
              cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, resPtr);
          Type eleTy =
              cast<cudaq::cc::PointerType>(arg0.getType()).getElementType();
          Value bytes = cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, eleTy);
          auto notVolatile = arith::ConstantIntOp::create(builder, loc, 0, 1);
          auto castArg0 =
              cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, arg0);
          func::CallOp::create(
              builder, loc, TypeRange{}, cudaq::llvmMemCopyIntrinsic,
              ValueRange{castArg0, castMsgBuff, bytes, notVolatile});
        }
      }
    }

    // Return the result (if any).
    func::ReturnOp::create(builder, loc, results);
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
    auto initFun = LLVM::LLVMFuncOp::create(
        builder, loc, classNameStr + ".kernelRegFunc",
        LLVM::LLVMFunctionType::get(cudaq::opt::factory::getVoidType(ctx), {}));
    OpBuilder::InsertionGuard guard(builder);
    auto *initFunEntry = initFun.addEntryBlock(builder);
    builder.setInsertionPointToStart(initFunEntry);
    auto kernRef = LLVM::AddressOfOp::create(
        builder, loc,
        cudaq::opt::factory::getPointerType(kernelNameObj.getType()),
        kernelNameObj.getSymName());
    auto castKernRef =
        cudaq::cc::CastOp::create(builder, loc, ptrType, kernRef);
    func::CallOp::create(builder, loc, TypeRange{},
                         cudaq::runtime::CudaqRegisterKernelName,
                         ValueRange{castKernRef});

    if (cudaq::opt::marshal::isCodegenPackedData(codegenKind)) {
      // Register the argsCreator too
      auto ptrPtrType = cudaq::cc::PointerType::get(ptrType);
      auto argsCreatorFuncType = FunctionType::get(
          ctx, {ptrPtrType, ptrPtrType}, {builder.getI64Type()});
      Value loadArgsCreator = func::ConstantOp::create(
          builder, loc, argsCreatorFuncType, argsCreatorFunc.getName());
      auto castLoadArgsCreator = cudaq::cc::FuncToPtrOp::create(
          builder, loc, ptrType, loadArgsCreator);
      func::CallOp::create(builder, loc, TypeRange{},
                           cudaq::runtime::CudaqRegisterArgsCreator,
                           ValueRange{castKernRef, castLoadArgsCreator});
    }

    // Check if this is a lambda mangled name
    auto demangledPtr = abi::__cxa_demangle(mangledName.str().c_str(), nullptr,
                                            nullptr, nullptr);
    if (demangledPtr) {
      std::string demangledName(demangledPtr);
      free(demangledPtr);
      demangledName =
          std::regex_replace(demangledName, std::regex("::operator()(.*)"), "");
      if (demangledName.find("$_") != std::string::npos) {
        auto insertPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToStart(module.getBody());

        // Create this global name, it is unique for any lambda
        // bc classNameStr contains the parentFunc + varName
        auto lambdaName = LLVM::GlobalOp::create(
            builder, loc,
            cudaq::opt::factory::getStringType(ctx, demangledName.size() + 1),
            /*isConstant=*/true, LLVM::Linkage::External,
            classNameStr + ".lambdaName",
            builder.getStringAttr(demangledName + '\0'), /*alignment=*/0);

        builder.restoreInsertionPoint(insertPoint);
        auto lambdaRef = LLVM::AddressOfOp::create(
            builder, loc,
            cudaq::opt::factory::getPointerType(lambdaName.getType()),
            lambdaName.getSymName());

        auto castLambdaRef = cudaq::cc::CastOp::create(
            builder, loc, cudaq::opt::factory::getPointerType(ctx), lambdaRef);
        auto castKernelRef = cudaq::cc::CastOp::create(
            builder, loc, cudaq::opt::factory::getPointerType(ctx),
            castKernRef);
        LLVM::CallOp::create(builder, loc, TypeRange{},
                             cudaq::runtime::CudaqRegisterLambdaName,
                             ValueRange{castLambdaRef, castKernelRef});
      }
    }

    LLVM::ReturnOp::create(builder, loc, ValueRange{});
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
    if (failed(irBuilder.loadIntrinsic(module, cudaq::llvmMemSetIntrinsic)))
      return module.emitError(std::string("could not load ") +
                              cudaq::llvmMemSetIntrinsic);
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_zeroDynamicResult")))
      return module.emitError("could not load __nvqpp_zeroDynamicResult");
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_createDynamicResult")))
      return module.emitError("could not load __nvqpp_createDynamicResult");
    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::runtime::getPauliWordSize)))
      return module.emitError(
          "could not load cudaq::pauli_word::_nvqpp_size or _nvqpp_data");
    if (failed(irBuilder.loadIntrinsic(module, cudaq::runtime::cleanupArrays)))
      return module.emitError("could not load __nvqpp_cleanup_arrays");
    return success();
  }

  /// Copy the argument attributes from \p origEntryFunc to \p runEntryKern.
  /// This assumes that \p origEntryFunc \e must have a return value which is
  /// elided in \p runEntryKern.
  static void copyArgumentAttributes(func::FuncOp origEntryFunc,
                                     func::FuncOp runEntryKern) {
    assert(runEntryKern && "must have a run entry FuncOp");
    auto ua = UnitAttr::get(runEntryKern->getContext());
    runEntryKern->setAttr("no_this", ua);
    if (!origEntryFunc)
      return;
    auto attrs = origEntryFunc.getArgAttrs();
    if (!attrs)
      return;
    auto arrAttrs = dyn_cast<ArrayAttr>(*attrs);
    if (!arrAttrs)
      return;
    const bool hasSRet =
        cudaq::opt::factory::hasHiddenSRet(origEntryFunc.getFunctionType());
    const bool hasThis = !origEntryFunc->hasAttr("no_this");
    const unsigned numHidden = cudaq::cc::numberOfHiddenArgs(hasThis, hasSRet);

    // TODO: this assumes that the layout of the argument array attrs uses
    // the 0-th position for the result attributes. We should find a more
    // robust way to make copies that doesn't rely on this assumption.
    for (unsigned i = numHidden + 1, j = 0, end = arrAttrs.size(); i < end;
         ++i, ++j)
      if (auto dict = dyn_cast_if_present<DictionaryAttr>(arrAttrs[i])) {
        SmallVector<NamedAttribute> attrCopy;
        attrCopy.append(dict.getValue().begin(), dict.getValue().end());
        runEntryKern.setArgAttrs(j, attrCopy);
      }
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = module.getContext();
    auto builder = OpBuilder::atBlockEnd(module.getBody());
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
        if (funcOp.getName().starts_with(cudaq::runtime::cudaqGenPrefixName) &&
            cudaq::opt::marshal::hasLegalType(funcOp.getFunctionType()) &&
            !funcOp.empty() && !funcOp->hasAttr(cudaq::generatorAnnotation))
          workList.push_back(funcOp);

    if (genRunStack) {
      // For each kernel on the worklist, create a duplicate entry-point kernel
      // but drop the return values on the floor.
      SmallVector<func::FuncOp> runKernels;
      for (auto epKern : workList) {
        if (epKern.getFunctionType().getResults().empty() ||
            epKern->hasAttr(cudaq::generatorAnnotation))
          continue;

        std::string runKernName = epKern.getName().str() + ".run";
        auto runKernTy =
            FunctionType::get(ctx, epKern.getFunctionType().getInputs(), {});
        auto loc = epKern.getLoc();
        {
          // Create the run kernel and drop the return result on the floor.
          auto runKern =
              func::FuncOp::create(builder, loc, runKernName, runKernTy);
          auto unitAttr = builder.getUnitAttr();
          runKern->setAttr(cudaq::entryPointAttrName, unitAttr);
          runKern->setAttr(cudaq::kernelAttrName, unitAttr);
          runKern->setAttr("no_this", unitAttr);
          SmallVector<Attribute> resultTys;
          for (auto rt : epKern.getFunctionType().getResults())
            resultTys.emplace_back(TypeAttr::get(rt));
          auto arrAttr = ArrayAttr::get(ctx, resultTys);
          runKern->setAttr(cudaq::runtime::enableCudaqRun, arrAttr);
          OpBuilder::InsertionGuard guard(builder);
          Block *entry = runKern.addEntryBlock();
          builder.setInsertionPointToStart(entry);
          auto kern = func::CallOp::create(
              builder, loc, epKern.getFunctionType().getResults(),
              epKern.getName(), entry->getArguments());
          cudaq::cc::LogOutputOp::create(builder, loc, kern.getResults());
          func::ReturnOp::create(builder, loc);
          runKernels.push_back(runKern);
        }
        {
          // Create the run kernel entry point and drop the return result on the
          // floor.

          // TODO: This design of cudaq::run translates calls of both plain old
          // functions and member functions into an autogenerated specialization
          // which appears to clang as a plain old function (the autogeneration
          // will take place here). This necessarily means that the run_entry
          // function will never have a hidden `this` pointer, regardless of
          // whether the original call was a member function or not. Obviously,
          // without the `this` pointer, a call operator cannot access any data
          // members from this instance.
          auto runKernEntryName = runKernName + ".entry";
          auto runEntryKernTy = cudaq::opt::factory::toHostSideFuncType(
              runKernTy, /*hasThisPointer=*/false, module);
          runEntryKernTy =
              FunctionType::get(ctx, runEntryKernTy.getInputs(), {});
          auto runEntryKern = func::FuncOp::create(
              builder, loc, runKernEntryName, runEntryKernTy);
          auto origEntryFunc = [&]() -> func::FuncOp {
            auto mangledNameMap = module->getAttrOfType<DictionaryAttr>(
                cudaq::runtime::mangledNameMap);
            if (!mangledNameMap)
              return {};
            auto kernName = mangledNameMap.getAs<StringAttr>(epKern.getName());
            if (!kernName)
              return {};
            return module.lookupSymbol<func::FuncOp>(kernName.getValue());
          }();
          copyArgumentAttributes(origEntryFunc, runEntryKern);
          OpBuilder::InsertionGuard guard(builder);
          Block *entry = runEntryKern.addEntryBlock();
          builder.setInsertionPointToStart(entry);
          func::ReturnOp::create(builder, loc);
          // Append this to the kernel name map.
          auto dict = module->getAttrOfType<DictionaryAttr>(
              cudaq::runtime::mangledNameMap);
          SmallVector<NamedAttribute> mapVals{dict.begin(), dict.end()};
          mapVals.emplace_back(StringAttr::get(ctx, runKernName),
                               StringAttr::get(ctx, runKernEntryName));
          module->setAttr(cudaq::runtime::mangledNameMap,
                          DictionaryAttr::get(ctx, mapVals));
        }
      }
      workList.append(runKernels.begin(), runKernels.end());
    }

    if (deferToJIT) {
      // TODO: In Python, GKE is used to generate thunks at JIT time which skip
      // some kernel arguments. It needs to be investigated why that isn't done
      // earlier when the kernel is first compiled.
      LLVM_DEBUG(llvm::dbgs() << "deferring GKE until JIT compilation\n");
    } else {
      auto mangledNameMap =
          module->getAttrOfType<DictionaryAttr>(cudaq::runtime::mangledNameMap);
      LLVM_DEBUG(llvm::dbgs()
                 << workList.size() << " kernel entry functions to process\n");

      for (auto funcOp : workList) {
        auto loc = funcOp.getLoc();
        [[maybe_unused]] auto className =
            funcOp.getName().drop_front(cudaq::runtime::cudaqGenPrefixLength);
        LLVM_DEBUG(llvm::dbgs() << "processing function " << className << '\n');
        auto classNameStr = className.str();

        // Create a constant with the name of the kernel as a C string.
        auto kernelNameObj = LLVM::GlobalOp::create(
            builder, loc,
            cudaq::opt::factory::getStringType(ctx, className.size() + 1),
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
            cudaq::opt::marshal::lookupHostEntryPointFunc(mangledName, module,
                                                          funcOp);
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
          // Autogenerate an assumed host side function signature for the
          // purpose of constructing the argsCreator function.
          hostFuncTy = cudaq::opt::factory::toHostSideFuncType(
              funcTy, hasThisPtr, module);
        }

        func::FuncOp thunk;
        func::FuncOp argsCreatorFunc;

        if (cudaq::opt::marshal::isCodegenPackedData(codegenKind)) {
          auto thunkStructTy = structTy;
          auto thunkFuncTy = funcTy;
          if (positNullary) {
            // The compiler posits that the entry-point kernel is nullary (no
            // arguments) regardless of the signature. This is the case when it
            // is known that all the arguments are (or will be) synthesized and
            // the kernel is accordingly specialized in place.
            thunkFuncTy =
                FunctionType::get(ctx, ArrayRef<Type>{}, funcTy.getResults());
            thunkStructTy =
                cudaq::opt::factory::buildInvokeStructType(thunkFuncTy);
          }
          // Generate the function that computes the return offset.
          cudaq::opt::marshal::genReturnOffsetFunction(
              loc, builder, thunkFuncTy, thunkStructTy, classNameStr);

          // Generate thunk, `<kernel>.thunk`, to call back to the MLIR code.
          thunk = genThunkFunction(loc, builder, classNameStr, thunkStructTy,
                                   thunkFuncTy, funcOp);

          // Generate the argsCreator function used by synthesis.
          if (startingArgIdx == 0) {
            argsCreatorFunc = genKernelArgsCreatorFunction(
                loc, builder, module, funcTy, structTy, classNameStr,
                hostFuncTy, hasThisPtr);
          } else {
            // We are operating in a very special case where we want the
            // argsCreator function to ignore the first `startingArgIdx`
            // arguments. In this situation, the argsCreator function will not
            // be compatible with the other helper functions created in this
            // pass, so it is assumed that the caller is OK with that.
            auto structTy_argsCreator =
                cudaq::opt::factory::buildInvokeStructType(funcTy,
                                                           startingArgIdx);
            argsCreatorFunc = genKernelArgsCreatorFunction(
                loc, builder, module, funcTy, structTy_argsCreator,
                classNameStr, hostFuncTy, hasThisPtr);
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
    }
    out.keep();
  }
};
} // namespace
