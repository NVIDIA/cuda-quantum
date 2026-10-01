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
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_DISTRIBUTEDDEVICECALL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "distributed-device-call"

using namespace mlir;

namespace {

// The `llvm.sret`/`llvm.byval`-style attributes carry a Type payload that
// LLVM IR requires to be an LLVM-dialect type. func::FuncOp's own conversion
// to llvm.func type-converts such payloads automatically, but func::CallOp's
// conversion to llvm.call copies arg_attrs verbatim without doing so -
// setting a raw CC-dialect type (e.g. !cc.struct<...>) on a call site's
// arg_attrs crashes translateModuleToLLVMIR later, once it tries to
// translate that CC type as if it were an LLVM one. This is a small,
// local, non-recursive-into-CC-specific-types replacement for that missing
// conversion, sufficient for the plain pointer/integer/float/struct shapes
// that a host-side sret element type (already run through
// convertToHostSideType) is built from.
static Type ccTypeToLLVMType(Type ty) {
  auto *ctx = ty.getContext();
  if (isa<cudaq::cc::PointerType>(ty))
    return LLVM::LLVMPointerType::get(ctx);
  if (auto strTy = dyn_cast<cudaq::cc::StructType>(ty)) {
    SmallVector<Type> members;
    for (auto mem : strTy.getMembers())
      members.push_back(ccTypeToLLVMType(mem));
    return LLVM::LLVMStructType::getLiteral(ctx, members, strTy.getPacked());
  }
  if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(ty))
    return LLVM::LLVMArrayType::get(ccTypeToLLVMType(arrTy.getElementType()),
                                    arrTy.getSize());
  // Integer/float types are already the same in both type systems.
  return ty;
}

// On x86_64, a static (trivially copyable) struct that is too large to travel
// in registers is passed as a pointer to a copy on the stack, which the callee
// accesses with the `byval` attribute. Without that attribute at both the
// declaration and the call site, the pointer is passed in a register and the
// callee reads whatever happens to be on its stack instead. Mirrors the
// handling in ASTBridge.cpp for kernel host entry points.
//
// Two kinds of struct are passed by a plain pointer instead, without `byval`:
// dynamic structs (those with vectors or strings), and structs without a size.
// The frontend leaves the size off a struct whose host layout it does not know,
// such as a non-trivially copyable class like std::tuple, which the host
// passes by invisible reference.
static bool isByValStructArg(Type devArgTy, Type abiArgTy, ModuleOp module) {
  return cudaq::opt::factory::isX86_64(module) &&
         isa<cudaq::cc::StructType>(devArgTy) &&
         cast<cudaq::cc::StructType>(devArgTy).getBitSize() != 0 &&
         !cudaq::cc::isDynamicType(devArgTy) &&
         isa<cudaq::cc::PointerType>(abiArgTy);
}

// Returns the position in the ABI-converted signature of each argument that is
// passed `byval`. A small struct may occupy 0 or 2 slots, not 1, so the
// positions drift from the device function's argument positions.
static SmallVector<unsigned> getByValArgPositions(FunctionType devFuncTy,
                                                  FunctionType abiFuncTy,
                                                  ModuleOp module) {
  SmallVector<unsigned> positions;
  unsigned pos = cudaq::opt::factory::hasHiddenSRet(devFuncTy) ? 1 : 0;
  for (Type devArgTy : devFuncTy.getInputs()) {
    if (pos < abiFuncTy.getNumInputs() &&
        isByValStructArg(devArgTy, abiFuncTy.getInput(pos), module)) {
      positions.push_back(pos);
      ++pos;
      continue;
    }
    auto strTy = dyn_cast<cudaq::cc::StructType>(devArgTy);
    if (strTy && cudaq::opt::factory::isX86_64(module) &&
        cudaq::opt::factory::structUsesTwoArguments(strTy))
      ++pos;
    ++pos;
  }
  return positions;
}

// Rewrites the signature of a device function marked with the device-call
// attribute so it matches the host-side ABI, since the generalized lowering
// calls it (via the unmarshal function) from host code.
class DistributedFuncPat : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    if (!func->hasAttr(cudaq::deviceCallAttrName))
      return failure();
    FunctionType devFuncTy = func.getFunctionType();
    auto module = func->getParentOfType<ModuleOp>();
    FunctionType newDevFuncTy = cudaq::opt::factory::toHostSideFuncType(
        devFuncTy, /*addThisPtr=*/false, module);
    // toHostSideFuncType is not idempotent: applying it a second time to its
    // own output can further rewrite already-ABI-converted types. The greedy
    // pattern driver revisits this op after modifyOpInPlace below, so without
    // this attribute this pattern would fire again and settle on that second,
    // over-converted type. Clearing the marker makes the rewrite one-shot
    // instead.
    rewriter.modifyOpInPlace(func, [&]() {
      func.setFunctionType(newDevFuncTy);
      func->removeAttr(cudaq::deviceCallAttrName);
      // When this callback returns a dynamic type (e.g. std::vector<T>), the
      // ABI-converted signature above prepends a hidden sret pointer as
      // argument 0. That alone is not enough: on AAPCS64, the sret pointer
      // is passed in a dedicated register (X8), separate from the normal
      // argument registers - but only if the call/declaration actually
      // marks that parameter with the `sret` attribute. Without it, LLVM's
      // AArch64 lowering treats it as an ordinary pointer argument in X0,
      // silently shifting every other argument (including the real
      // first argument) into the wrong register. (X86_64 tolerates this
      // omission because its ABI folds sret into the normal argument
      // sequence, which is why this was never caught there.) Mirror
      // ASTBridge.cpp's identical handling for kernel host entry points.
      if (cudaq::opt::factory::hasHiddenSRet(devFuncTy)) {
        if (auto ptrTy =
                dyn_cast<cudaq::cc::PointerType>(newDevFuncTy.getInput(0))) {
          auto eleTy = ptrTy.getElementType();
          if (isa<cudaq::cc::StructType>(eleTy))
            func.setArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName(),
                            TypeAttr::get(eleTy));
        }
      }
      // Large static structs are passed byval on x86_64.
      for (unsigned pos : getByValArgPositions(devFuncTy, newDevFuncTy, module))
        func.setArgAttr(pos, LLVM::LLVMDialect::getByValAttrName(),
                        TypeAttr::get(cast<cudaq::cc::PointerType>(
                                          newDevFuncTy.getInput(pos))
                                          .getElementType()));
    });
    return success();
  }
};

class ResolveDevicePtrOpPat
    : public OpRewritePattern<cudaq::cc::ResolveDevicePtrOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::ResolveDevicePtrOp resolve,
                                PatternRewriter &rewriter) const override {
    auto loc = resolve.getLoc();
    auto call = func::CallOp::create(
        rewriter, loc,
        TypeRange{cudaq::cc::PointerType::get(rewriter.getI8Type())},
        cudaq::runtime::extractDevPtr, ValueRange{resolve.getDevicePtr()});
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(
        resolve, resolve.getResult().getType(), call.getResult(0));
    return success();
  }
};

// The layout of a dynamic result that travels as a message. A result is
// encoded exactly like a single argument of a kernel launch: a buffer type
// whose members are the pointer-free (size-slot) encodings, followed by the
// trailing data that holds the contents of vectors and strings.
static cudaq::cc::StructType getResultMessageType(Type resTy) {
  return cudaq::opt::factory::buildInvokeStructType(
      FunctionType::get(resTy.getContext(), TypeRange{resTy}, TypeRange{}), 0);
}

// This is a kernel launch in reverse. The callback, running on the host, has
// produced its dynamic result as a real host value (\p hostResult points to
// it). Encode that value into a new pointer-free message on the heap, release
// the host value (it is not needed again), and return the message and its size
// in bytes as the thunk's result. The caller takes ownership of the message.
static Value packDynamicResultMessage(Location loc, PatternRewriter &rewriter,
                                      ModuleOp module, Type resTy,
                                      Value hostResult, Type thunkResultTy) {
  auto i64Ty = rewriter.getI64Type();
  auto ptrI8Ty = cudaq::cc::PointerType::get(rewriter.getI8Type());
  auto msgTy = getResultMessageType(resTy);
  Value heapTracker =
      cudaq::opt::marshal::createEmptyHeapTracker(loc, rewriter);
  // std::vector<bool> is not like any other vector. Unpack it, as the launch
  // side does.
  Value packArg = cudaq::opt::marshal::unpackAnySequenceBool(
                      loc, rewriter, module, hostResult, resTy, heapTracker)
                      .first;
  SmallVector<std::tuple<unsigned, Value, Type>> zippy;
  zippy.emplace_back(0, packArg, resTy);
  Value sizeScratch = cudaq::cc::AllocaOp::create(rewriter, loc, i64Ty);
  Value size = cudaq::opt::marshal::genSizeOfDynamicMessageBuffer(
      loc, rewriter, module, msgTy, zippy, sizeScratch);
  Value raw =
      func::CallOp::create(rewriter, loc, ptrI8Ty, "malloc", ValueRange{size})
          .getResult(0);
  Value typed = cudaq::cc::CastOp::create(
      rewriter, loc, cudaq::cc::PointerType::get(msgTy), raw);
  Value prefixSize = cudaq::cc::SizeOfOp::create(rewriter, loc, i64Ty, msgTy);
  auto rawArr = cudaq::cc::CastOp::create(
      rewriter, loc,
      cudaq::cc::PointerType::get(
          cudaq::cc::ArrayType::get(rewriter.getI8Type())),
      raw);
  Value addendum = cudaq::cc::ComputePtrOp::create(
      rewriter, loc, ptrI8Ty, rawArr,
      ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
  Value addendumScratch = cudaq::cc::AllocaOp::create(rewriter, loc, ptrI8Ty);
  cudaq::opt::marshal::populateMessageBuffer(loc, rewriter, module, typed,
                                             zippy, addendum, addendumScratch);
  cudaq::opt::marshal::maybeFreeHeapAllocations(loc, rewriter, heapTracker);
  // The message is complete and self-contained. Release the host value.
  cudaq::opt::marshal::destroyHostValue(loc, rewriter, module, resTy,
                                        hostResult);
  Value result = cudaq::cc::UndefOp::create(rewriter, loc, thunkResultTy);
  result = cudaq::cc::InsertValueOp::create(rewriter, loc, thunkResultTy,
                                            result, raw, 0);
  return cudaq::cc::InsertValueOp::create(rewriter, loc, thunkResultTy, result,
                                          size, 1);
}

// Generalized, distributed-memory reference lowering. Here we rewrite
// `device_call` to a call into an autogenerated marshal function, which packs
// the arguments into the same pointer-free buffer format used by a
// kernel-launch, then dispatches through the runtime's `callDeviceCallback`
// hook. On the callee side, an autogenerated unmarshal function unpacks the
// buffer and calls the real device function. This works for any number of
// device functions and any valid combination of supported argument/result types
// (arithmetic types, std::vector, struct, std::tuple, and recursive vector
// definitions), since it reuses the same Marshal.h helpers as kernel launch.
class DistributedDeviceCallPat
    : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto loc = devcall.getLoc();
    // These are views into devcall's operands. They must be captured here,
    // before devcall is erased below, since devcall itself cannot be safely
    // queried again afterwards.
    SmallVector<Value> numBlocksVals(devcall.getNumBlocks().begin(),
                                     devcall.getNumBlocks().end());
    SmallVector<Value> numThreadsPerBlockVals(
        devcall.getNumThreadsPerBlock().begin(),
        devcall.getNumThreadsPerBlock().end());
    Value deviceOperand = devcall.getDevice();
    auto module = devcall->getParentOfType<ModuleOp>();
    auto devFuncName = devcall.getCallee();
    // DeviceCallOp::verifySymbolUses already resolves `callee` to a
    // func::FuncOp with this exact lookup and rejects the op otherwise, so
    // by the time a verified module reaches this pattern, the lookup below
    // cannot fail.
    auto devFunc = module.lookupSymbol<func::FuncOp>(devFuncName);

    // Is there already a marshaling function for capturing argument values for
    // this devFuncName?
    std::string marshalName = "marshal." + devFuncName.str();
    auto [marshalFunc, alreadyAdded] = cudaq::opt::factory::getOrAddFunc(
        loc, marshalName, devFunc.getFunctionType(), module);
    rewriter.replaceOpWithNewOp<func::CallOp>(devcall, devcall.getResultTypes(),
                                              marshalName, devcall.getArgs());

    if (alreadyAdded) {
      // This may happen if another kernel called this same callback.
      LLVM_DEBUG(llvm::dbgs() << "marshal function " << marshalName
                              << " already in module\n");
      return success();
    }

    // Create a constant with the name of the callback func as a C string.
    auto callbackNameObj = [&]() -> mlir::LLVM::GlobalOp {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      return LLVM::GlobalOp::create(
          rewriter, loc,
          cudaq::opt::factory::getStringType(ctx, devFuncName.size() + 1),
          /*isConstant=*/true, LLVM::Linkage::External,
          devFuncName.str() + ".callbackName",
          rewriter.getStringAttr(devFuncName.str() + '\0'), /*alignment=*/0);
    }();

    // The device call dispatcher will then call the `unmarshalFunc`, which is
    // added here. In a practical implementation, this code would be compiled
    // and linked into the host side of the application explicitly.
    std::string unmarshalName = "unmarshal." + devFuncName.str();
    auto unmarshalTy = cudaq::opt::marshal::getThunkType(ctx);
    auto [unmarshalFunc, alreadyAdded2] = cudaq::opt::factory::getOrAddFunc(
        loc, unmarshalName, unmarshalTy, module);
    if (alreadyAdded2) {
      unmarshalFunc.emitOpError("unmarshal function must not be present");
      return success();
    }

    // Create a new struct type to pass arguments and results.
    auto structTy = cudaq::opt::factory::buildInvokeStructType(
        devFunc.getFunctionType(), 0);

    // The unmarshaling code is autogenerated into `unmarshalFunc`.
    genNewUnmarshalFunc(loc, unmarshalFunc, rewriter, devFunc, structTy);

    // Autogenerate the code to marshal the arguments here in the function
    // `marshalFunc`.
    SmallVector<std::optional<std::uint64_t>> vecNumBlocks;
    for (auto v : numBlocksVals)
      vecNumBlocks.push_back(cudaq::opt::factory::maybeValueOfIntConstant(v));
    SmallVector<std::optional<std::uint64_t>> vecNumThreadsPerBlock;
    for (auto v : numThreadsPerBlockVals)
      vecNumThreadsPerBlock.push_back(
          cudaq::opt::factory::maybeValueOfIntConstant(v));
    std::optional<std::uint64_t> deviceId =
        deviceOperand
            ? cudaq::opt::factory::maybeValueOfIntConstant(deviceOperand)
            : std::nullopt;
    genNewMarshalFunc(loc, marshalFunc, rewriter, deviceId, callbackNameObj,
                      unmarshalFunc, structTy, devFunc, module, vecNumBlocks,
                      vecNumThreadsPerBlock);

    // Finally, create the registration code for this particular callback.
    genRegistrationHook(loc, devFuncName, rewriter, callbackNameObj,
                        unmarshalFunc, devFunc, module);

    return success();
  }

  static void genNewMarshalFunc(
      Location loc, func::FuncOp marshalFunc, PatternRewriter &rewriter,
      std::optional<std::uint64_t> deviceId, LLVM::GlobalOp callbackNameObj,
      func::FuncOp unmarshalFunc, cudaq::cc::StructType bufferTy,
      func::FuncOp devFunc, ModuleOp module,
      ArrayRef<std::optional<std::uint64_t>> vecNumBlocks,
      ArrayRef<std::optional<std::uint64_t>> vecNumThreadsPerBlock) {
    auto i64Ty = rewriter.getI64Type();
    Block *entryBlock = marshalFunc.addEntryBlock();
    auto ptrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);

    SmallVector<Value> dispatchArgs;
    // Arg 1: the device id. `device_call`'s device operand is optional and,
    // like numBlocks/numThreadsPerBlock, must be a compile-time constant
    // here since marshalFunc is a single, shared, freestanding function per
    // callback (it cannot reference a value from the original call site's
    // region). Default to 0 when not specified or not a constant.
    Value devId =
        arith::ConstantIntOp::create(rewriter, loc, deviceId.value_or(0), 64);
    dispatchArgs.push_back(devId);

    // Arg 2: the callback name.
    Value callbackNameVal = cudaq::cc::AddressOfOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(callbackNameObj.getType()),
        callbackNameObj.getSymName());
    dispatchArgs.push_back(
        cudaq::cc::CastOp::create(rewriter, loc, ptrTy, callbackNameVal));

    // Arg 3: pointer to the unmarshal func
    auto unmar =
        func::ConstantOp::create(rewriter, loc, unmarshalFunc.getFunctionType(),
                                 unmarshalFunc.getName());
    Value unmarPtr =
        cudaq::cc::FuncToPtrOp::create(rewriter, loc, ptrTy, unmar);
    dispatchArgs.push_back(unmarPtr);

    // Arg 4: pointer to the argument buffer
    auto devFuncTy = devFunc.getFunctionType();
    const bool hasDynamicSignature =
        cudaq::opt::marshal::isDynamicSignature(devFuncTy);
    auto sizeScratch = cudaq::cc::AllocaOp::create(rewriter, loc, i64Ty);
    SmallVector<std::tuple<unsigned, Value, Type>> zippedArgs;
    for (auto v : llvm::enumerate(entryBlock->getArguments()))
      zippedArgs.emplace_back(v.index(), v.value(), v.value().getType());
    auto bufferSize = [&]() -> Value {
      if (hasDynamicSignature)
        return cudaq::opt::marshal::genSizeOfDynamicCallbackBuffer(
            loc, rewriter, module, bufferTy, zippedArgs, sizeScratch);
      return cudaq::cc::SizeOfOp::create(rewriter, loc, i64Ty, bufferTy);
    }();
    auto i8Ty = rewriter.getI8Type();
    Value rawBuffer =
        cudaq::cc::AllocaOp::create(rewriter, loc, i8Ty, bufferSize);
    Value typedBuffer = cudaq::cc::CastOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(bufferTy), rawBuffer);
    if (hasDynamicSignature) {
      auto addendumScratch = cudaq::cc::AllocaOp::create(rewriter, loc, ptrTy);
      Value prefixSize =
          cudaq::cc::SizeOfOp::create(rewriter, loc, i64Ty, bufferTy);
      Value addendumPtr = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, ptrTy, rawBuffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
      cudaq::opt::marshal::populateCallbackBuffer(loc, rewriter, module,
                                                  typedBuffer, zippedArgs,
                                                  addendumPtr, addendumScratch);

    } else {
      cudaq::opt::marshal::populateCallbackBuffer(loc, rewriter, module,
                                                  typedBuffer, zippedArgs);
    }
    dispatchArgs.push_back(cudaq::cc::CastOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(rewriter.getI8Type()),
        rawBuffer));

    // Arg 5: argument buffer size
    dispatchArgs.push_back(bufferSize);

    // Arg 6: offset of the return value in the buffer
    Value returnOffset = cudaq::opt::marshal::genComputeReturnOffset(
        loc, rewriter, devFuncTy, bufferTy);
    dispatchArgs.push_back(returnOffset);

    // Add the block and grid size if available.
    // FIXME: This is pushing 1 dimension only. (The template only supports 1.)
    auto getVal0 =
        [&](ArrayRef<std::optional<std::uint64_t>> vec) -> std::uint64_t {
      if (!vec.empty() && vec[0].has_value())
        return *vec[0];
      return 0;
    };
    dispatchArgs.push_back(
        arith::ConstantIntOp::create(rewriter, loc, getVal0(vecNumBlocks), 64));
    dispatchArgs.push_back(arith::ConstantIntOp::create(
        rewriter, loc, getVal0(vecNumThreadsPerBlock), 64));

    auto spanTy = cudaq::cc::StructType::get(rewriter.getContext(),
                                             ArrayRef<Type>{ptrTy, i64Ty});
    auto callback =
        func::CallOp::create(rewriter, loc, spanTy,
                             cudaq::runtime::callDeviceCallback, dispatchArgs);
    auto resTys = marshalFunc.getFunctionType().getResults();
    marshalFunc.setPublic();
    if (resTys.empty()) {
      // This device call returns `void`, so we're done.
      func::ReturnOp::create(rewriter, loc);
      return;
    }

    assert(resTys.size() == 1);
    Type resTy = resTys.front();
    std::int32_t numInputs = devFuncTy.getNumInputs();
    if (cudaq::cc::isDynamicType(resTy)) {
      // As on the launch side, the callee returns a message buffer and its size
      // in bytes if, and only if, it does not share an address space with this
      // code. In that case the result was packed, without pointers, into the
      // message and it must be unpacked here.
      Value callbackResult = callback.getResult(0);
      Value msgPtr = cudaq::cc::ExtractValueOp::create(rewriter, loc, ptrTy,
                                                       callbackResult, 0);
      Value msgSize = cudaq::cc::ExtractValueOp::create(rewriter, loc, i64Ty,
                                                        callbackResult, 1);
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      Value hasMessage = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::ne, msgSize, zero);
      Block *messageBlock = rewriter.createBlock(&marshalFunc.getBody());
      Block *sharedBlock = rewriter.createBlock(&marshalFunc.getBody());
      rewriter.setInsertionPointToEnd(&marshalFunc.getBody().front());
      cf::CondBranchOp::create(rewriter, loc, hasMessage, messageBlock,
                               sharedBlock);

      // Unpack the message, which has the same format as a single argument of
      // a kernel launch.
      // TODO: The spans built here refer to the message's own storage. They
      // must eventually be copied to the caller's stack, and the message freed
      // (by this side, which owns it), as a kernel launch does with a message
      // it gets back.
      rewriter.setInsertionPointToEnd(messageBlock);
      auto msgTy = getResultMessageType(resTy);
      Value typedMsg = cudaq::cc::CastOp::create(
          rewriter, loc, cudaq::cc::PointerType::get(msgTy), msgPtr);
      Value prefixSize =
          cudaq::cc::SizeOfOp::create(rewriter, loc, i64Ty, msgTy);
      auto rawMsg = cudaq::cc::CastOp::create(
          rewriter, loc,
          cudaq::cc::PointerType::get(
              cudaq::cc::ArrayType::get(rewriter.getI8Type())),
          msgPtr);
      Value trailingData = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, ptrTy, rawMsg,
          ArrayRef<cudaq::cc::ComputePtrArg>{prefixSize});
      auto [unpacked, unusedTrailing] = cudaq::opt::marshal::processInputValue(
          loc, rewriter, module, trailingData, typedMsg, resTy, 0, msgTy);
      func::ReturnOp::create(rewriter, loc, unpacked);

      // Reference implementation: the unmarshal function (running synchronously
      // within the call above) already wrote the dynamic-output-slot-shaped
      // value for the result directly into the shared buffer. Read it back and
      // reconstruct the real result value from it.
      rewriter.setInsertionPointToEnd(sharedBlock);
      Type slotTy = bufferTy.getMember(numInputs);
      auto outputPtr = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, cudaq::cc::PointerType::get(slotTy), typedBuffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
      Value slotVal = cudaq::cc::LoadOp::create(rewriter, loc, outputPtr);
      Value resVal =
          cudaq::opt::marshal::bufferSlotToValue(loc, rewriter, resTy, slotVal);
      func::ReturnOp::create(rewriter, loc, resVal);
      return;
    }
    // The buffer's member type may be a structurally-equivalent but
    // anonymized version of resTy, so a plain compute_ptr using resTy
    // directly can fail to verify. Compute the pointer using the buffer's
    // own member type, then cast the pointer if needed.
    Type bufMemberTy = bufferTy.getMember(numInputs);
    auto outputPtr = cudaq::cc::ComputePtrOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(bufMemberTy), typedBuffer,
        ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
    Value castOutputPtr = outputPtr;
    if (bufMemberTy != resTy)
      castOutputPtr = cudaq::cc::CastOp::create(
          rewriter, loc, cudaq::cc::PointerType::get(resTy), outputPtr);
    Value resVal = cudaq::cc::LoadOp::create(rewriter, loc, castOutputPtr);
    func::ReturnOp::create(rewriter, loc, resVal);
  }

  static void genNewUnmarshalFunc(Location loc, func::FuncOp unmarshalFunc,
                                  PatternRewriter &rewriter,
                                  func::FuncOp devFunc,
                                  cudaq::cc::StructType bufferTy) {
    Block *entryBlock = unmarshalFunc.addEntryBlock();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(entryBlock);
    auto *ctx = rewriter.getContext();

    // Unmarshal arguments from the buffer and call the device function.
    auto i64Ty = rewriter.getI64Type();
    Value bufferSize =
        cudaq::cc::SizeOfOp::create(rewriter, loc, i64Ty, bufferTy);
    auto ptrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    auto ptrArrTy = cudaq::cc::PointerType::get(
        cudaq::cc::ArrayType::get(rewriter.getI8Type()));
    auto rawBuffer = cudaq::cc::CastOp::create(rewriter, loc, ptrArrTy,
                                               entryBlock->getArgument(0));
    Value trailingData = cudaq::cc::ComputePtrOp::create(
        rewriter, loc, ptrTy, rawBuffer,
        ArrayRef<cudaq::cc::ComputePtrArg>{bufferSize});
    auto ptrBuffTy = cudaq::cc::PointerType::get(bufferTy);
    auto argsBuffer = cudaq::cc::CastOp::create(rewriter, loc, ptrBuffTy,
                                                entryBlock->getArgument(0));
    FunctionType devFuncTy = devFunc.getFunctionType();
    SmallVector<Value> args;
    auto module = devFunc->getParentOfType<ModuleOp>();
    auto newDevFuncTy = cudaq::opt::factory::toHostSideFuncType(
        devFuncTy, /*addThisPtr=*/false, module);

    // If the device function returns a dynamically sized value, or a static
    // value too large to be returned in registers, the real host ABI passes it
    // back via a hidden sret pointer argument that is prepended before the
    // other arguments.
    Value sretVar;
    if (devFuncTy.getNumResults() == 1 &&
        (cudaq::cc::isDynamicType(devFuncTy.getResult(0)) ||
         cudaq::opt::factory::hasHiddenSRet(devFuncTy))) {
      Type sretHostTy = cudaq::opt::factory::convertToHostSideType(
          cudaq::opt::factory::getSRetElementType(devFuncTy, module), module);
      sretVar = cudaq::cc::AllocaOp::create(rewriter, loc, sretHostTy);
      args.push_back(sretVar);
    }

    // `offset` is the drift between an argument's position in the device
    // function's signature and its (first) position in the ABI-converted one:
    // the hidden sret pointer shifts every argument by one, and a struct passed
    // as two separate register arguments shifts the arguments after it by one.
    std::size_t offset = sretVar ? 1 : 0;
    SmallVector<Value> stringArgs;
    for (auto iter : llvm::enumerate(devFuncTy.getInputs())) {
      auto [a, t] = cudaq::opt::marshal::processCallbackInputValue(
          loc, rewriter, module, trailingData, argsBuffer, iter.value(),
          iter.index(), bufferTy);
      // Save any strings, as we need to deconstruct them after the call.
      if (isa<cudaq::cc::CharspanType>(iter.value()))
        stringArgs.push_back(a);
      trailingData = t;
      if (auto strTy = dyn_cast<cudaq::cc::StructType>(iter.value())) {
        if (cudaq::opt::factory::isX86_64(module) &&
            cudaq::opt::factory::structUsesTwoArguments(strTy)) {
          auto i = iter.index() + offset;
          SmallVector<Type> inputs{newDevFuncTy.getInputs().begin(),
                                   newDevFuncTy.getInputs().end()};
          SmallVector<Type> pair{inputs[i], inputs[i + 1]};
          ++offset;
          auto dubTy = cudaq::cc::PointerType::get(
              cudaq::cc::StructType::get(ctx, pair));
          auto dubPtr = cudaq::cc::CastOp::create(rewriter, loc, dubTy, a);
          auto ptr0 = cudaq::cc::ComputePtrOp::create(
              rewriter, loc, cudaq::cc::PointerType::get(inputs[i]), dubPtr,
              ArrayRef<cudaq::cc::ComputePtrArg>{0});
          args.push_back(cudaq::cc::LoadOp::create(rewriter, loc, ptr0));
          auto ptr1 = cudaq::cc::ComputePtrOp::create(
              rewriter, loc, cudaq::cc::PointerType::get(inputs[i + 1]), dubPtr,
              ArrayRef<cudaq::cc::ComputePtrArg>{1});
          args.push_back(cudaq::cc::LoadOp::create(rewriter, loc, ptr1));
          continue;
        }
        // Some small (<= 128-bit), non-empty structs are passed as a single
        // packed register value (a scalar or a fixed-size array) rather than
        // by pointer, per toHostSideFuncType's ABI classification (e.g.
        // AArch64 packs such a struct into one i64 or [2 x T]; X86_64 does
        // the same when the struct fits a single register, i.e. the
        // structUsesTwoArguments case above does not apply). `a` above is
        // always a raw pointer to the struct's bytes in the comm buffer;
        // when the ABI wants a packed value instead of that pointer here,
        // reinterpret and load it so the call operand matches the callee's
        // actual (ABI-converted) argument type instead of silently
        // desyncing from it.
        auto i = iter.index() + offset;
        Type abiTy = newDevFuncTy.getInputs()[i];
        if (!strTy.isEmpty() && !isa<cudaq::cc::PointerType>(abiTy)) {
          auto abiPtrTy = cudaq::cc::PointerType::get(abiTy);
          auto abiPtr = cudaq::cc::CastOp::create(rewriter, loc, abiPtrTy, a);
          args.push_back(cudaq::cc::LoadOp::create(rewriter, loc, abiPtr));
          continue;
        }
      }
      args.push_back(a);
    }

    auto callDevFunc = func::CallOp::create(
        rewriter, loc, newDevFuncTy.getResults(), devFunc.getName(), args);
    // func.call's operand attributes are independent of the callee
    // declaration's arg attributes (LLVM::CallOp has its own separate
    // arg_attrs, which MLIR's func-to-llvm lowering does not populate from
    // the callee automatically): setting `llvm.sret` only on devFunc's
    // declaration above is not sufficient. AArch64's calling-convention
    // lowering keys off the call instruction's own attribute list to decide
    // whether the first argument routes through the dedicated indirect-result
    // register (X8) or an ordinary argument register (X0) - the
    // declaration's attribute is not consulted for that decision, at least
    // not reliably. Mirror the same attribute onto the call site.
    SmallVector<Attribute> argAttrs(args.size(), DictionaryAttr::get(ctx));
    bool hasArgAttrs = false;
    auto setCallArgAttr = [&](unsigned pos, StringRef name, Type eleTy) {
      argAttrs[pos] = DictionaryAttr::get(
          ctx, NamedAttribute(StringAttr::get(ctx, name),
                              TypeAttr::get(ccTypeToLLVMType(eleTy))));
      hasArgAttrs = true;
    };
    if (cudaq::opt::factory::hasHiddenSRet(devFuncTy)) {
      if (auto ptrTy =
              dyn_cast<cudaq::cc::PointerType>(newDevFuncTy.getInput(0))) {
        auto eleTy = ptrTy.getElementType();
        if (isa<cudaq::cc::StructType>(eleTy))
          setCallArgAttr(0, LLVM::LLVMDialect::getStructRetAttrName(), eleTy);
      }
    }
    // Large static structs are passed byval on x86_64. See isByValStructArg.
    for (unsigned pos : getByValArgPositions(devFuncTy, newDevFuncTy, module))
      setCallArgAttr(pos, LLVM::LLVMDialect::getByValAttrName(),
                     cast<cudaq::cc::PointerType>(newDevFuncTy.getInput(pos))
                         .getElementType());
    if (hasArgAttrs)
      callDevFunc.setArgAttrsAttr(ArrayAttr::get(ctx, argAttrs));

    // Deconstruct any strings.
    for (Value v : stringArgs) {
      Value cast = cudaq::cc::CastOp::create(rewriter, loc, ptrTy, v);
      func::CallOp::create(rewriter, loc, TypeRange{},
                           cudaq::runtime::bindingDeconstructString,
                           ValueRange{cast});
    }

    // If the device function has a return value, then store it to the result
    // space in the buffer.
    if (!devFuncTy.getResults().empty()) {
      auto resTy = devFuncTy.getResult(0);
      if (cudaq::cc::isDynamicType(resTy)) {
        // The real value was constructed by the call via the sret argument.
        // As on the launch side, the thunk's second argument says whether the
        // caller is in a different address space (client-server). If so, the
        // result must be returned as a pointer-free message and this side
        // releases its own copy.
        Value isRemote = entryBlock->getArgument(1);
        Block *remoteBlock = rewriter.createBlock(entryBlock->getParent());
        Block *sharedBlock = rewriter.createBlock(entryBlock->getParent());
        rewriter.setInsertionPointToEnd(entryBlock);
        cf::CondBranchOp::create(rewriter, loc, isRemote, remoteBlock,
                                 sharedBlock);
        rewriter.setInsertionPointToEnd(remoteBlock);
        Value message = packDynamicResultMessage(
            loc, rewriter, module, resTy, sretVar,
            unmarshalFunc.getFunctionType().getResult(0));
        func::ReturnOp::create(rewriter, loc, message);
        // Otherwise, the caller shares this address space, so the value is
        // handed back in place. Reduce it into the buffer's dynamic-output-slot
        // shape and store that into the result slot.
        rewriter.setInsertionPointToEnd(sharedBlock);
        Value realVal = cudaq::opt::marshal::reduceHostToDeviceValue(
            loc, rewriter, module, resTy, sretVar);
        Value slotVal = cudaq::opt::marshal::valueToBufferSlot(loc, rewriter,
                                                               resTy, realVal);
        std::int32_t numInputs = devFuncTy.getNumInputs();
        Type bufMemberTy = bufferTy.getMember(numInputs);
        auto outputPtr = cudaq::cc::ComputePtrOp::create(
            rewriter, loc, cudaq::cc::PointerType::get(bufMemberTy), argsBuffer,
            ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
        cudaq::cc::StoreOp::create(rewriter, loc, slotVal, outputPtr);
      } else {
        // callDevFunc's result is typed using the host-side ABI conversion,
        // which for a small, non-empty struct result may be a raw packed
        // register value (e.g. AArch64's [2 x i64] for a mixed-member
        // struct) rather than the buffer's plain, unconverted member type.
        // These are not always interchangeable even when they happen to be
        // the same size: on real AArch64 hardware, storing/reloading such an
        // ABI-packed value AS the plain struct type (relying on them being
        // structurally identical) has been observed to silently corrupt
        // floating-point members. Always go through an explicit memory
        // round-trip - store the raw ABI-typed result, then reload it
        // through a pointer cast to the buffer's member type - so the
        // reinterpretation is explicit rather than assumed.
        std::int32_t numInputs = devFuncTy.getNumInputs();
        Type bufMemberTy = bufferTy.getMember(numInputs);
        auto outputPtr = cudaq::cc::ComputePtrOp::create(
            rewriter, loc, cudaq::cc::PointerType::get(bufMemberTy), argsBuffer,
            ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
        if (sretVar) {
          // A static result too large for registers was returned in place via
          // the hidden sret argument.
          auto sretPtr = cudaq::cc::CastOp::create(
              rewriter, loc, cudaq::cc::PointerType::get(bufMemberTy), sretVar);
          auto sretVal = cudaq::cc::LoadOp::create(rewriter, loc, sretPtr);
          cudaq::cc::StoreOp::create(rewriter, loc, sretVal, outputPtr);
        } else {
          Value callResult = callDevFunc.getResult(0);
          if (callResult.getType() == bufMemberTy) {
            cudaq::cc::StoreOp::create(rewriter, loc, callResult, outputPtr);
          } else {
            auto scratch = cudaq::cc::AllocaOp::create(rewriter, loc,
                                                       callResult.getType());
            cudaq::cc::StoreOp::create(rewriter, loc, callResult, scratch);
            auto reinterpPtr = cudaq::cc::CastOp::create(
                rewriter, loc, cudaq::cc::PointerType::get(bufMemberTy),
                scratch);
            auto reinterpVal =
                cudaq::cc::LoadOp::create(rewriter, loc, reinterpPtr);
            cudaq::cc::StoreOp::create(rewriter, loc, reinterpVal, outputPtr);
          }
        }
      }
    }

    auto zeroCall = func::CallOp::create(
        rewriter, loc, unmarshalFunc.getFunctionType().getResult(0),
        "__nvqpp_zeroDynamicResult", ValueRange{});
    func::ReturnOp::create(rewriter, loc, zeroCall.getResult(0));
    unmarshalFunc.setPublic();
  }

  static void genRegistrationHook(Location loc, StringRef callbackName,
                                  PatternRewriter &rewriter,
                                  LLVM::GlobalOp callbackNameObj,
                                  func::FuncOp unmarshalFunc,
                                  func::FuncOp devFunc, ModuleOp module) {
    auto *ctx = rewriter.getContext();
    auto ptrType = cudaq::cc::PointerType::get(rewriter.getI8Type());
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());
    auto initFun = LLVM::LLVMFuncOp::create(
        rewriter, loc, callbackName.str() + ".callbackRegFunc",
        LLVM::LLVMFunctionType::get(cudaq::opt::factory::getVoidType(ctx), {}));
    auto *initFunEntry = initFun.addEntryBlock(rewriter);
    rewriter.setInsertionPointToStart(initFunEntry);
    auto callbackAddr = cudaq::cc::AddressOfOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(callbackNameObj.getType()),
        callbackNameObj.getSymName());
    auto castCallbackRef =
        cudaq::cc::CastOp::create(rewriter, loc, ptrType, callbackAddr);
    auto unmarshalCon =
        func::ConstantOp::create(rewriter, loc, unmarshalFunc.getFunctionType(),
                                 unmarshalFunc.getName());
    auto unmarshalAddr =
        cudaq::cc::CastOp::create(rewriter, loc, ptrType, unmarshalCon);
    // Use the same host-side ABI type that DistributedFuncPat will rewrite
    // devFunc's declared signature to. Using devFunc's current type here would
    // leave a stale, mismatched func.constant reference once that rewrite
    // happens.
    auto devFuncHostTy = cudaq::opt::factory::toHostSideFuncType(
        devFunc.getFunctionType(), /*addThisPtr=*/false, module);
    auto devFuncCon = func::ConstantOp::create(rewriter, loc, devFuncHostTy,
                                               devFunc.getName());
    auto devFuncAddr =
        cudaq::cc::CastOp::create(rewriter, loc, ptrType, devFuncCon);
    func::CallOp::create(
        rewriter, loc, TypeRange{}, cudaq::runtime::CudaqRegisterCallbackName,
        ValueRange{castCallbackRef, unmarshalAddr, devFuncAddr});
    LLVM::ReturnOp::create(rewriter, loc, ValueRange{});

    // The registration is called by a global ctor at init time.
    cudaq::opt::factory::createGlobalCtorCall(
        module, FlatSymbolRefAttr::get(ctx, initFun.getName()));
  }
};

/// This pass replaces cc.device_call ops with the generalized,
/// distributed-memory reference implementation. This means autogenerated
/// marshal and unmarshal functions that dispatch through a single runtime hook.
/// This is required if different processing units in the aggregate QPU have
/// distributed (not shared) memory spaces. The actual reference implementation
/// makes the simplification for the sake of implementation where the "device"
/// and host share the same process and address space. Thus the runtime hook is
/// a same-process NOP that hands the marshaled buffer straight to the unmarshal
/// function. This is clearly not how a distributed memory model would actually
/// work in practice.
class DistributedDeviceCallPass
    : public cudaq::opt::impl::DistributedDeviceCallBase<
          DistributedDeviceCallPass> {
public:
  using DistributedDeviceCallBase::DistributedDeviceCallBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ModuleOp module = getOperation();
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::runtime::extractDevPtr))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::extractDevPtr);
      signalPassFailure();
      return;
    }

    patterns.add<ResolveDevicePtrOpPat>(ctx);

    // The generalized reference lowering: the compiler generates the
    // marshal and unmarshal code and calls through the runtime's dispatch
    // hook, which is target-specific, which is a same-process NOP for the
    // reference implementation.
    if (failed(irBuilder.loadIntrinsic(
            module, cudaq::runtime::CudaqRegisterCallbackName))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::CudaqRegisterCallbackName);
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::runtime::callDeviceCallback))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::callDeviceCallback);
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_zeroDynamicResult"))) {
      module.emitError("could not load __nvqpp_zeroDynamicResult");
      signalPassFailure();
      return;
    }
    // reduceHostToDeviceValue allocates the backing store of a returned
    // recursively dynamic value on the heap.
    if (failed(irBuilder.loadIntrinsic(module, "malloc"))) {
      module.emitError("could not load malloc");
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, cudaq::llvmMemCopyIntrinsic))) {
      module.emitError(std::string("could not load ") +
                       cudaq::llvmMemCopyIntrinsic);
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(
            module, cudaq::runtime::bindingInitializeString))) {
      module.emitError(std::string("could not load ") +
                       cudaq::runtime::bindingInitializeString);
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(
            module, cudaq::runtime::bindingDeconstructString))) {
      module.emitError(std::string("could not load ") +
                       cudaq::runtime::bindingDeconstructString);
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::sequenceBoolCtorFromInitList))) {
      module.emitError(std::string("could not load ") +
                       cudaq::sequenceBoolCtorFromInitList);
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::sequenceBoolUnpackToInitList))) {
      module.emitError(std::string("could not load ") +
                       cudaq::sequenceBoolUnpackToInitList);
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_vectorCopyCtor"))) {
      module.emitError("could not load __nvqpp_vectorCopyCtor");
      signalPassFailure();
      return;
    }
    // Used to pack a dynamic result in a message and release the host value.
    for (const char *name :
         {cudaq::runtime::hostDeallocate, cudaq::sequenceBoolDestroy,
          cudaq::sequenceBoolFreeTemporaryLists}) {
      if (failed(irBuilder.loadIntrinsic(module, name))) {
        module.emitError(std::string("could not load ") + name);
        signalPassFailure();
        return;
      }
    }

    patterns.insert<DistributedDeviceCallPat>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();

    RewritePatternSet patterns2(ctx);
    patterns2.insert<DistributedFuncPat>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns2))))
      signalPassFailure();
  }
};
} // namespace
