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

// Any dynamic return type is supported as a device_call return type. Given a
// real device-side SSA value of type \p devTy, decompose it into the
// buffer's dynamic-output-slot shape. A SpanLikeType decomposes to a `{ptr,
// count}` pair whose pointer already refers to fully-realized device values; a
// struct decomposes member by member, recursing only into dynamic members.
static Value valueToBufferSlot(Location loc, PatternRewriter &rewriter,
                               Type devTy, Value realVal) {
  auto *ctx = rewriter.getContext();
  auto i64Ty = rewriter.getI64Type();
  if (auto spanTy = dyn_cast<cudaq::cc::SpanLikeType>(devTy)) {
    auto eleTy = spanTy.getElementType();
    auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
    Value ptr =
        cudaq::cc::SequenceDataOp::create(rewriter, loc, elePtrTy, realVal);
    Value count =
        cudaq::cc::SequenceSizeOp::create(rewriter, loc, i64Ty, realVal);
    auto slotTy =
        cudaq::cc::StructType::get(ctx, ArrayRef<Type>{elePtrTy, i64Ty});
    Value slot = cudaq::cc::UndefOp::create(rewriter, loc, slotTy);
    slot =
        cudaq::cc::InsertValueOp::create(rewriter, loc, slotTy, slot, ptr, 0);
    slot =
        cudaq::cc::InsertValueOp::create(rewriter, loc, slotTy, slot, count, 1);
    return slot;
  }
  auto strTy = cast<cudaq::cc::StructType>(devTy);
  SmallVector<Type> slotMemberTys;
  SmallVector<Value> memberVals;
  for (auto iter : llvm::enumerate(strTy.getMembers())) {
    std::int32_t idx = iter.index();
    Type memTy = iter.value();
    Value realMember =
        cudaq::cc::ExtractValueOp::create(rewriter, loc, memTy, realVal, idx);
    Value memVal = cudaq::cc::isDynamicType(memTy)
                       ? valueToBufferSlot(loc, rewriter, memTy, realMember)
                       : realMember;
    slotMemberTys.push_back(memVal.getType());
    memberVals.push_back(memVal);
  }
  auto slotTy = cudaq::cc::StructType::get(ctx, slotMemberTys);
  Value slot = cudaq::cc::UndefOp::create(rewriter, loc, slotTy);
  for (auto iter : llvm::enumerate(memberVals))
    slot = cudaq::cc::InsertValueOp::create(rewriter, loc, slotTy, slot,
                                            iter.value(), iter.index());
  return slot;
}

// The inverse of valueToBufferSlot. Given an already-loaded buffer output-slot
// value, reconstruct the real device-side SSA value of type \p devTy. A
// SpanLikeType reconstructs directly via cc.sequence_init; a struct
// reconstructs member by member.
static Value bufferSlotToValue(Location loc, PatternRewriter &rewriter,
                               Type devTy, Value slotVal) {
  if (auto spanTy = dyn_cast<cudaq::cc::SpanLikeType>(devTy)) {
    auto eleTy = spanTy.getElementType();
    auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
    auto i64Ty = rewriter.getI64Type();
    Value ptr =
        cudaq::cc::ExtractValueOp::create(rewriter, loc, elePtrTy, slotVal, 0);
    Value count =
        cudaq::cc::ExtractValueOp::create(rewriter, loc, i64Ty, slotVal, 1);
    return cudaq::cc::SequenceInitOp::create(rewriter, loc, spanTy, ptr, count);
  }
  auto strTy = cast<cudaq::cc::StructType>(devTy);
  auto slotStrTy = cast<cudaq::cc::StructType>(slotVal.getType());
  Value result = cudaq::cc::UndefOp::create(rewriter, loc, strTy);
  for (auto iter : llvm::enumerate(strTy.getMembers())) {
    std::int32_t idx = iter.index();
    Type memTy = iter.value();
    Type slotMemTy = slotStrTy.getMember(idx);
    Value memSlot = cudaq::cc::ExtractValueOp::create(rewriter, loc, slotMemTy,
                                                      slotVal, idx);
    Value memVal = cudaq::cc::isDynamicType(memTy)
                       ? bufferSlotToValue(loc, rewriter, memTy, memSlot)
                       : memSlot;
    result = cudaq::cc::InsertValueOp::create(rewriter, loc, strTy, result,
                                              memVal, idx);
  }
  return result;
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
    (void)callback;
    std::int32_t numInputs = devFuncTy.getNumInputs();
    if (cudaq::cc::isDynamicType(resTy)) {
      // Reference implementation: the unmarshal function (running synchronously
      // within the call above) already wrote the dynamic-output-slot-shaped
      // value for the result directly into the shared buffer. Read it back and
      // reconstruct the real result value from it.
      Type slotTy = bufferTy.getMember(numInputs);
      auto outputPtr = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, cudaq::cc::PointerType::get(slotTy), typedBuffer,
          ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
      Value slotVal = cudaq::cc::LoadOp::create(rewriter, loc, outputPtr);
      Value resVal = bufferSlotToValue(loc, rewriter, resTy, slotVal);
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

    // If the device function returns a dynamically sized value, the real host
    // ABI passes it back via a hidden sret pointer argument that is prepended
    // before the other arguments.
    Value sretVar;
    if (devFuncTy.getNumResults() == 1 &&
        cudaq::cc::isDynamicType(devFuncTy.getResult(0))) {
      Type sretHostTy = cudaq::opt::factory::convertToHostSideType(
          cudaq::opt::factory::getSRetElementType(devFuncTy, module), module);
      sretVar = cudaq::cc::AllocaOp::create(rewriter, loc, sretHostTy);
      args.push_back(sretVar);
    }

    std::size_t offset = 0;
    SmallVector<Value> stringArgs;
    for (auto iter : llvm::enumerate(devFuncTy.getInputs())) {
      auto [a, t] = cudaq::opt::marshal::processCallbackInputValue(
          loc, rewriter, trailingData, argsBuffer, iter.value(), iter.index(),
          bufferTy);
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
      }
      args.push_back(a);
    }

    auto callDevFunc = func::CallOp::create(
        rewriter, loc, newDevFuncTy.getResults(), devFunc.getName(), args);

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
        // Reduce it into the buffer's dynamic-output-slot shape and store that
        // into the result slot.
        Value realVal = cudaq::opt::marshal::reduceHostToDeviceValue(
            loc, rewriter, module, resTy, sretVar);
        Value slotVal = valueToBufferSlot(loc, rewriter, resTy, realVal);
        std::int32_t numInputs = devFuncTy.getNumInputs();
        Type bufMemberTy = bufferTy.getMember(numInputs);
        auto outputPtr = cudaq::cc::ComputePtrOp::create(
            rewriter, loc, cudaq::cc::PointerType::get(bufMemberTy), argsBuffer,
            ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
        cudaq::cc::StoreOp::create(rewriter, loc, slotVal, outputPtr);
      } else {
        // callDevFunc's result is already typed using the host-side ABI
        // conversion which does not preserve struct names. So, unlike the read
        // side in genNewMarshalFunc, no cast is needed here. One can just use
        // the buffer's own member type.
        std::int32_t numInputs = devFuncTy.getNumInputs();
        Type bufMemberTy = bufferTy.getMember(numInputs);
        auto outputPtr = cudaq::cc::ComputePtrOp::create(
            rewriter, loc, cudaq::cc::PointerType::get(bufMemberTy), argsBuffer,
            ArrayRef<cudaq::cc::ComputePtrArg>{numInputs});
        cudaq::cc::StoreOp::create(rewriter, loc, callDevFunc.getResult(0),
                                   outputPtr);
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
