/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Hash.h"
#include "llvm/Support/MD5.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace cudaq::opt {
#define GEN_PASS_DEF_DISTRIBUTEDDEVICECALL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "distributed-device-call"

using namespace mlir;

namespace {
// Matches the device_call ABI status for a malformed successful response.
constexpr std::int32_t RealtimeRemoteErrorStatus = 6;

static bool isSupportedRealtimeScalar(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    unsigned width = intTy.getWidth();
    return width == 1 || width == 8 || width == 16 || width == 32 ||
           width == 64;
  }
  return isa<Float32Type, Float64Type>(ty);
}

static std::optional<Type>
getSupportedRealtimePrimitiveArrayElement(Type elemTy) {
  if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    unsigned width = intTy.getWidth();
    if (width == 1 || width == 8 || width == 32)
      return elemTy;
  }
  if (isa<Float32Type, Float64Type>(elemTy))
    return elemTy;
  return std::nullopt;
}

static std::optional<Type> getSupportedRealtimeStdvecElement(Type ty) {
  auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(ty);
  if (!stdvecTy)
    return std::nullopt;
  Type elemTy = stdvecTy.getElementType();
  if (isa<cudaq::cc::MeasureHandleType>(elemTy))
    return IntegerType::get(elemTy.getContext(), 1);
  return getSupportedRealtimePrimitiveArrayElement(elemTy);
}

static std::optional<Type> getSupportedRealtimeStdvecResultElement(Type ty) {
  auto stdvecTy = dyn_cast<cudaq::cc::StdvecType>(ty);
  if (!stdvecTy)
    return std::nullopt;
  Type elemTy = stdvecTy.getElementType();
  if (isa<cudaq::cc::MeasureHandleType>(elemTy))
    return std::nullopt;
  return getSupportedRealtimePrimitiveArrayElement(elemTy);
}

static bool isRealtimeStdvecArg(Value arg) {
  return getSupportedRealtimeStdvecElement(arg.getType()).has_value();
}

static ArrayRef<std::int64_t>
getRealtimeResponseStdvecIndices(cudaq::cc::DeviceCallOp op) {
  if (auto attr = op.getByRefVecArgIndicesAttr())
    return attr.asArrayRef();
  return {};
}

static std::optional<unsigned>
getRealtimeResponseStdvecIndex(cudaq::cc::DeviceCallOp op) {
  auto indices = getRealtimeResponseStdvecIndices(op);
  if (indices.empty())
    return std::nullopt;
  return static_cast<unsigned>(indices.front());
}

static std::uint64_t realtimePrimitiveSize(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return cudaq::opt::convertBitsToBytes<std::uint64_t>(intTy.getWidth());
  if (auto floatTy = dyn_cast<FloatType>(ty))
    return cudaq::opt::convertBitsToBytes<std::uint64_t>(floatTy.getWidth());
  return 1;
}

static std::uint64_t realtimeScalarAlignment(Type ty) {
  return realtimePrimitiveSize(ty);
}

static std::uint64_t realtimeArrayElementSize(Type ty) {
  return realtimePrimitiveSize(ty);
}

static LogicalResult validateRealtimeDeviceCall(cudaq::cc::DeviceCallOp op) {
  if (op.getNumResults() > 1)
    return op.emitOpError(
        "realtime device_call lowering supports at most one result");

  auto args = op.getArgs();
  auto responseStdvecIndices = getRealtimeResponseStdvecIndices(op);
  if (responseStdvecIndices.size() > 1)
    return op.emitOpError(
        "realtime device_call lowering supports at most one by-ref stdvec "
        "result argument");
  if (!responseStdvecIndices.empty()) {
    if (op.getNumResults() != 0)
      return op.emitOpError(
          "realtime device_call lowering does not support both scalar and "
          "by-ref stdvec results");

    std::int64_t index = responseStdvecIndices.front();
    assert(index >= 0 && static_cast<std::size_t>(index) < args.size() &&
           "DeviceCallOp verifier should have validated by-ref vec indices");
    if (!getSupportedRealtimeStdvecResultElement(args[index].getType()))
      return op.emitOpError("realtime device_call lowering does not support "
                            "by-ref stdvec result type ")
             << args[index].getType();
  }

  for (unsigned i = 0, e = args.size(); i < e; ++i) {
    auto arg = args[i];
    if (isRealtimeStdvecArg(arg))
      continue;
    if (isa<cudaq::cc::PointerType>(arg.getType()))
      return op.emitOpError(
          "realtime device_call lowering does not support raw pointer "
          "arguments");
    if (!isSupportedRealtimeScalar(arg.getType()))
      return op.emitOpError("realtime device_call lowering does not support "
                            "argument type ")
             << arg.getType();
  }

  if (op.getNumResults() == 1) {
    Type resultTy = op.getResult(0).getType();
    if (!isSupportedRealtimeScalar(resultTy)) {
      return op.emitOpError("realtime device_call lowering does not support "
                            "result type ")
             << resultTy;
    }
  }

  return success();
}

static LogicalResult validateRealtimeDeviceCalls(ModuleOp module) {
  WalkResult result = module.walk([](cudaq::cc::DeviceCallOp op) {
    if (failed(validateRealtimeDeviceCall(op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return success(!result.wasInterrupted());
}

static Value i64Constant(OpBuilder &builder, Location loc, std::int64_t value) {
  return arith::ConstantIntOp::create(builder, loc, value, 64);
}

static Value castIntegerToI32(OpBuilder &builder, Location loc, Value value) {
  auto i32Ty = builder.getI32Type();
  if (value.getType() == i32Ty)
    return value;
  auto intTy = cast<IntegerType>(value.getType());
  if (intTy.getWidth() > 32)
    return arith::TruncIOp::create(builder, loc, i32Ty, value);
  return arith::ExtUIOp::create(builder, loc, i32Ty, value);
}

/// Emit IR that rounds `offset` up to the next multiple of `alignment`
/// (a power of two). When `alignment <= 1` this is a no-op and `offset` is
/// returned unchanged.
static Value alignOffsetTo(OpBuilder &builder, Location loc, Value offset,
                           std::uint64_t alignment) {
  if (alignment <= 1)
    return offset;
  auto addend = arith::ConstantIntOp::create(builder, loc, alignment - 1, 64);
  auto mask = arith::ConstantIntOp::create(
      builder, loc, -static_cast<std::int64_t>(alignment), 64);
  auto incremented = arith::AddIOp::create(builder, loc, offset, addend);
  return arith::AndIOp::create(builder, loc, incremented, mask);
}

static Value bytePtrAt(OpBuilder &builder, Location loc, Value buffer,
                       Value offset) {
  return cudaq::cc::ComputePtrOp::create(
      builder, loc, cudaq::cc::PointerType::get(builder.getI8Type()), buffer,
      ArrayRef<cudaq::cc::ComputePtrArg>{offset});
}

/// Compute `buffer + offset` as a pointer to `elemTy`.
static Value typedPtrAt(OpBuilder &builder, Location loc, Value buffer,
                        Value offset, Type elemTy) {
  Value bytePtr = bytePtrAt(builder, loc, buffer, offset);
  return cudaq::cc::CastOp::create(
      builder, loc, cudaq::cc::PointerType::get(elemTy), bytePtr);
}

/// Align `cursor` to `alignment` (power of two), store `length` as an i64
/// length prefix into the request `buffer` at the aligned cursor, and return
/// the new cursor positioned just past the length prefix.
static Value writeLengthPrefix(OpBuilder &builder, Location loc, Value cursor,
                               Value buffer, Value length) {
  auto i64Ty = builder.getI64Type();
  // The prefix is an i64, so round the cursor up to an 8-byte boundary
  cursor = alignOffsetTo(builder, loc, cursor, sizeof(std::uint64_t));
  Value lenPtr = typedPtrAt(builder, loc, buffer, cursor, i64Ty);
  cudaq::cc::StoreOp::create(builder, loc, length, lenPtr);
  return arith::AddIOp::create(
      builder, loc, cursor, i64Constant(builder, loc, sizeof(std::uint64_t)));
}

static Value computeArrayPayloadSize(OpBuilder &builder, Location loc,
                                     Value length, Type elementTy) {
  std::uint64_t elementSize = realtimeArrayElementSize(elementTy);
  if (elementSize == 1)
    return length;
  return arith::MulIOp::create(builder, loc, length,
                               i64Constant(builder, loc, elementSize));
}

/// Compute the total request payload size (in bytes) for a realtime
/// device_call with the given `args`.
static Value
computeRealtimePayloadSize(OpBuilder &builder, Location loc, ValueRange args,
                           std::optional<unsigned> responseStdvec) {
  auto i64Ty = builder.getI64Type();
  Value lenSize = i64Constant(builder, loc, sizeof(std::uint64_t));
  Value size = i64Constant(builder, loc, 0);
  auto addAlignedLength = [&]() {
    // Length prefixes are i64; round the running size up to an 8-byte
    // boundary.
    size = alignOffsetTo(builder, loc, size, sizeof(std::uint64_t));
    size = arith::AddIOp::create(builder, loc, size, lenSize);
  };
  for (unsigned i = 0, e = args.size(); i < e; ++i) {
    Value arg = args[i];
    if (auto elementTy = getSupportedRealtimeStdvecElement(arg.getType())) {
      Value arrayLength =
          cudaq::cc::StdvecSizeOp::create(builder, loc, i64Ty, arg);
      addAlignedLength();
      if (responseStdvec && *responseStdvec == i)
        continue;
      Value arrayBytes =
          computeArrayPayloadSize(builder, loc, arrayLength, *elementTy);
      size = arith::AddIOp::create(builder, loc, size, arrayBytes);
      continue;
    }
    // Pad the payload size to the scalar's natural alignment so the eventual
    // store lands on an aligned offset. No-op for i1/i8 args.
    size = alignOffsetTo(builder, loc, size,
                         realtimeScalarAlignment(arg.getType()));
    Value argSize =
        cudaq::cc::SizeOfOp::create(builder, loc, i64Ty, arg.getType());
    size = arith::AddIOp::create(builder, loc, size, argSize);
  }
  return size;
}

static Type getStdvecSourceElementType(Value stdvec) {
  return cast<cudaq::cc::StdvecType>(stdvec.getType()).getElementType();
}

static Value getStdvecLength(OpBuilder &builder, Location loc, Value stdvec) {
  return cudaq::cc::StdvecSizeOp::create(builder, loc, builder.getI64Type(),
                                         stdvec);
}

static Value getStdvecData(OpBuilder &builder, Location loc, Value stdvec,
                           Type elementTy) {
  auto arrayPtrTy =
      cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(elementTy));
  return cudaq::cc::StdvecDataOp::create(builder, loc, arrayPtrTy, stdvec);
}

static Value marshalArrayArgument(OpBuilder &builder, Location loc,
                                  Value cursor, Value requestBuffer, Value data,
                                  Value arrayLength, Type sourceElementTy,
                                  Type payloadElementTy) {
  auto i8Ty = builder.getI8Type();
  cursor = writeLengthPrefix(builder, loc, cursor, requestBuffer, arrayLength);

  // Copy array elements one at a time so generated IR can preserve the source
  // element type. Measurement handles are first discriminated to bool payload
  // bytes; the transport never carries the opaque handle representation.
  Value dataStart = cursor;
  auto arrayDataTy =
      cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(sourceElementTy));
  if (data.getType() != arrayDataTy)
    data = cudaq::cc::CastOp::create(builder, loc, arrayDataTy, data);
  auto sourcePtrTy = cudaq::cc::PointerType::get(sourceElementTy);
  std::uint64_t elementSize = realtimeArrayElementSize(payloadElementTy);
  cudaq::opt::factory::createInvariantLoop(
      builder, loc, arrayLength,
      [&, elementSize, payloadElementTy, sourceElementTy](
          OpBuilder &builder, Location loc, Region &, Block &block) {
        Value index = block.getArgument(0);
        auto elementPtr = cudaq::cc::ComputePtrOp::create(
            builder, loc, sourcePtrTy, data,
            ArrayRef<cudaq::cc::ComputePtrArg>{index});
        auto loadedElement =
            cudaq::cc::LoadOp::create(builder, loc, elementPtr);
        Value element = loadedElement.getResult();
        if (isa<cudaq::cc::MeasureHandleType>(sourceElementTy))
          element = cudaq::quake::DiscriminateOp::create(
              builder, loc, builder.getI1Type(), element);

        Value dstIndexOffset = index;
        if (elementSize != 1)
          dstIndexOffset = arith::MulIOp::create(
              builder, loc, index, i64Constant(builder, loc, elementSize));
        Value dstOffset =
            arith::AddIOp::create(builder, loc, dataStart, dstIndexOffset);
        if (auto intTy = dyn_cast<IntegerType>(payloadElementTy);
            intTy && intTy.getWidth() == 1) {
          auto byte = cudaq::cc::CastOp::create(
              builder, loc, i8Ty, element, cudaq::cc::CastOpMode::Unsigned);
          Value dstPtr = bytePtrAt(builder, loc, requestBuffer, dstOffset);
          cudaq::cc::StoreOp::create(builder, loc, byte, dstPtr);
        } else {
          Value dstPtr = typedPtrAt(builder, loc, requestBuffer, dstOffset,
                                    payloadElementTy);
          cudaq::cc::StoreOp::create(builder, loc, element, dstPtr);
        }
      });
  Value arrayBytes =
      computeArrayPayloadSize(builder, loc, arrayLength, payloadElementTy);
  return arith::AddIOp::create(builder, loc, cursor, arrayBytes);
}

static void copyResponseToStdvec(OpBuilder &builder, Location loc,
                                 Value responseBuffer, Value responseLen,
                                 Value stdvec) {
  auto i8Ty = builder.getI8Type();
  Type sourceElementTy = getStdvecSourceElementType(stdvec);
  Value data = getStdvecData(builder, loc, stdvec, sourceElementTy);
  Value byteData = cudaq::cc::CastOp::create(
      builder, loc,
      cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty)), data);
  cudaq::opt::factory::createInvariantLoop(
      builder, loc, responseLen,
      [&](OpBuilder &builder, Location loc, Region &, Block &block) {
        Value index = block.getArgument(0);
        Value srcPtr = bytePtrAt(builder, loc, responseBuffer, index);
        auto loadedByte = cudaq::cc::LoadOp::create(builder, loc, srcPtr);
        Value dstPtr = cudaq::cc::ComputePtrOp::create(
            builder, loc, cudaq::cc::PointerType::get(i8Ty), byteData,
            ArrayRef<cudaq::cc::ComputePtrArg>{index});
        cudaq::cc::StoreOp::create(builder, loc, loadedByte.getResult(),
                                   dstPtr);
      });
}

static void emitTrapOnCondition(PatternRewriter &rewriter, Location loc,
                                Value condition, Value status,
                                Value frameHandle = {}) {
  cudaq::cc::IfOp::create(
      rewriter, loc, TypeRange{}, condition,
      [&](OpBuilder &builder, Location loc, Region &region) {
        region.push_back(new Block());
        auto &body = region.front();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&body);
        if (frameHandle) {
          // Realtime frame release ABI:
          //   extern "C" void __cudaq_device_call_release_realtime_frame(
          //       void *frame_handle);
          // This lowering uses the non-throwing release wrapper with the same
          // argument shape.
          func::CallOp::create(
              builder, loc, TypeRange{},
              cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
              ValueRange{frameHandle});
        }
        auto status64 =
            cudaq::cc::CastOp::create(builder, loc, builder.getI64Type(),
                                      status, cudaq::cc::CastOpMode::Signed);
        func::CallOp::create(builder, loc, TypeRange{}, cudaq::opt::QISTrap,
                             ValueRange{status64});
        cudaq::cc::ContinueOp::create(builder, loc);
      });
}

static void emitTrapOnFailure(PatternRewriter &rewriter, Location loc,
                              Value status, Value frameHandle = {}) {
  auto zeroStatus = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  auto failedStatus = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::ne, status, zeroStatus);
  emitTrapOnCondition(rewriter, loc, failedStatus, status, frameHandle);
}

static void addTrapImplementation(cudaq::cc::DeviceCallOp devcall,
                                  func::FuncOp devFunc,
                                  PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto &entryBlock = *devFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(&entryBlock);

  // Error code 2 indicates illegal execution of unreachable host code.
  Value errorCodeTwo =
      arith::ConstantIntOp::create(rewriter, devcall.getLoc(), 2, 64);
  func::CallOp::create(rewriter, devcall.getLoc(), TypeRange{},
                       cudaq::opt::QISTrap, ValueRange{errorCodeTwo});

  // Return unreachable values of the declared result types. The values only
  // make the IR well-formed; execution traps before reaching them.
  SmallVector<Value> trapResults;
  for (Type resTy : devFunc.getFunctionType().getResults()) {
    auto nullPtr = arith::ConstantOp::create(
        rewriter, devcall.getLoc(),
        rewriter.getZeroAttr(rewriter.getIntegerType(64)));
    auto ptrTy = cudaq::cc::PointerType::get(resTy);
    auto castedNullPtr =
        cudaq::cc::CastOp::create(rewriter, devcall.getLoc(), ptrTy, nullPtr);
    auto loadedVal =
        cudaq::cc::LoadOp::create(rewriter, devcall.getLoc(), castedNullPtr);
    trapResults.push_back(loadedVal);
  }

  func::ReturnOp::create(rewriter, devcall.getLoc(), trapResults);
}

static void setTrapImplementationLinkage(func::FuncOp devFunc,
                                         PatternRewriter &rewriter) {
  devFunc.setPrivate();
  auto weakOdrLinkage = mlir::LLVM::linkage::Linkage::WeakODR;
  auto linkage =
      mlir::LLVM::LinkageAttr::get(rewriter.getContext(), weakOdrLinkage);
  devFunc->setAttr("llvm.linkage", linkage);
}

class QIRVendorDeviceCallPat
    : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
  bool insertTrapImplementation;

public:
  using OpRewritePattern::OpRewritePattern;

  QIRVendorDeviceCallPat(MLIRContext *context, bool insertTrapImpl)
      : OpRewritePattern(context), insertTrapImplementation(insertTrapImpl) {}

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    constexpr const char PassthroughAttr[] = "passthrough";
    constexpr const char QIRVendorAttr[] = "cudaq-fnid";
    auto module = devcall->getParentOfType<ModuleOp>();
    auto devFuncName = devcall.getCallee();
    auto devFunc = module.lookupSymbol<func::FuncOp>(devFuncName);
    if (!devFunc) {
      LLVM_DEBUG(llvm::dbgs() << "cannot find the function " << devFuncName
                              << " in module\n");
      return failure();
    }

    llvm::MD5 hash;
    hash.update(devFuncName);
    llvm::MD5::MD5Result result;
    hash.final(result);
    std::uint32_t callbackCode = result.low();

    if (insertTrapImplementation && devFunc.isDeclaration()) {
      // If `insertTrapImplementation` is enabled (e.g., AOT compilation for
      // remote hardware providers), we want to insert a trap implementation for
      // any unresolved device function (declaration only), so that we can
      // perform AOT compilation without needing the actual device function
      // definitions. This trap function will never be executed as the remote
      // JIT pipeline would not be using the `device_call` functions anyway.
      // Rather, these functions will only be resolved at runtime by the remote
      // provider's runtime library.

      // (1) Add a trap implementation for this device function declaration.
      addTrapImplementation(devcall, devFunc, rewriter);

      // (2) Set this trap function as private and weak_odr linkage, to allow
      // multiple definitions across translation units without linker errors.
      // For example, compiling for a remote hardware provider with the actual
      // device call library linkage (even though unused) should not cause any
      // problems.
      setTrapImplementationLinkage(devFunc, rewriter);

      // (3) Replace the device call with a no-inline call to prevent inlining
      // of the trap function.
      // We use a no-inline call here to ensure that the call to the device
      // function is preserved as a call in the IR (even in the presence of the
      // trap implementation). If the actual implementation is provided at link
      // time, it will be used instead of the trap implementation due to the
      // weak_odr linkage.
      rewriter.replaceOpWithNewOp<cudaq::cc::NoInlineCallOp>(
          devcall, devFunc.getFunctionType().getResults(), devFuncName,
          devcall.getArgs(), ArrayAttr{}, ArrayAttr{});

      return success();
    }

    bool needToAddIt = true;
    SmallVector<Attribute> funcIdAttr;
    if (auto passthruAttr = devFunc->getAttr(PassthroughAttr)) {
      auto arrayAttr = cast<ArrayAttr>(passthruAttr);
      funcIdAttr.append(arrayAttr.begin(), arrayAttr.end());
      for (auto a : arrayAttr) {
        if (auto strArrAttr = dyn_cast<ArrayAttr>(a)) {
          auto strAttr = dyn_cast<StringAttr>(strArrAttr[0]);
          if (!strAttr)
            continue;
          if (strAttr.getValue() == QIRVendorAttr) {
            needToAddIt = false;
            break;
          }
        }
      }
    }
    if (needToAddIt) {
      auto callbackCodeAsStr = std::to_string(callbackCode);
      funcIdAttr.push_back(rewriter.getStrArrayAttr(
          {QIRVendorAttr, rewriter.getStringAttr(callbackCodeAsStr)}));
      devFunc->setAttr(PassthroughAttr, rewriter.getArrayAttr(funcIdAttr));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(
        devcall, devFunc.getFunctionType().getResults(), devFuncName,
        devcall.getArgs());
    return success();
  }
};

/// Lower a validated `cc.device_call` to the realtime protocol.
///
/// The generated IR follows the sequence: compute request and
/// response sizes, acquire a frame from the selected device channel, marshal
/// the operands into the request payload, dispatch the frame, and read the
/// response before releasing the frame.
class RealtimeDeviceCallPat : public OpRewritePattern<cudaq::cc::DeviceCallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::DeviceCallOp devcall,
                                PatternRewriter &rewriter) const override {
    // Module-level validation has already been performed by the pass before
    // patterns are applied; assert here to catch programming errors.
    assert(succeeded(validateRealtimeDeviceCall(devcall)) &&
           "realtime device_call should have been validated");

    auto loc = devcall.getLoc();
    auto callee = devcall.getCallee();
    // Function id is a FNV-1a hash of the callee name
    std::uint32_t functionId =
        llvm::getKCFITypeID(callee, llvm::KCFIHashAlgorithm::FNV1a);

    auto i8Ty = rewriter.getI8Type();
    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);

    auto args = devcall.getArgs();
    std::optional<unsigned> responseStdvec =
        getRealtimeResponseStdvecIndex(devcall);
    // Size the request payload
    Value requestSize =
        computeRealtimePayloadSize(rewriter, loc, args, responseStdvec);

    const auto [resultTy, responseCapacity] = [&]() -> std::pair<Type, Value> {
      if (devcall.getNumResults() == 1) {
        const Type resultTy = devcall.getResult(0).getType();
        const Value responseCapacity =
            cudaq::cc::SizeOfOp::create(rewriter, loc, i64Ty, resultTy);
        return {resultTy, responseCapacity};
      }
      if (responseStdvec) {
        const Value outVector = args[*responseStdvec];
        const Type elementTy =
            *getSupportedRealtimeStdvecResultElement(outVector.getType());
        const Value outLength = getStdvecLength(rewriter, loc, outVector);
        return {Type{},
                computeArrayPayloadSize(rewriter, loc, outLength, elementTy)};
      }
      return {Type{}, i64Constant(rewriter, loc, 0)};
    }();

    auto functionIdValue = arith::ConstantIntOp::create(
        rewriter, loc, static_cast<std::int64_t>(functionId), 32);

    // Device Id is optional; use 0 as the default value for the runtime API
    // when not specified.
    Value deviceIdValue =
        devcall.getDevice()
            ? castIntegerToI32(rewriter, loc, devcall.getDevice())
            : arith::ConstantIntOp::create(rewriter, loc, 0, 32);

    // The acquire runtime API returns the opaque frame handle and the request /
    // response payload pointers through out-parameters, so create stack slots
    // for those values in the lowered IR.
    Value frameHandleSlot = cudaq::cc::AllocaOp::create(rewriter, loc, ptrI8Ty);
    Value requestPayloadSlot =
        cudaq::cc::AllocaOp::create(rewriter, loc, ptrI8Ty);
    Value responsePayloadSlot =
        cudaq::cc::AllocaOp::create(rewriter, loc, ptrI8Ty);
    Value responseLen = cudaq::cc::AllocaOp::create(rewriter, loc, i64Ty);
    cudaq::cc::StoreOp::create(rewriter, loc, i64Constant(rewriter, loc, 0),
                               responseLen);

    // Reserve a frame from the selected channel. On acquire failure no frame is
    // owned yet, so trapping does not need a release callback.
    // Function signature of the acquire:
    //   extern "C" int32_t __cudaq_device_call_acquire_realtime_frame(
    //       uint32_t device_id, uint32_t function_id,
    //       uint64_t request_size, uint64_t response_capacity,
    //       void **frame_handle, void **request_payload,
    //       void **response_payload);
    auto acquireStatus = func::CallOp::create(
        rewriter, loc, i32Ty, cudaq::runtime::deviceCallAcquireRealtimeFrame,
        ValueRange{deviceIdValue, functionIdValue, requestSize,
                   responseCapacity, frameHandleSlot, requestPayloadSlot,
                   responsePayloadSlot});
    emitTrapOnFailure(rewriter, loc, acquireStatus.getResult(0));

    Value frameHandle =
        cudaq::cc::LoadOp::create(rewriter, loc, frameHandleSlot);
    Value requestPayload =
        cudaq::cc::LoadOp::create(rewriter, loc, requestPayloadSlot);
    auto payloadArrayPtrTy =
        cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i8Ty));
    Value requestBuffer = cudaq::cc::CastOp::create(
        rewriter, loc, payloadArrayPtrTy, requestPayload);
    const Value responseBuffer = [&] {
      if (!resultTy && !responseStdvec)
        return Value{};
      const Value responsePayload =
          cudaq::cc::LoadOp::create(rewriter, loc, responsePayloadSlot);
      const Value buffer = cudaq::cc::CastOp::create(
          rewriter, loc, payloadArrayPtrTy, responsePayload);
      return buffer;
    }();

    // Marshal operands into the request payload. Scalars are aligned and stored
    // directly. Supported stdvec arguments use a length prefix followed by
    // element bytes; a response stdvec carries its extent in the request and
    // receives its elements through the response payload.
    Value cursor = i64Constant(rewriter, loc, 0);
    for (unsigned i = 0, e = args.size(); i < e; ++i) {
      auto arg = args[i];
      if (auto elementTy = getSupportedRealtimeStdvecElement(arg.getType())) {
        Value arrayLength = getStdvecLength(rewriter, loc, arg);
        if (responseStdvec && *responseStdvec == i) {
          cursor = writeLengthPrefix(rewriter, loc, cursor, requestBuffer,
                                     arrayLength);
          continue;
        }

        Type sourceElementTy = getStdvecSourceElementType(arg);
        Value data = getStdvecData(rewriter, loc, arg, sourceElementTy);
        cursor =
            marshalArrayArgument(rewriter, loc, cursor, requestBuffer, data,
                                 arrayLength, sourceElementTy, *elementTy);
        continue;
      }
      // Pad the cursor to the scalar's natural alignment before storing it.
      // No-op for i1/i8 args.
      cursor = alignOffsetTo(rewriter, loc, cursor,
                             realtimeScalarAlignment(arg.getType()));
      Value typedArgPtr =
          typedPtrAt(rewriter, loc, requestBuffer, cursor, arg.getType());
      cudaq::cc::StoreOp::create(rewriter, loc, arg, typedArgPtr);
      Value argSize =
          cudaq::cc::SizeOfOp::create(rewriter, loc, i64Ty, arg.getType());
      cursor = arith::AddIOp::create(rewriter, loc, cursor, argSize);
    }

    // Dispatch the populated frame:
    //   extern "C" int32_t __cudaq_device_call_dispatch_realtime_frame(
    //       void *frame_handle, uint64_t *response_size);
    auto status = func::CallOp::create(
        rewriter, loc, i32Ty, cudaq::runtime::deviceCallDispatchRealtimeFrame,
        ValueRange{frameHandle, responseLen});
    emitTrapOnFailure(rewriter, loc, status.getResult(0), frameHandle);

    if (!resultTy) {
      if (responseStdvec) {
        // This is a by-ref stdvec result case.
        Value returnedLen =
            cudaq::cc::LoadOp::create(rewriter, loc, responseLen);
        auto unexpectedResponseLen =
            arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ne,
                                  returnedLen, responseCapacity);
        auto remoteError = arith::ConstantIntOp::create(
            rewriter, loc, RealtimeRemoteErrorStatus, 32);
        emitTrapOnCondition(rewriter, loc, unexpectedResponseLen, remoteError,
                            frameHandle);
        copyResponseToStdvec(rewriter, loc, responseBuffer, responseCapacity,
                             args[*responseStdvec]);
      }
      // Realtime frame release ABI:
      //   extern "C" void __cudaq_device_call_release_realtime_frame(
      //       void *frame_handle);
      // This lowering uses the non-throwing release wrapper with the same
      // argument shape.
      func::CallOp::create(rewriter, loc, TypeRange{},
                           cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
                           ValueRange{frameHandle});
      rewriter.eraseOp(devcall);
      return success();
    }

    // Direct scalar result case
    // The realtime lowering currently admits at most one scalar result, stored
    // at the start of the response payload.
    auto resultPtr = cudaq::cc::CastOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(resultTy), responseBuffer);
    auto result = cudaq::cc::LoadOp::create(rewriter, loc, resultPtr);
    // Realtime frame release ABI:
    //   extern "C" void __cudaq_device_call_release_realtime_frame(
    //       void *frame_handle);
    // This lowering uses the non-throwing release wrapper with the same
    // argument shape.
    func::CallOp::create(rewriter, loc, TypeRange{},
                         cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
                         ValueRange{frameHandle});
    rewriter.replaceOp(devcall, ValueRange{result.getResult()});
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
    if (realtimeLowering) {
      if (failed(validateRealtimeDeviceCalls(module))) {
        signalPassFailure();
        return;
      }

      for (auto name : {cudaq::runtime::deviceCallAcquireRealtimeFrame,
                        cudaq::runtime::deviceCallDispatchRealtimeFrame,
                        cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame}) {
        if (failed(irBuilder.loadIntrinsic(module, name))) {
          module.emitError(std::string{"could not load "} + name);
          signalPassFailure();
          return;
        }
      }

      if (failed(irBuilder.loadIntrinsic(module, cudaq::opt::QISTrap))) {
        module.emitError("could not load QIR trap function.");
        signalPassFailure();
        return;
      }

      patterns.add<RealtimeDeviceCallPat>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(patterns))))
        signalPassFailure();
      return;
    }

    if (failed(
            irBuilder.loadIntrinsic(module, cudaq::runtime::extractDevPtr))) {
      module.emitError(std::string{"could not load "} +
                       cudaq::runtime::extractDevPtr);
      signalPassFailure();
      return;
    }

    if (failed(irBuilder.loadIntrinsic(module, cudaq::opt::QISTrap))) {
      module.emitError("could not load QIR trap function.");
      signalPassFailure();
      return;
    }

    patterns.add<ResolveDevicePtrOpPat>(ctx);
    patterns.insert<QIRVendorDeviceCallPat>(ctx, insertTrapImplementation);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
    return;
  }
};
} // namespace
