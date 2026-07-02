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
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Hash.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace cudaq::opt {
#define GEN_PASS_DEF_REALTIMEDEVICECALL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "realtime-device-call"

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

/// Natural byte size of a supported primitive type. In the realtime payload
/// ABI this is also the scalar store alignment and the array element stride.
static std::uint64_t realtimePrimitiveSize(Type ty) {
  if (auto intTy = dyn_cast<IntegerType>(ty))
    return cudaq::opt::convertBitsToBytes<std::uint64_t>(intTy.getWidth());
  if (auto floatTy = dyn_cast<FloatType>(ty))
    return cudaq::opt::convertBitsToBytes<std::uint64_t>(floatTy.getWidth());
  return 1;
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

static LogicalResult validateRealtimeDeviceCalls(ModuleOp module,
                                                 bool &hasDeviceCalls) {
  hasDeviceCalls = false;
  WalkResult result = module.walk([&](cudaq::cc::DeviceCallOp op) {
    hasDeviceCalls = true;
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

/// Compute the number of payload bytes required for `length` logical array
/// elements. Realtime represents `i1` arrays as `TYPE_BIT_PACKED`, so eight
/// logical elements share one byte and `(length + 7) / 8` rounds a partial
/// final byte up to full storage. All other arrays use one naturally sized
/// payload element per logical element.
static Value computeArrayPayloadSize(OpBuilder &builder, Location loc,
                                     Value length, Type elementTy) {
  if (elementTy.isInteger(1)) {
    Value rounded = arith::AddIOp::create(builder, loc, length,
                                          i64Constant(builder, loc, 7));
    return arith::DivUIOp::create(builder, loc, rounded,
                                  i64Constant(builder, loc, 8));
  }
  std::uint64_t elementSize = realtimePrimitiveSize(elementTy);
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
    std::uint64_t scalarBytes = realtimePrimitiveSize(arg.getType());
    size = alignOffsetTo(builder, loc, size, scalarBytes);
    size = arith::AddIOp::create(builder, loc, size,
                                 i64Constant(builder, loc, scalarBytes));
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

static Value getStdvecData(OpBuilder &builder, Location loc, Value stdvec) {
  auto arrayPtrTy = cudaq::cc::PointerType::get(
      cudaq::cc::ArrayType::get(getStdvecSourceElementType(stdvec)));
  return cudaq::cc::StdvecDataOp::create(builder, loc, arrayPtrTy, stdvec);
}

static Value marshalArrayArgument(OpBuilder &builder, Location loc,
                                  Value cursor, Value requestBuffer, Value data,
                                  Value arrayLength, Type sourceElementTy,
                                  Type payloadElementTy) {
  auto i8Ty = builder.getI8Type();
  cursor = writeLengthPrefix(builder, loc, cursor, requestBuffer, arrayLength);

  if (payloadElementTy.isInteger(1)) {
    Value packedBytes =
        computeArrayPayloadSize(builder, loc, arrayLength, payloadElementTy);
    auto sourcePtrTy = cudaq::cc::PointerType::get(sourceElementTy);
    Value eight = i64Constant(builder, loc, 8);

    // Pack one output byte per outer iteration and keep the partially packed
    // byte as an SSA loop-carried value. The explicit least-significant-bit
    // first encoding is independent of the native data layout on either
    // endpoint. This intentionally avoids a separate zeroing pass and the
    // per-logical-bit load/OR/store sequence that would repeatedly update the
    // same request byte. Measurement discrimination remains fused with packing
    // so handle vectors need no temporary bool buffer.
    //
    // Pseudocode for the generated loops:
    //   for (byte = 0; byte < packedBytes; ++byte) {
    //     packed = 0;
    //     bits = min(arrayLength - byte * 8, 8);
    //     for (bit = 0; bit < bits; ++bit)
    //       packed |= loadAndDiscriminateIfNeeded(data[byte * 8 + bit]) << bit;
    //     request[cursor + byte] = packed;
    //   }
    cudaq::opt::factory::createInvariantLoop(
        builder, loc, packedBytes,
        [&](OpBuilder &builder, Location loc, Region &, Block &block) {
          Value byteIndex = block.getArgument(0);
          Value byteStart =
              arith::MulIOp::create(builder, loc, byteIndex, eight);
          Value remaining =
              arith::SubIOp::create(builder, loc, arrayLength, byteStart);
          Value isPartial = arith::CmpIOp::create(
              builder, loc, arith::CmpIPredicate::ult, remaining, eight);
          Value bitsInByte = arith::SelectOp::create(
              builder, loc, builder.getI64Type(), isPartial, remaining, eight);
          Value zeroIndex = i64Constant(builder, loc, 0);
          Value one = i64Constant(builder, loc, 1);
          Value zeroByte = arith::ConstantIntOp::create(builder, loc, 0, 8);
          auto packLoop = cudaq::cc::LoopOp::create(
              builder, loc, TypeRange{builder.getI64Type(), i8Ty},
              ValueRange{zeroIndex, zeroByte}, /*postCondition=*/false,
              [&](OpBuilder &builder, Location loc, Region &region) {
                cudaq::cc::RegionBuilderGuard guard(
                    builder, loc, region,
                    TypeRange{builder.getI64Type(), i8Ty});
                Block &block = *builder.getBlock();
                Value keepPacking = arith::CmpIOp::create(
                    builder, loc, arith::CmpIPredicate::slt,
                    block.getArgument(0), bitsInByte);
                cudaq::cc::ConditionOp::create(builder, loc, keepPacking,
                                               block.getArguments());
              },
              [&](OpBuilder &builder, Location loc, Region &region) {
                cudaq::cc::RegionBuilderGuard guard(
                    builder, loc, region,
                    TypeRange{builder.getI64Type(), i8Ty});
                Block &block = *builder.getBlock();
                Value bitIndex = block.getArgument(0);
                Value packedByte = block.getArgument(1);
                Value logicalIndex =
                    arith::AddIOp::create(builder, loc, byteStart, bitIndex);
                Value elementPtr = cudaq::cc::ComputePtrOp::create(
                    builder, loc, sourcePtrTy, data,
                    ArrayRef<cudaq::cc::ComputePtrArg>{logicalIndex});
                Value element =
                    cudaq::cc::LoadOp::create(builder, loc, elementPtr)
                        .getResult();
                if (isa<cudaq::cc::MeasureHandleType>(sourceElementTy))
                  element = cudaq::quake::DiscriminateOp::create(
                      builder, loc, builder.getI1Type(), element);
                Value bit = arith::ExtUIOp::create(builder, loc, i8Ty, element);
                Value shift =
                    arith::TruncIOp::create(builder, loc, i8Ty, bitIndex);
                Value shifted = arith::ShLIOp::create(builder, loc, bit, shift);
                Value nextPackedByte =
                    arith::OrIOp::create(builder, loc, packedByte, shifted);
                cudaq::cc::ContinueOp::create(
                    builder, loc, ValueRange{bitIndex, nextPackedByte});
              },
              [&](OpBuilder &builder, Location loc, Region &region) {
                cudaq::cc::RegionBuilderGuard guard(
                    builder, loc, region,
                    TypeRange{builder.getI64Type(), i8Ty});
                Block &block = *builder.getBlock();
                Value nextBit = arith::AddIOp::create(
                    builder, loc, block.getArgument(0), one);
                cudaq::cc::ContinueOp::create(
                    builder, loc, ValueRange{nextBit, block.getArgument(1)});
              });
          packLoop->setAttr("invariant", builder.getUnitAttr());
          Value dstOffset =
              arith::AddIOp::create(builder, loc, cursor, byteIndex);
          Value dstPtr = bytePtrAt(builder, loc, requestBuffer, dstOffset);
          cudaq::cc::StoreOp::create(builder, loc, packLoop.getResult(1),
                                     dstPtr);
        });
    return arith::AddIOp::create(builder, loc, cursor, packedBytes);
  }

  // All remaining supported payload element types (i8, i32, f32, f64) have
  // the same layout as the source elements, so marshal the whole array into
  // the request payload with a single memcpy. This preserves the source
  // machine's native element layout and assumes the remote endpoint uses a
  // compatible representation.
  // TODO: Consider layout canonicalization for remote endpoints with an
  // incompatible native representation. The data is currently copied as-is,
  // which is compatible with NVIDIA GPUs in shared-memory mode. For remote
  // execution, canonicalization could be performed by this lowering, by the
  // host realtime runtime before transmission, or by the remote endpoint while
  // decoding the payload.
  // Note: The common CUDA host architectures (`x86-64`, `AArch64`, and
  // `ppc64le`) and NVIDIA GPUs are all little-endian, so no byte-order
  // conversion is required for those combinations.
  Value arrayBytes =
      computeArrayPayloadSize(builder, loc, arrayLength, payloadElementTy);
  Value dstPtr = bytePtrAt(builder, loc, requestBuffer, cursor);
  Value srcPtr = cudaq::cc::CastOp::create(
      builder, loc, cudaq::cc::PointerType::get(i8Ty), data);
  Value notVolatile = arith::ConstantIntOp::create(builder, loc, 0, 1);
  func::CallOp::create(builder, loc, TypeRange{}, cudaq::llvmMemCopyIntrinsic,
                       ValueRange{dstPtr, srcPtr, arrayBytes, notVolatile});
  return arith::AddIOp::create(builder, loc, cursor, arrayBytes);
}

/// Copy the raw response payload into the output vector's storage. This uses
/// the same compatible-native-layout assumption as request marshalling, so the
/// response bytes can be copied directly into the vector's element storage.
static void copyResponseToStdvec(OpBuilder &builder, Location loc,
                                 Value responseBuffer, Value responseLen,
                                 Value stdvec) {
  auto ptrI8Ty = cudaq::cc::PointerType::get(builder.getI8Type());
  Value data = getStdvecData(builder, loc, stdvec);
  Value dstPtr = cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, data);
  Value srcPtr =
      cudaq::cc::CastOp::create(builder, loc, ptrI8Ty, responseBuffer);
  Value notVolatile = arith::ConstantIntOp::create(builder, loc, 0, 1);
  func::CallOp::create(builder, loc, TypeRange{}, cudaq::llvmMemCopyIntrinsic,
                       ValueRange{dstPtr, srcPtr, responseLen, notVolatile});
}

static void unpackResponseToBoolStdvec(OpBuilder &builder, Location loc,
                                       Value responseBuffer,
                                       Value logicalLength, Value stdvec) {
  auto i8Ty = builder.getI8Type();
  Type elementTy = getStdvecSourceElementType(stdvec);
  assert(elementTy.isInteger(1));
  Value data = getStdvecData(builder, loc, stdvec);
  auto elementPtrTy = cudaq::cc::PointerType::get(elementTy);
  Value packedBytes =
      computeArrayPayloadSize(builder, loc, logicalLength, elementTy);
  Value eight = i64Constant(builder, loc, 8);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 8);

  // Load each response byte once, then extract all of its logical bits. The
  // byte-oriented outer loop avoids reloading the same packed byte for every
  // destination bool while preserving the canonical LSB-first wire order.
  //
  // Pseudocode for the generated loops:
  //   for (byte = 0; byte < packedBytes; ++byte) {
  //     packed = responseBuffer[byte];
  //     bits = min(logicalLength - byte * 8, 8);
  //     for (bit = 0; bit < bits; ++bit)
  //       data[byte * 8 + bit] = (packed >> bit) & 1;
  //   }
  cudaq::opt::factory::createInvariantLoop(
      builder, loc, packedBytes,
      [&](OpBuilder &builder, Location loc, Region &, Block &block) {
        Value byteIndex = block.getArgument(0);
        Value byteStart = arith::MulIOp::create(builder, loc, byteIndex, eight);
        Value remaining =
            arith::SubIOp::create(builder, loc, logicalLength, byteStart);
        Value isPartial = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::ult, remaining, eight);
        Value bitsInByte = arith::SelectOp::create(
            builder, loc, builder.getI64Type(), isPartial, remaining, eight);
        Value srcPtr = bytePtrAt(builder, loc, responseBuffer, byteIndex);
        Value packedByte =
            cudaq::cc::LoadOp::create(builder, loc, srcPtr).getResult();
        cudaq::opt::factory::createInvariantLoop(
            builder, loc, bitsInByte,
            [&](OpBuilder &builder, Location loc, Region &, Block &block) {
              Value bitIndex = block.getArgument(0);
              Value logicalIndex =
                  arith::AddIOp::create(builder, loc, byteStart, bitIndex);
              Value shift =
                  arith::TruncIOp::create(builder, loc, i8Ty, bitIndex);
              Value shifted =
                  arith::ShRUIOp::create(builder, loc, packedByte, shift);
              Value masked = arith::AndIOp::create(builder, loc, shifted, one);
              Value bit = arith::TruncIOp::create(builder, loc,
                                                  builder.getI1Type(), masked);
              Value dstPtr = cudaq::cc::ComputePtrOp::create(
                  builder, loc, elementPtrTy, data,
                  ArrayRef<cudaq::cc::ComputePtrArg>{logicalIndex});
              cudaq::cc::StoreOp::create(builder, loc, bit, dstPtr);
            });
      });
}

/// Emit a call to the realtime frame release ABI:
///   extern "C" void __cudaq_device_call_release_realtime_frame(
///       void *frame_handle);
/// This lowering uses the non-throwing release wrapper with the same argument
/// shape.
static void emitReleaseFrame(OpBuilder &builder, Location loc,
                             Value frameHandle) {
  func::CallOp::create(builder, loc, TypeRange{},
                       cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
                       ValueRange{frameHandle});
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
        if (frameHandle)
          emitReleaseFrame(builder, loc, frameHandle);
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
            i64Constant(rewriter, loc, realtimePrimitiveSize(resultTy));
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
        Value data = getStdvecData(rewriter, loc, arg);
        cursor =
            marshalArrayArgument(rewriter, loc, cursor, requestBuffer, data,
                                 arrayLength, sourceElementTy, *elementTy);
        continue;
      }
      // Pad the cursor to the scalar's natural alignment before storing it.
      // No-op for i1/i8 args.
      std::uint64_t scalarBytes = realtimePrimitiveSize(arg.getType());
      cursor = alignOffsetTo(rewriter, loc, cursor, scalarBytes);
      Value typedArgPtr =
          typedPtrAt(rewriter, loc, requestBuffer, cursor, arg.getType());
      cudaq::cc::StoreOp::create(rewriter, loc, arg, typedArgPtr);
      cursor = arith::AddIOp::create(rewriter, loc, cursor,
                                     i64Constant(rewriter, loc, scalarBytes));
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
        Value outVector = args[*responseStdvec];
        Type elementTy =
            *getSupportedRealtimeStdvecResultElement(outVector.getType());
        if (elementTy.isInteger(1)) {
          Value outLength = getStdvecLength(rewriter, loc, outVector);
          unpackResponseToBoolStdvec(rewriter, loc, responseBuffer, outLength,
                                     outVector);
        } else {
          copyResponseToStdvec(rewriter, loc, responseBuffer, responseCapacity,
                               outVector);
        }
      }
      emitReleaseFrame(rewriter, loc, frameHandle);
      rewriter.eraseOp(devcall);
      return success();
    }

    // Direct scalar result case
    // The realtime lowering currently admits at most one scalar result, stored
    // at the start of the response payload. A successful dispatch must have
    // returned exactly sizeof(result) bytes; trap on a malformed response
    // rather than reading garbage payload bytes.
    Value returnedLen = cudaq::cc::LoadOp::create(rewriter, loc, responseLen);
    auto unexpectedResponseLen = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ne, returnedLen, responseCapacity);
    auto remoteError = arith::ConstantIntOp::create(
        rewriter, loc, RealtimeRemoteErrorStatus, 32);
    emitTrapOnCondition(rewriter, loc, unexpectedResponseLen, remoteError,
                        frameHandle);
    auto resultPtr = cudaq::cc::CastOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(resultTy), responseBuffer);
    auto result = cudaq::cc::LoadOp::create(rewriter, loc, resultPtr);
    emitReleaseFrame(rewriter, loc, frameHandle);
    rewriter.replaceOp(devcall, ValueRange{result.getResult()});
    return success();
  }
};

class RealtimeDeviceCallPass
    : public cudaq::opt::impl::RealtimeDeviceCallBase<RealtimeDeviceCallPass> {
public:
  using RealtimeDeviceCallBase::RealtimeDeviceCallBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ModuleOp module = getOperation();
    bool hasDeviceCalls = false;
    if (failed(validateRealtimeDeviceCalls(module, hasDeviceCalls))) {
      signalPassFailure();
      return;
    }
    // Do not inject the runtime ABI declarations into modules that have no
    // device calls to lower.
    if (!hasDeviceCalls)
      return;

    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    for (const char *name :
         {cudaq::runtime::deviceCallAcquireRealtimeFrame,
          cudaq::runtime::deviceCallDispatchRealtimeFrame,
          cudaq::runtime::deviceCallSafelyReleaseRealtimeFrame,
          cudaq::llvmMemCopyIntrinsic}) {
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
  }
};
} // namespace
