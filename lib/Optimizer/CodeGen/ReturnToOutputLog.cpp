/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "return-to-output-log"

namespace cudaq::opt {
#define GEN_PASS_DEF_RETURNTOOUTPUTLOG
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
class ReturnRewrite : public OpRewritePattern<cudaq::cc::LogOutputOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // This is where the heavy lifting is done. We take the return op's operand(s)
  // and convert them to calls to the QIR output logging functions with the
  // appropriate label information.
  LogicalResult matchAndRewrite(cudaq::cc::LogOutputOp log,
                                PatternRewriter &rewriter) const override {
    auto loc = log.getLoc();
    // For each operand, generate a QIR logging call.
    for (auto operand : log.getOperands())
      genOutputLog(loc, rewriter, operand, std::nullopt);
    rewriter.eraseOp(log);
    return success();
  }

  static void genOutputLog(Location loc, PatternRewriter &rewriter, Value val,
                           std::optional<StringRef> prefix,
                           std::optional<Value> customLabel = std::nullopt) {
    Type valTy = val.getType();
    TypeSwitch<Type>(valTy)
        .Case([&](IntegerType intTy) {
          int width = intTy.getWidth();
          std::string labelStr = std::string("i") + std::to_string(width);
          if (prefix)
            labelStr = prefix->str();
          Value label =
              customLabel.value_or(makeLabel(loc, rewriter, labelStr));
          if (intTy.getWidth() == 1) {
            rewriter.create<func::CallOp>(loc, TypeRange{},
                                          cudaq::opt::QIRBoolRecordOutput,
                                          ArrayRef<Value>{val, label});
            return;
          }
          // Integer: convert to (signed) i64. The decoder *must* lop off any
          // higher-order bits added by the sign-extension to get this to 64
          // bits by examining the real integer type.
          Value castVal = val;
          if (intTy.getWidth() < 64)
            castVal = rewriter.create<cudaq::cc::CastOp>(
                loc, rewriter.getI64Type(), val, cudaq::cc::CastOpMode::Signed);
          else if (intTy.getWidth() > 64)
            castVal = rewriter.create<cudaq::cc::CastOp>(
                loc, rewriter.getI64Type(), val);
          rewriter.create<func::CallOp>(loc, TypeRange{},
                                        cudaq::opt::QIRIntegerRecordOutput,
                                        ArrayRef<Value>{castVal, label});
        })
        .Case([&](FloatType floatTy) {
          int width = floatTy.getWidth();
          std::string labelStr = std::string("f") + std::to_string(width);
          if (prefix)
            labelStr = prefix->str();
          Value label =
              customLabel.value_or(makeLabel(loc, rewriter, labelStr));
          // Floating point: convert it to double, whatever it actually is.
          Value castVal = val;
          if (floatTy != rewriter.getF64Type())
            castVal = rewriter.create<cudaq::cc::CastOp>(
                loc, rewriter.getF64Type(), val);
          rewriter.create<func::CallOp>(loc, TypeRange{},
                                        cudaq::opt::QIRDoubleRecordOutput,
                                        ArrayRef<Value>{castVal, label});
        })
        .Case([&](cudaq::cc::StructType structTy) {
          auto labelStr = translateType(structTy);
          if (prefix)
            labelStr = prefix->str();
          Value label =
              customLabel.value_or(makeLabel(loc, rewriter, labelStr));
          std::int32_t sz = structTy.getNumMembers();
          Value size = rewriter.create<arith::ConstantIntOp>(loc, sz, 64);
          rewriter.create<func::CallOp>(loc, TypeRange{},
                                        cudaq::opt::QIRTupleRecordOutput,
                                        ArrayRef<Value>{size, label});
          std::string preStr = prefix ? prefix->str() : std::string{};
          for (std::int32_t i = 0; i < sz; ++i) {
            std::string offset = preStr + std::string(".") + std::to_string(i);
            Value w = rewriter.create<cudaq::cc::ExtractValueOp>(
                loc, structTy.getMember(i), val,
                ArrayRef<cudaq::cc::ExtractValueArg>{i});
            genOutputLog(loc, rewriter, w, offset);
          }
        })
        .Case([&](cudaq::cc::ArrayType arrTy) {
          auto labelStr = translateType(arrTy);
          Value label =
              customLabel.value_or(makeLabel(loc, rewriter, labelStr));
          std::int32_t sz = arrTy.getSize();
          Value size = rewriter.create<arith::ConstantIntOp>(loc, sz, 64);
          rewriter.create<func::CallOp>(loc, TypeRange{},
                                        cudaq::opt::QIRArrayRecordOutput,
                                        ArrayRef<Value>{size, label});
          std::string preStr = prefix ? prefix->str() : std::string{};
          for (std::int32_t i = 0; i < sz; ++i) {
            std::string offset = preStr + std::string("[") + std::to_string(i) +
                                 std::string("]");
            Value w = rewriter.create<cudaq::cc::ExtractValueOp>(
                loc, arrTy.getElementType(), val,
                ArrayRef<cudaq::cc::ExtractValueArg>{i});
            genOutputLog(loc, rewriter, w, offset);
          }
        })
        .Case([&](cudaq::cc::StdvecType vecTy) {
          if (auto vecInit = val.getDefiningOp<cudaq::cc::StdvecInitOp>())
            if (auto maybeLen = cudaq::opt::factory::maybeValueOfIntConstant(
                    vecInit.getLength())) {
              // For this type, we expect a cc.stdvec_init operation as the
              // input.
              // The data will be in a variable.
              std::int32_t sz = *maybeLen;
              auto labelStr = translateType(vecTy, sz);
              Value label = makeLabel(loc, rewriter, labelStr);
              Value size = rewriter.create<arith::ConstantIntOp>(loc, sz, 64);
              rewriter.create<func::CallOp>(loc, TypeRange{},
                                            cudaq::opt::QIRArrayRecordOutput,
                                            ArrayRef<Value>{size, label});
              std::string preStr = prefix ? prefix->str() : std::string{};
              Value rawBuffer = vecInit.getBuffer();
              auto eleTy = vecTy.getElementType();
              auto buffTy = cudaq::cc::PointerType::get(eleTy);
              auto ptrArrTy =
                  cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(eleTy));
              Value buffer =
                  rewriter.create<cudaq::cc::CastOp>(loc, ptrArrTy, rawBuffer);
              for (std::int32_t i = 0; i < sz; ++i) {
                std::string offset = preStr + std::string("[") +
                                     std::to_string(i) + std::string("]");
                auto v = rewriter.create<cudaq::cc::ComputePtrOp>(
                    loc, buffTy, buffer, ArrayRef<cudaq::cc::ComputePtrArg>{i});
                Value w = rewriter.create<cudaq::cc::LoadOp>(loc, v);
                genOutputLog(loc, rewriter, w, offset);
              }
              return;
            }

          // If we reach here and we cannot determine the constant size of the
          // buffer, then we will generate dynamic output logging with a for
          // loop.
          Value vecSz = rewriter.template create<cudaq::cc::StdvecSizeOp>(
              loc, rewriter.getI64Type(), val);
          const std::string arrayLabelPrefix =
              "array<" + translateType(vecTy.getElementType()) + " x ";
          Value labelBuffer =
              makeLabel(loc, rewriter, arrayLabelPrefix, vecSz, ">");
          rewriter.create<func::CallOp>(loc, TypeRange{},
                                        cudaq::opt::QIRArrayRecordOutput,
                                        ArrayRef<Value>{vecSz, labelBuffer});
          auto eleTy = vecTy.getElementType();
          const bool isBool = (eleTy == rewriter.getI1Type());
          if (isBool)
            eleTy = rewriter.getI8Type();
          auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
          auto eleArrTy =
              cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(eleTy));
          auto vecPtr =
              rewriter.create<cudaq::cc::StdvecDataOp>(loc, eleArrTy, val);
          const std::string preStr = prefix ? prefix->str() : std::string{};
          cudaq::opt::factory::createInvariantLoop(
              rewriter, loc, vecSz,
              [&](OpBuilder &builder, Location loc, Region &, Block &block) {
                Value indexVar = block.getArgument(0);
                auto eleAddr = rewriter.create<cudaq::cc::ComputePtrOp>(
                    loc, elePtrTy, vecPtr, ValueRange{indexVar});

                Value w = [&]() {
                  if (isBool) {
                    auto i1PtrTy =
                        cudaq::cc::PointerType::get(rewriter.getI1Type());
                    auto i1Cast = rewriter.create<cudaq::cc::CastOp>(
                        loc, i1PtrTy, eleAddr);
                    return rewriter.create<cudaq::cc::LoadOp>(loc, i1Cast);
                  }

                  return rewriter.create<cudaq::cc::LoadOp>(loc, eleAddr);
                }();
                const std::string prefix = preStr + "[";
                const std::string postfix = "]";
                Value dynamicLabel =
                    makeLabel(loc, rewriter, prefix, indexVar, postfix);
                genOutputLog(loc, rewriter, w, std::nullopt, dynamicLabel);
              });
        })
        .Default([&](Type) {
          // If we reach here, we don't know how to handle this type.
          Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
          rewriter.create<func::CallOp>(loc, TypeRange{}, cudaq::opt::QISTrap,
                                        ValueRange{one});
        });
  }

  static std::string
  translateType(Type ty, std::optional<std::int32_t> vecSz = std::nullopt) {
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      int width = intTy.getWidth();
      return {std::string("i") + std::to_string(width)};
    }
    if (auto floatTy = dyn_cast<FloatType>(ty)) {
      int width = floatTy.getWidth();
      return {std::string("f") + std::to_string(width)};
    }
    if (auto structTy = dyn_cast<cudaq::cc::StructType>(ty)) {
      std::string result = "tuple<";
      if (structTy.getMembers().empty())
        return {result + std::string(">")};
      result += translateType(structTy.getMembers().front());
      for (auto memTy : structTy.getMembers().drop_front())
        result += std::string(", ") + translateType(memTy);
      return {result + std::string(">")};
    }
    if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(ty)) {
      std::int32_t size = arrTy.getSize();
      return {std::string("array<") + translateType(arrTy.getElementType()) +
              std::string(" x ") + std::to_string(size) + std::string(">")};
    }
    if (auto arrTy = dyn_cast<cudaq::cc::StdvecType>(ty))
      return {std::string("array<") + translateType(arrTy.getElementType()) +
              std::string(" x ") + std::to_string(*vecSz) + std::string(">")};
    return {"error"};
  }

  static Value makeLabel(Location loc, PatternRewriter &rewriter,
                         StringRef label) {
    auto strLitTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(
        rewriter.getContext(), rewriter.getI8Type(), label.size() + 1));
    Value lit = rewriter.create<cudaq::cc::CreateStringLiteralOp>(
        loc, strLitTy, rewriter.getStringAttr(label));
    auto i8PtrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    return rewriter.create<cudaq::cc::CastOp>(loc, i8PtrTy, lit);
  }

  static Value makeLabel(Location loc, PatternRewriter &rewriter,
                         const std::string &prefix, Value val,
                         const std::string &postFix) {
    auto i64Ty = rewriter.getI64Type();
    auto i8Ty = rewriter.getI8Type();
    auto i8PtrTy = cudaq::cc::PointerType::get(i8Ty);
    // Value must be i64
    if (val.getType() != i64Ty)
      val = rewriter.create<cudaq::cc::CastOp>(loc, i64Ty, val);
    // Compute the number of digits required
    Value numDigits = rewriter
                          .create<func::CallOp>(
                              loc, i64Ty, "__nvqpp_internal_number_of_digits",
                              ArrayRef<Value>{val})
                          .getResult(0);
    // Allocate a <i8 x 32> buffer
    auto bufferSize = rewriter.create<arith::ConstantIntOp>(loc, 32, 64);
    auto buffer = rewriter.create<cudaq::cc::AllocaOp>(loc, i8Ty, bufferSize);
    rewriter.create<func::CallOp>(loc, TypeRange{}, "__nvqpp_internal_tostring",
                                  ArrayRef<Value>{buffer, val});
    auto valStrBuf = rewriter.create<cudaq::cc::CastOp>(loc, i8PtrTy, buffer);
    Value arrayPrefix = makeLabel(loc, rewriter, prefix);
    Value arrayPostfix = makeLabel(loc, rewriter, postFix);
    const int preFixLen = prefix.size();
    const int postFixLen = postFix.size();
    Value totalStrSize = rewriter.create<arith::AddIOp>(
        loc, numDigits,
        rewriter.create<arith::ConstantIntOp>(loc, preFixLen + postFixLen + 1,
                                              64));
    auto labelBufferAlloc =
        rewriter.create<cudaq::cc::AllocaOp>(loc, i8Ty, totalStrSize);
    Value labelBuffer =
        rewriter.create<cudaq::cc::CastOp>(loc, i8PtrTy, labelBufferAlloc);

    // Copy the prefix
    rewriter.create<func::CallOp>(
        loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
        ValueRange{labelBuffer, arrayPrefix,
                   rewriter.create<arith::ConstantIntOp>(loc, preFixLen, 64),
                   rewriter.create<arith::ConstantIntOp>(loc, 0, 1)});
    // Copy the integer string
    auto toPtr = rewriter.create<cudaq::cc::ComputePtrOp>(
        loc, i8PtrTy, labelBufferAlloc,
        ValueRange{rewriter.create<arith::ConstantIntOp>(loc, preFixLen, 64)});
    rewriter.create<func::CallOp>(
        loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
        ValueRange{toPtr, valStrBuf, numDigits,
                   rewriter.create<arith::ConstantIntOp>(loc, 0, 1)});
    // Copy the postfix + null terminator
    Value shift = rewriter.create<arith::AddIOp>(
        loc, numDigits,
        rewriter.create<arith::ConstantIntOp>(loc, preFixLen, 64));
    toPtr = rewriter.create<cudaq::cc::ComputePtrOp>(
        loc, i8PtrTy, labelBufferAlloc, ValueRange{shift});
    rewriter.create<func::CallOp>(
        loc, std::nullopt, cudaq::llvmMemCopyIntrinsic,
        ValueRange{
            toPtr, arrayPostfix,
            rewriter.create<arith::ConstantIntOp>(loc, postFixLen + 1, 64),
            rewriter.create<arith::ConstantIntOp>(loc, 0, 1)});
    return labelBuffer;
  }
};

struct ReturnToOutputLogPass
    : public cudaq::opt::impl::ReturnToOutputLogBase<ReturnToOutputLogPass> {
  using ReturnToOutputLogBase::ReturnToOutputLogBase;

  void runOnOperation() override {
    auto module = getOperation();

    auto *ctx = &getContext();
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (failed(irBuilder.loadIntrinsic(module,
                                       cudaq::opt::QIRArrayRecordOutput))) {
      module.emitError("could not load QIR output logging functions.");
      signalPassFailure();
      return;
    }
    if (failed(irBuilder.loadIntrinsic(module, cudaq::opt::QISTrap))) {
      module.emitError("could not load QIR trap function.");
      signalPassFailure();
      return;
    }

    if (failed(irBuilder.loadIntrinsic(module, "__nvqpp_internal_tostring"))) {
      module.emitError("could not load string conversion function.");
      signalPassFailure();
      return;
    }

    if (failed(irBuilder.loadIntrinsic(module,
                                       "__nvqpp_internal_number_of_digits"))) {
      module.emitError("could not load number of digits function.");
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(ctx);
    patterns.insert<ReturnRewrite>(ctx);
    LLVM_DEBUG(llvm::dbgs() << "Before return to output logging:\n" << module);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After return to output logging:\n" << module);
  }
};

} // namespace
