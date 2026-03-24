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
                           std::optional<StringRef> prefix) {
    Type valTy = val.getType();
    TypeSwitch<Type>(valTy)
        .Case([&](IntegerType intTy) {
          int width = intTy.getWidth();
          std::string labelStr = std::string("i") + std::to_string(width);
          if (prefix)
            labelStr = prefix->str();
          Value label = makeLabel(loc, rewriter, labelStr);
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
          Value label = makeLabel(loc, rewriter, labelStr);
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
          Value label = makeLabel(loc, rewriter, labelStr);
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
          Value label = makeLabel(loc, rewriter, labelStr);
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
          // For this type, we expect a cc.stdvec_init operation as the input.
          // The data will be in a variable.
          // If we reach here and we cannot determine the constant size of the
          // buffer, then we will not generate any output logging.
          if (auto vecInit = val.getDefiningOp<cudaq::cc::StdvecInitOp>())
            if (auto maybeLen = cudaq::opt::factory::maybeValueOfIntConstant(
                    vecInit.getLength())) {
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
            }
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

    RewritePatternSet patterns(ctx);
    patterns.insert<ReturnRewrite>(ctx);
    LLVM_DEBUG(llvm::dbgs() << "Before return to output logging:\n" << module);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After return to output logging:\n" << module);
  }
};

} // namespace
