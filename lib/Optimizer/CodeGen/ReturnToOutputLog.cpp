/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "return-to-output-log"

namespace cudaq::opt {
#define GEN_PASS_DEF_RETURNTOOUTPUTLOG
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {
class FuncSignature : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Simple type conversion: drop the result type on the floor.
  LogicalResult matchAndRewrite(func::FuncOp fn,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto inputTys = fn.getFunctionType().getInputs();
    auto funcTy = FunctionType::get(ctx, inputTys, {});
    rewriter.updateRootInPlace(fn, [&]() { fn.setFunctionType(funcTy); });
    return success();
  }
};

class FuncConstant : public OpRewritePattern<func::ConstantOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Simple type conversion: drop the result type on the floor.
  LogicalResult matchAndRewrite(func::ConstantOp con,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto inputTys = cast<FunctionType>(con.getType()).getInputs();
    auto funcTy = FunctionType::get(ctx, inputTys, {});
    auto val = con.getValue();
    rewriter.replaceOpWithNewOp<func::ConstantOp>(con, funcTy, val);
    return success();
  }
};

class ReturnRewrite : public OpRewritePattern<func::ReturnOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // This is where the heavy lifting is done. We take the return op's operand(s)
  // and convert them to calls to the QIR output logging functions with the
  // appropriate label information.
  LogicalResult matchAndRewrite(func::ReturnOp ret,
                                PatternRewriter &rewriter) const override {
    auto loc = ret.getLoc();
    // For each operand:
    for (auto operand : ret.getOperands())
      genOutputLog(loc, rewriter, operand, std::nullopt);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(ret);
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
        .Case([&](FloatType fltTy) {
          int width = fltTy.getWidth();
          std::string labelStr = std::string("f") + std::to_string(width);
          if (prefix)
            labelStr = prefix->str();
          Value label = makeLabel(loc, rewriter, labelStr);
          // Floating point: convert it to double, whatever it actually is.
          Value castVal = val;
          if (fltTy != rewriter.getF64Type())
            castVal = rewriter.create<cudaq::cc::CastOp>(
                loc, rewriter.getF64Type(), val);
          rewriter.create<func::CallOp>(loc, TypeRange{},
                                        cudaq::opt::QIRDoubleRecordOutput,
                                        ArrayRef<Value>{castVal, label});
        })
        .Case([&](cudaq::cc::StructType strTy) {
          auto labelStr = translateType(strTy);
          if (prefix)
            labelStr = prefix->str();
          Value label = makeLabel(loc, rewriter, labelStr);
          std::int32_t sz = strTy.getNumMembers();
          Value size = rewriter.create<arith::ConstantIntOp>(loc, sz, 64);
          rewriter.create<func::CallOp>(loc, TypeRange{},
                                        cudaq::opt::QIRTupleRecordOutput,
                                        ArrayRef<Value>{size, label});
          std::string preStr = prefix ? prefix->str() : std::string{};
          for (std::int32_t i = 0; i < sz; ++i) {
            std::string offset = preStr + std::string(".") + std::to_string(i);
            Value w = rewriter.create<cudaq::cc::ExtractValueOp>(
                loc, strTy.getMember(i), val,
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
              auto rawBuffer = vecInit.getBuffer();
              auto buffTy = cast<cudaq::cc::PointerType>(rawBuffer.getType());
              Type ptrArrTy = buffTy;
              if (!isa<cudaq::cc::ArrayType>(buffTy.getElementType()))
                ptrArrTy = cudaq::cc::PointerType::get(
                    cudaq::cc::ArrayType::get(buffTy.getElementType()));
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
        });
  }

  static std::string
  translateType(Type ty, std::optional<std::int32_t> vecSz = std::nullopt) {
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      int width = intTy.getWidth();
      return {std::string("i") + std::to_string(width)};
    }
    if (auto fltTy = dyn_cast<FloatType>(ty)) {
      int width = fltTy.getWidth();
      return {std::string("f") + std::to_string(width)};
    }
    if (auto strTy = dyn_cast<cudaq::cc::StructType>(ty)) {
      std::string result = "tuple<";
      if (strTy.getMembers().empty())
        return {result + std::string(">")};
      result += translateType(strTy.getMembers().front());
      for (auto memTy : strTy.getMembers().drop_front())
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

class CallRewrite : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // It should be a violation of the CUDA-Q spec to call an entry-point function
  // that returns a value from another entry-point function and use the result
  // value(s). Under a run context, no entry-point kernel will actually return a
  // value.
  LogicalResult
  matchAndRewrite(func::CallOp call, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = call.getLoc();
    rewriter.create<func::CallOp>(loc, TypeRange{}, call.getCallee(),
                                  adaptor.getOperands());
    SmallVector<Value> poisons;
    for (auto ty : call.getResultTypes())
      poisons.push_back(rewriter.create<cudaq::cc::PoisonOp>(loc, ty));
    rewriter.replaceOp(call, poisons);
    return success();
  }
};

class FuncPtrConvert : public OpConversionPattern<cudaq::cc::FuncToPtrOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::FuncToPtrOp fnptr, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cudaq::cc::FuncToPtrOp>(fnptr, fnptr.getType(),
                                                        adaptor.getFunc());
    return success();
  }
};

struct ReturnToOutputLogPass
    : public cudaq::opt::impl::ReturnToOutputLogBase<ReturnToOutputLogPass> {
  using ReturnToOutputLogBase::ReturnToOutputLogBase;

  void runOnOperation() override {
    auto module = getOperation();
    //if (!module->hasAttr(cudaq::runtime::enableCudaqRun))
    //  return;

    auto *ctx = &getContext();
    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());
    if (failed(irBuilder.loadIntrinsic(module, "qir_output_logging"))) {
      module.emitError("could not load QIR output logging declarations.");
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(ctx);
    patterns.insert<CallRewrite, FuncConstant, FuncPtrConvert, FuncSignature,
                    ReturnRewrite>(ctx);
    LLVM_DEBUG(llvm::dbgs() << "Before return to output logging:\n" << module);
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                           func::FuncDialect>();
    auto constantOpLegal = [&](func::ConstantOp con) {
      // Legal unless calling an entry-point function with a result.
      if (auto module = con->getParentOfType<ModuleOp>()) {
        auto val = con.getValue();
        if (auto fn = module.lookupSymbol<func::FuncOp>(val)) {
          auto fnTy = cast<FunctionType>(con.getResult().getType());
          return !fn->hasAttr(cudaq::entryPointAttrName) ||
                 fnTy.getResults().empty();
        }
      }
      return true;
    };
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp fn) {
      // Legal unless an entry-point function, with a body, that returns a
      // value.
      return fn.getBody().empty() || !fn->hasAttr(cudaq::entryPointAttrName) ||
             fn.getFunctionType().getResults().empty();
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp call) {
      // Legal unless calling an entry-point function with a result.
      if (auto module = call->getParentOfType<ModuleOp>()) {
        auto callee = call.getCallee();
        if (auto fn = module.lookupSymbol<func::FuncOp>(callee)) {
          return !fn->hasAttr(cudaq::entryPointAttrName) ||
                 call.getResults().empty();
        }
      }
      return true;
    });
    target.addDynamicallyLegalOp<func::ConstantOp>(constantOpLegal);
    target.addDynamicallyLegalOp<cudaq::cc::FuncToPtrOp>(
        [&](cudaq::cc::FuncToPtrOp funcPtr) {
          if (auto con = funcPtr.getFunc().getDefiningOp<func::ConstantOp>())
            return constantOpLegal(con);
          return true;
        });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp ret) {
      // Legal if return is not in an entry-point or does not return a value.
      if (auto fn = ret->getParentOfType<func::FuncOp>())
        return !fn->hasAttr(cudaq::entryPointAttrName) ||
               ret.getOperands().empty();
      return true;
    });
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After return to output logging:\n" << module);
  }
};

} // namespace
