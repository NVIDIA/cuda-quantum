/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuakeToCodegen.h"
#include "CodeGenOps.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// Code generation: converts the Quake IR to Codegen IR.
//===----------------------------------------------------------------------===//

class CodeGenRAIIPattern
    : public OpConversionPattern<cudaq::quake::InitializeStateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::InitializeStateOp init, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value mem = adaptor.getTargets();
    auto alloc = mem.getDefiningOp<cudaq::quake::AllocaOp>();
    if (!alloc)
      return init.emitOpError("init_state must have alloca as input");
    rewriter.replaceOpWithNewOp<cudaq::codegen::RAIIOp>(
        init, init.getType(), adaptor.getState(),
        cast<cudaq::cc::PointerType>(adaptor.getState().getType())
            .getElementType(),
        alloc.getType(), alloc.getSize());
    rewriter.eraseOp(alloc);
    return success();
  }
};

class ExpandComplexCast : public OpConversionPattern<cudaq::cc::CastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CastOp castOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto complexTy = dyn_cast<ComplexType>(castOp.getType());
    if (!complexTy)
      return failure();
    auto loc = castOp.getLoc();
    Value val = adaptor.getValue();
    auto ty = cast<ComplexType>(val.getType()).getElementType();
    Value rePart = complex::ReOp::create(rewriter, loc, ty, val);
    Value imPart = complex::ImOp::create(rewriter, loc, ty, val);
    auto eleTy = complexTy.getElementType();
    auto reCast = cudaq::cc::CastOp::create(rewriter, loc, eleTy, rePart);
    auto imCast = cudaq::cc::CastOp::create(rewriter, loc, eleTy, imPart);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(castOp, complexTy, reCast,
                                                   imCast);
    return success();
  }
};

class CreateStateOpPattern
    : public OpConversionPattern<cudaq::quake::CreateStateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::CreateStateOp createStateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = createStateOp->getParentOfType<ModuleOp>();
    auto loc = createStateOp.getLoc();
    auto *ctx = createStateOp.getContext();
    Value buffer = adaptor.getData();
    Value size = adaptor.getLength();

    auto bufferTy = buffer.getType();
    auto ptrTy = cast<cudaq::cc::PointerType>(bufferTy);
    auto arrTy = dyn_cast<cudaq::cc::ArrayType>(ptrTy.getElementType());
    auto eleTy = arrTy ? arrTy.getElementType() : ptrTy.getElementType();
    bool is64Bit = isa<Float64Type>(eleTy);
    bool isComplex = false;

    if (auto cTy = dyn_cast<ComplexType>(eleTy)) {
      is64Bit = isa<Float64Type>(cTy.getElementType());
      isComplex = true;
    }

    auto createStateFunc = [&]() {
      if (isComplex) {
        if (is64Bit)
          return cudaq::createCudaqStateFromDataComplexF64;
        return cudaq::createCudaqStateFromDataComplexF32;
      }
      if (is64Bit)
        return cudaq::createCudaqStateFromDataF64;
      return cudaq::createCudaqStateFromDataF32;
    }();

    cudaq::IRBuilder irBuilder(ctx);
    auto result = irBuilder.loadIntrinsic(module, createStateFunc);
    assert(succeeded(result) && "loading intrinsic should never fail");

    auto stateTy = cudaq::quake::StateType::get(ctx);
    auto statePtrTy = cudaq::cc::PointerType::get(stateTy);
    auto i8PtrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    auto cast = cudaq::cc::CastOp::create(rewriter, loc, i8PtrTy, buffer);

    rewriter.replaceOpWithNewOp<func::CallOp>(
        createStateOp, statePtrTy, createStateFunc, ValueRange{cast, size});
    return success();
  }
};

class DeleteStateOpPattern
    : public OpConversionPattern<cudaq::quake::DeleteStateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::DeleteStateOp deleteStateOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = deleteStateOp->getParentOfType<ModuleOp>();
    auto *ctx = deleteStateOp.getContext();
    Value state = adaptor.getState();

    cudaq::IRBuilder irBuilder(ctx);
    auto result = irBuilder.loadIntrinsic(module, cudaq::deleteCudaqState);
    assert(succeeded(result) && "loading intrinsic should never fail");

    rewriter.replaceOpWithNewOp<func::CallOp>(deleteStateOp, mlir::TypeRange{},
                                              cudaq::deleteCudaqState,
                                              mlir::ValueRange{state});
    return success();
  }
};

class GetNumberOfQubitsOpPattern
    : public OpConversionPattern<cudaq::quake::GetNumberOfQubitsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::GetNumberOfQubitsOp getNumQubitsOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = getNumQubitsOp->getParentOfType<ModuleOp>();
    auto ctx = getNumQubitsOp.getContext();
    Value state = adaptor.getState();

    cudaq::IRBuilder irBuilder(ctx);
    auto result =
        irBuilder.loadIntrinsic(module, cudaq::getNumQubitsFromCudaqState);
    assert(succeeded(result) && "loading intrinsic should never fail");

    rewriter.replaceOpWithNewOp<func::CallOp>(
        getNumQubitsOp, rewriter.getI64Type(),
        cudaq::getNumQubitsFromCudaqState, state);
    return success();
  }
};

} // namespace

void cudaq::codegen::populateQuakeToCodegenPatterns(
    mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns
      .insert<CodeGenRAIIPattern, CreateStateOpPattern, DeleteStateOpPattern,
              ExpandComplexCast, GetNumberOfQubitsOpPattern>(ctx);
}

void cudaq::codegen::setQuakeToCodegenLegality(ConversionTarget &target) {
  target
      .addLegalDialect<cudaq::codegen::CodeGenDialect, complex::ComplexDialect,
                       cudaq::cc::CCDialect, func::FuncDialect>();
  target.addIllegalOp<cudaq::quake::InitializeStateOp,
                      cudaq::quake::CreateStateOp, cudaq::quake::DeleteStateOp,
                      cudaq::quake::GetNumberOfQubitsOp>();
  target.addDynamicallyLegalOp<cudaq::cc::CastOp>([](cudaq::cc::CastOp castOp) {
    return !isa<ComplexType>(castOp.getType());
  });
}
