/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuakeToCodegen.h"
#include "CodeGenOps.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Complex/IR/Complex.h"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// Code generation: converts the Quake IR to Codegen IR.
//===----------------------------------------------------------------------===//

class CodeGenRAIIPattern : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp init,
                                PatternRewriter &rewriter) const override {
    Value mem = init.getTargets();
    auto alloc = mem.getDefiningOp<quake::AllocaOp>();
    if (!alloc)
      return init.emitOpError("init_state must have alloca as input");
    rewriter.replaceOpWithNewOp<cudaq::codegen::RAIIOp>(
        init, init.getType(), init.getState(),
        cast<cudaq::cc::PointerType>(init.getState().getType())
            .getElementType(),
        alloc.getType(), alloc.getSize());
    rewriter.eraseOp(alloc);
    return success();
  }
};

class ExpandComplexCast : public OpRewritePattern<cudaq::cc::CastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::CastOp castOp,
                                PatternRewriter &rewriter) const override {
    auto complexTy = dyn_cast<ComplexType>(castOp.getType());
    if (!complexTy)
      return failure();
    auto loc = castOp.getLoc();
    auto ty = cast<ComplexType>(castOp.getValue().getType()).getElementType();
    Value rePart = rewriter.create<complex::ReOp>(loc, ty, castOp.getValue());
    Value imPart = rewriter.create<complex::ImOp>(loc, ty, castOp.getValue());
    auto eleTy = complexTy.getElementType();
    auto reCast = rewriter.create<cudaq::cc::CastOp>(loc, eleTy, rePart);
    auto imCast = rewriter.create<cudaq::cc::CastOp>(loc, eleTy, imPart);
    rewriter.replaceOpWithNewOp<complex::CreateOp>(castOp, complexTy, reCast,
                                                   imCast);
    return success();
  }
};

class CreateStateOpPattern : public OpRewritePattern<quake::CreateStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::CreateStateOp createStateOp,
                                PatternRewriter &rewriter) const override {
    auto module = createStateOp->getParentOfType<ModuleOp>();
    auto loc = createStateOp.getLoc();
    auto ctx = createStateOp.getContext();
    auto buffer = createStateOp.getOperand(0);
    auto size = createStateOp.getOperand(1);

    auto bufferTy = buffer.getType();
    auto ptrTy = cast<cudaq::cc::PointerType>(bufferTy);
    auto arrTy = dyn_cast<cudaq::cc::ArrayType>(ptrTy.getElementType());
    auto eleTy = arrTy ? arrTy.getElementType() : ptrTy.getElementType();
    auto is64Bit = isa<Float64Type>(eleTy);
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

    auto stateTy = quake::StateType::get(ctx);
    auto statePtrTy = cudaq::cc::PointerType::get(stateTy);
    auto i8PtrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    auto cast = rewriter.create<cudaq::cc::CastOp>(loc, i8PtrTy, buffer);

    rewriter.replaceOpWithNewOp<func::CallOp>(
        createStateOp, statePtrTy, createStateFunc, ValueRange{cast, size});
    return success();
  }
};

class DeleteStateOpPattern : public OpRewritePattern<quake::DeleteStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::DeleteStateOp deleteStateOp,
                                PatternRewriter &rewriter) const override {
    auto module = deleteStateOp->getParentOfType<ModuleOp>();
    auto ctx = deleteStateOp.getContext();
    auto state = deleteStateOp.getOperand();

    cudaq::IRBuilder irBuilder(ctx);
    auto result = irBuilder.loadIntrinsic(module, cudaq::deleteCudaqState);
    assert(succeeded(result) && "loading intrinsic should never fail");

    rewriter.replaceOpWithNewOp<func::CallOp>(deleteStateOp, std::nullopt,
                                              cudaq::deleteCudaqState,
                                              mlir::ValueRange{state});
    return success();
  }
};

class GetNumberOfQubitsOpPattern
    : public OpRewritePattern<quake::GetNumberOfQubitsOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::GetNumberOfQubitsOp getNumQubitsOp,
                                PatternRewriter &rewriter) const override {
    auto module = getNumQubitsOp->getParentOfType<ModuleOp>();
    auto ctx = getNumQubitsOp.getContext();
    auto state = getNumQubitsOp.getOperand();

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
