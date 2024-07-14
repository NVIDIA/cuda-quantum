/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CONSTPROPCOMPLEX
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "const-prop-complex"

using namespace mlir;

namespace {

// Replace array ptr casts that throw away the size by a cast to element
// pointer.
//
//%1 = cc.cast %0 : (!cc.ptr<!cc.array<complex<f32> x 4>>) ->
//! cc.ptr<!cc.array<complex<f32> x ?>>
// ->
//%1 = cc.cast %0 : (!cc.ptr<!cc.array<complex<f32> x 4>>) ->
//! cc.ptr<complex<f32>>
class CastArrayPtrPattern : public OpRewritePattern<cudaq::cc::CastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::CastOp cast,
                                PatternRewriter &rewriter) const override {

    auto fromTy = cast.getOperand().getType();
    auto toTy = cast.getType();

    if (auto ptrFromTy = dyn_cast<cudaq::cc::PointerType>(fromTy)) {
      if (auto arrayFromTy =
              dyn_cast<cudaq::cc::ArrayType>(ptrFromTy.getElementType())) {
        if (auto ptrToTy = dyn_cast<cudaq::cc::PointerType>(toTy)) {
          if (auto arrayToTy =
                  dyn_cast<cudaq::cc::ArrayType>(ptrToTy.getElementType())) {
            if (arrayFromTy.getElementType() == arrayToTy.getElementType()) {
              auto eleTy = arrayFromTy.getElementType();
              auto elePtrType = cudaq::cc::PointerType::get(eleTy);
              rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(cast, elePtrType,
                                                             cast.getOperand());
              return success();
            }
          }
        }
      }
    }
    return failure();
  }
};

// Fold complex.create ops if the arguments are constants.
class ComplexCreatePattern : public OpRewritePattern<complex::CreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(complex::CreateOp create,
                                PatternRewriter &rewriter) const override {
    auto re = create.getReal();
    auto im = create.getImaginary();
    auto reCon = re.getDefiningOp<arith::ConstantOp>();
    auto imCon = im.getDefiningOp<arith::ConstantOp>();
    if (reCon && imCon) {
      auto aa = ArrayAttr::get(
          rewriter.getContext(),
          ArrayRef<Attribute>{reCon.getValue(), imCon.getValue()});
      rewriter.replaceOpWithNewOp<complex::ConstantOp>(create, create.getType(),
                                                       aa);
      return success();
    }
    return failure();
  }
};

// Fold arith.trunc ops if the argument is constant.
class FloatTruncatePattern : public OpRewritePattern<arith::TruncFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp truncate,
                                PatternRewriter &rewriter) const override {
    auto val = truncate.getOperand();
    auto valCon = val.getDefiningOp<arith::ConstantFloatOp>();
    if (valCon) {
      auto v = valCon.value().convertToDouble();
      auto fTy = dyn_cast<FloatType>(truncate.getType());
      rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(
          truncate, APFloat{static_cast<float>(v)}, fTy);
      return success();
    }
    return failure();
  }
};

// Fold arith.ext ops if the argument is constant.
class FloatExtendPattern : public OpRewritePattern<arith::ExtFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::ExtFOp extend,
                                PatternRewriter &rewriter) const override {
    auto val = extend.getOperand();
    auto valCon = val.getDefiningOp<arith::ConstantFloatOp>();
    if (valCon) {
      auto v = valCon.value().convertToFloat();
      auto fTy = dyn_cast<FloatType>(extend.getType());
      rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(
          extend, APFloat{static_cast<double>(v)}, fTy);
      return success();
    }
    return failure();
  }
};

// Fold complex.re ops if the argument is constant.
class ComplexRePattern : public OpRewritePattern<complex::ReOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(complex::ReOp re,
                                PatternRewriter &rewriter) const override {
    auto val = re.getOperand();
    auto valCon = val.getDefiningOp<complex::ConstantOp>();
    if (valCon) {
      auto attr = valCon.getValue();
      auto real = cast<FloatAttr>(attr[0]).getValue();
      auto fTy = dyn_cast<FloatType>(re.getType());
      auto v = fTy.isF64() ? real.convertToDouble() : real.convertToFloat();
      rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(re, APFloat{v}, fTy);
      return success();
    }
    return failure();
  }
};

// Fold complex.im ops if the argument is constant.
class ComplexImPattern : public OpRewritePattern<complex::ImOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(complex::ImOp im,
                                PatternRewriter &rewriter) const override {
    auto val = im.getOperand();
    auto valCon = val.getDefiningOp<complex::ConstantOp>();
    if (valCon) {
      auto attr = valCon.getValue();
      auto imag = cast<FloatAttr>(attr[1]).getValue();
      auto fTy = dyn_cast<FloatType>(im.getType());
      auto v = fTy.isF64() ? imag.convertToDouble() : imag.convertToFloat();
      rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(im, APFloat{v}, fTy);
      return success();
    }
    return failure();
  }
};

class ConstPropComplexPass
    : public cudaq::opt::impl::ConstPropComplexBase<ConstPropComplexPass> {
public:
  using ConstPropComplexBase::ConstPropComplexBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();
    for (Operation &op : *module.getBody()) {
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func)
        continue;
      DominanceInfo domInfo(func);
      std::string funcName = func.getName().str();
      RewritePatternSet patterns(ctx);
      patterns.insert<ComplexCreatePattern>(ctx);
      patterns.insert<FloatExtendPattern>(ctx);
      patterns.insert<FloatTruncatePattern>(ctx);
      patterns.insert<ComplexRePattern>(ctx);
      patterns.insert<ComplexImPattern>(ctx);
      patterns.insert<CastArrayPtrPattern>(ctx);

      LLVM_DEBUG(llvm::dbgs()
                 << "Before lifting constant array: " << func << '\n');

      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns))))
        signalPassFailure();

      LLVM_DEBUG(llvm::dbgs()
                 << "After lifting constant array: " << func << '\n');
    }
  }
};
} // namespace
