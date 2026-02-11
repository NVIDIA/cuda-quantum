/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/QuakeToExecMgr.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/CudaqFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "quake-to-cc"

using namespace mlir;

cudaq::cc::PointerType
cudaq::opt::getCudaqQubitType(mlir::MLIRContext *context) {
  auto i64Ty = IntegerType::get(context, 64);
  auto arrI64Ty = cc::ArrayType::get(i64Ty);
  return cc::PointerType::get(arrI64Ty);
}

cudaq::cc::StructType
cudaq::opt::getCudaqQubitSpanType(mlir::MLIRContext *context) {
  auto qubitTy = opt::getCudaqQubitType(context);
  auto i64Ty = IntegerType::get(context, 64);
  SmallVector<Type> members{qubitTy, i64Ty};
  return cudaq::cc::StructType::get(context, ".qubit_span", members);
}

static Value packQubitSpans(Location loc, ConversionPatternRewriter &rewriter,
                            ValueRange operands) {
  auto qspanTy = cudaq::opt::getCudaqQubitSpanType(rewriter.getContext());
  Value newspan;
  if (operands.empty()) {
    newspan = cudaq::cc::AllocaOp::create(rewriter, loc, qspanTy);
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
    auto nullPtrVal = cudaq::cc::CastOp::create(rewriter, 
        loc, cudaq::opt::getCudaqQubitType(rewriter.getContext()), zero);
    func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                  cudaq::opt::CudaqEMWriteToSpan,
                                  ValueRange{newspan, nullPtrVal, zero});
  } else if (operands.size() == 1) {
    // Nothing to concatenate in this case.
    newspan = operands[0];
  } else {
    newspan = cudaq::cc::AllocaOp::create(rewriter, loc, qspanTy);
    // Loop over all arguments and count the number of qubits.
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
    Value sum = zero;
    auto i64Ty = rewriter.getI64Type();
    auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
    for (auto v : operands) {
      auto sizePtr = cudaq::cc::ComputePtrOp::create(rewriter, 
          loc, ptrI64Ty, v, ArrayRef<cudaq::cc::ComputePtrArg>{1});
      auto size = cudaq::cc::LoadOp::create(rewriter, loc, sizePtr);
      sum = arith::AddIOp::create(rewriter, loc, sum, size);
    }
    // Allocate a fresh buffer.
    auto newBuffer = cudaq::cc::AllocaOp::create(rewriter, loc, i64Ty, sum);
    func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                  cudaq::opt::CudaqEMWriteToSpan,
                                  ValueRange{newspan, newBuffer, sum});
    // Copy the i64 values to the new buffer.
    sum = zero;
    Value size = zero;
    for (auto v : operands) {
      auto dest = cudaq::cc::ComputePtrOp::create(rewriter, 
          loc, ptrI64Ty, newBuffer, ArrayRef<cudaq::cc::ComputePtrArg>{sum});
      auto sizePtr = cudaq::cc::ComputePtrOp::create(rewriter, 
          loc, ptrI64Ty, v, ArrayRef<cudaq::cc::ComputePtrArg>{1});
      size = cudaq::cc::LoadOp::create(rewriter, loc, sizePtr);
      func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                    cudaq::opt::CudaqEMConcatSpan,
                                    ValueRange{dest, v, size});
      sum = arith::AddIOp::create(rewriter, loc, sum, size);
    }
  }
  return newspan;
}

namespace {
//===----------------------------------------------------------------------===//
// Conversion patterns for Quake dialect ops.
//===----------------------------------------------------------------------===//

/// Lower quake dialect to cc dialect by calling support functions.
/// The strategy is to convert quake references and quake veqs to a uniform span
/// representation, where the former is a span of size 1 and the latter may be a
/// span of non-constant size.
/// All support functions will be declared as intrinsics. Some will be
/// implemented in C++ and others in the intrinsics table.
class AllocaOpRewrite : public OpConversionPattern<quake::AllocaOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::AllocaOp alloca, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = alloca.getLoc();
    auto i64Ty = rewriter.getI64Type();
    auto qspanTy = cudaq::opt::getCudaqQubitSpanType(rewriter.getContext());
    Value qspan = cudaq::cc::AllocaOp::create(rewriter, loc, qspanTy);
    if (auto resultType = dyn_cast<quake::RefType>(alloca.getType())) {
      auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
      Value buffer = cudaq::cc::AllocaOp::create(rewriter, loc, i64Ty, one);
      auto call = func::CallOp::create(rewriter, 
          loc, i64Ty, cudaq::opt::CudaqEMAllocate, ValueRange{});
      auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
      auto toAddr = cudaq::cc::ComputePtrOp::create(rewriter, 
          loc, ptrI64Ty, buffer, ArrayRef<cudaq::cc::ComputePtrArg>{0});
      cudaq::cc::StoreOp::create(rewriter, loc, call.getResult(0), toAddr);
      func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                    cudaq::opt::CudaqEMWriteToSpan,
                                    ValueRange{qspan, buffer, one});
    } else {
      Value sizeOperand;
      if (adaptor.getOperands().empty()) {
        auto type = cast<quake::VeqType>(alloca.getType());
        assert(type.hasSpecifiedSize() && "veq must have a constant size");
        auto constantSize = type.getSize();
        sizeOperand =
            arith::ConstantIntOp::create(rewriter, loc, constantSize, 64);
      } else if (auto intSizeTy =
                     dyn_cast<IntegerType>(adaptor.getSize().getType())) {
        sizeOperand = adaptor.getSize();
        if (intSizeTy.getWidth() != 64)
          sizeOperand = cudaq::cc::CastOp::create(rewriter, 
              loc, i64Ty, sizeOperand, cudaq::cc::CastOpMode::Unsigned);
      }
      if (!sizeOperand)
        return failure();

      Value buffer =
          cudaq::cc::AllocaOp::create(rewriter, loc, i64Ty, sizeOperand);
      func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                    cudaq::opt::CudaqEMWriteToSpan,
                                    ValueRange{qspan, buffer, sizeOperand});
      func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                    cudaq::opt::CudaqEMAllocateVeq,
                                    ValueRange{qspan, sizeOperand});
    }
    rewriter.replaceOp(alloca, qspan);
    return success();
  }
};

class DeallocOpRewrite : public OpConversionPattern<quake::DeallocOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::DeallocOp dealloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        dealloc, mlir::TypeRange{}, cudaq::opt::CudaqEMReturn,
        ValueRange{adaptor.getReference()});
    return success();
  }
};

class ConcatOpRewrite : public OpConversionPattern<quake::ConcatOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::ConcatOp concat, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = concat.getLoc();
    auto newspan = packQubitSpans(loc, rewriter, adaptor.getOperands());
    // Result is the new buffer.
    rewriter.replaceOp(concat, newspan);
    return success();
  }
};

class DiscriminateOpRewrite
    : public OpConversionPattern<quake::DiscriminateOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::DiscriminateOp discr, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto m = discr.getMeasurement();
    rewriter.replaceOp(discr, m);
    return success();
  }
};

class ExtractRefOpRewrite : public OpConversionPattern<quake::ExtractRefOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::ExtractRefOp extract, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = extract.getLoc();
    auto offset = [&]() -> Value {
      if (extract.hasConstantIndex())
        return arith::ConstantIntOp::create(rewriter, 
            loc, extract.getConstantIndex(), 64);
      return adaptor.getIndex();
    }();

    // Create our types.
    auto i64Ty = rewriter.getI64Type();
    auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
    auto arrI64Ty = cudaq::cc::ArrayType::get(i64Ty);
    auto ptrArrTy = cudaq::cc::PointerType::get(arrI64Ty);
    auto ptrptrTy = cudaq::cc::PointerType::get(ptrArrTy);

    auto qspan = adaptor.getVeq();
    auto qspanDataPtr = cudaq::cc::ComputePtrOp::create(rewriter, 
        loc, ptrptrTy, qspan, ArrayRef<cudaq::cc::ComputePtrArg>{0});
    auto qspanData = cudaq::cc::LoadOp::create(rewriter, loc, qspanDataPtr);
    auto buffer = cudaq::cc::ComputePtrOp::create(rewriter, 
        loc, ptrI64Ty, qspanData, ArrayRef<cudaq::cc::ComputePtrArg>{offset});
    auto qspanTy = cudaq::opt::getCudaqQubitSpanType(rewriter.getContext());
    Value newspan = cudaq::cc::AllocaOp::create(rewriter, loc, qspanTy);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    auto buf1 = cudaq::cc::CastOp::create(rewriter, loc, ptrArrTy, buffer);
    func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                  cudaq::opt::CudaqEMWriteToSpan,
                                  ValueRange{newspan, buf1, one});
    rewriter.replaceOp(extract, newspan);
    return success();
  }
};

class SubveqOpRewrite : public OpConversionPattern<quake::SubVeqOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // True to the semantics, this operation does not allocate a new buffer and
  // does not make a copy. It simply constructs a new qubit span which is a
  // subspan of the original.
  LogicalResult
  matchAndRewrite(quake::SubVeqOp subveq, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subveq.getLoc();
    auto up = [&]() -> Value {
      if (!adaptor.getUpper())
        return arith::ConstantIntOp::create(rewriter, loc, adaptor.getRawUpper(),
                                                     64);
      return adaptor.getUpper();
    }();
    auto lo = [&]() -> Value {
      if (!adaptor.getLower())
        return arith::ConstantIntOp::create(rewriter, loc, adaptor.getRawLower(),
                                                     64);
      return adaptor.getLower();
    }();
    auto diff = arith::SubIOp::create(rewriter, loc, up, lo);
    auto one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    auto length = arith::AddIOp::create(rewriter, loc, diff, one);
    // Compute the pointer to the first element in the subveq and build a new
    // array type.
    auto i64Ty = rewriter.getI64Type();
    auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
    auto ptrTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(i64Ty));
    auto ptrptrTy = cudaq::cc::PointerType::get(ptrTy);
    auto qspanDataPtr = cudaq::cc::ComputePtrOp::create(rewriter, 
        loc, ptrptrTy, adaptor.getVeq(), ArrayRef<cudaq::cc::ComputePtrArg>{0});
    auto qspanData = cudaq::cc::LoadOp::create(rewriter, loc, qspanDataPtr);
    auto buffer = cudaq::cc::ComputePtrOp::create(rewriter, 
        loc, ptrI64Ty, qspanData, ArrayRef<cudaq::cc::ComputePtrArg>{lo});
    auto qspanTy = cudaq::opt::getCudaqQubitSpanType(rewriter.getContext());
    Value newspan = cudaq::cc::AllocaOp::create(rewriter, loc, qspanTy);
    func::CallOp::create(rewriter, loc, mlir::TypeRange{},
                                  cudaq::opt::CudaqEMWriteToSpan,
                                  ValueRange{newspan, buffer, length});
    rewriter.replaceOp(subveq, newspan);
    return success();
  }
};

class ResetRewrite : public OpConversionPattern<quake::ResetOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::ResetOp resetOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        resetOp, mlir::TypeRange{}, cudaq::opt::CudaqEMReset, adaptor.getOperands());
    return success();
  }
};

template <typename OP>
class GenericRewrite : public OpConversionPattern<OP> {
public:
  using Base = OpConversionPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP qop, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = qop.getLoc();
    auto instName = qop->getName().stripDialect().str();
    auto mod = qop->template getParentOfType<ModuleOp>();
    auto opName = createOpName(loc, mod, rewriter, instName);
    auto i8Ty = rewriter.getI8Type();
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto regTy = cudaq::cc::PointerType::get(opName.getType());
    auto addr = cudaq::cc::AddressOfOp::create(rewriter, loc, regTy,
                                                        opName.getSymName());
    auto opString = cudaq::cc::CastOp::create(rewriter, loc, ptrI8Ty, addr);
    auto paramSize = adaptor.getParameters().size();
    Value numParams = arith::ConstantIntOp::create(rewriter, loc, paramSize, 64);
    auto f64Ty = rewriter.getF64Type();
    auto arrF64Ty = cudaq::cc::ArrayType::get(f64Ty);
    auto ptrParamTy = cudaq::cc::PointerType::get(arrF64Ty);
    auto ptrF64Ty = cudaq::cc::PointerType::get(f64Ty);
    auto params = [&]() -> Value {
      if (paramSize == 0) {
        auto zero = arith::ConstantIntOp::create(rewriter, loc, paramSize, 64);
        return cudaq::cc::CastOp::create(rewriter, loc, ptrParamTy, zero);
      }
      auto buffer = cudaq::cc::AllocaOp::create(rewriter, loc, f64Ty, numParams);
      for (auto iter : llvm::enumerate(adaptor.getParameters())) {
        std::int32_t i = iter.index();
        auto p = iter.value();
        auto ptr = cudaq::cc::ComputePtrOp::create(rewriter, 
            loc, ptrF64Ty, buffer, ArrayRef<cudaq::cc::ComputePtrArg>{i});
        cudaq::cc::StoreOp::create(rewriter, loc, p, ptr);
      }
      return buffer;
    }();
    auto controls = packQubitSpans(loc, rewriter, adaptor.getControls());
    auto targets = packQubitSpans(loc, rewriter, adaptor.getTargets());
    auto isAdj = [&]() -> Value {
      if (qop.isAdj())
        return arith::ConstantIntOp::create(rewriter, loc, 1, 1);
      return arith::ConstantIntOp::create(rewriter, loc, 0, 1);
    }();
    rewriter.template replaceOpWithNewOp<func::CallOp>(
        qop, mlir::TypeRange{}, cudaq::opt::CudaqEMApply,
        ValueRange{opString, numParams, params, controls, targets, isAdj});
    return success();
  }

  // Create a global with the "name" to use for this operation.
  LLVM::GlobalOp createOpName(Location loc, ModuleOp mod,
                              ConversionPatternRewriter &rewriter,
                              StringRef name) const {
    OpBuilder::InsertionGuard guard(rewriter);
    auto builder = cudaq::IRBuilder::atBlockEnd(mod.getBody());
    return builder.genCStringLiteralAppendNul(loc, mod, name.str());
  }
};

class MzOpRewrite : public OpConversionPattern<quake::MzOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  // Create a global with the "name" to use for this measurement.
  LLVM::GlobalOp createRegisterName(Location loc, ModuleOp mod,
                                    ConversionPatternRewriter &rewriter,
                                    StringAttr nameAttr) const {
    std::string mzName;
    if (nameAttr) {
      mzName = nameAttr.getValue().str();
    } else {
      // No name was given. Use a hash based on the source file location of the
      // measurement operation, to give something that ought to be relatively
      // unique.
      std::size_t hash = mlir::hash_value(loc);
      hash = (hash ^ (hash >> 16)) & 0xFFFF;
      mzName = std::string("r") + std::to_string(hash);
    }
    OpBuilder::InsertionGuard guard(rewriter);
    auto builder = cudaq::IRBuilder::atBlockEnd(mod.getBody());
    return builder.genCStringLiteralAppendNul(loc, mod, mzName);
  }

  LogicalResult
  matchAndRewrite(quake::MzOp mzOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = mzOp.getLoc();
    auto mod = mzOp->getParentOfType<ModuleOp>();
    auto regName =
        createRegisterName(loc, mod, rewriter, mzOp.getRegisterNameAttr());
    auto i8Ty = rewriter.getI8Type();
    auto ptrI8Ty = cudaq::cc::PointerType::get(i8Ty);
    auto regTy = cudaq::cc::PointerType::get(regName.getType());
    auto addr = cudaq::cc::AddressOfOp::create(rewriter, loc, regTy,
                                                        regName.getSymName());
    auto nameAddr = cudaq::cc::CastOp::create(rewriter, loc, ptrI8Ty, addr);
    auto i32Ty = rewriter.getI32Type();
    rewriter.replaceOpWithNewOp<func::CallOp>(
        mzOp, i32Ty, cudaq::opt::CudaqEMMeasure,
        ValueRange{adaptor.getTargets()[0], nameAddr});
    return success();
  }
};

/// Convert a MX operation to a sequence H; MZ.
class MxToMzRewrite : public OpRewritePattern<quake::MxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::MxOp mx,
                                PatternRewriter &rewriter) const override {
    quake::HOp::create(rewriter,mx.getLoc(), mx.getTargets());
    rewriter.replaceOpWithNewOp<quake::MzOp>(
        mx, mx.getResultTypes(), mx.getTargets(), mx.getRegisterNameAttr());
    return success();
  }
};

/// Convert a MY operation to a sequence S; H; MZ.
class MyToMzRewrite : public OpRewritePattern<quake::MyOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::MyOp my,
                                PatternRewriter &rewriter) const override {
    quake::SOp::create(rewriter,my.getLoc(), true, ValueRange{}, ValueRange{},
                                my.getTargets());
    quake::HOp::create(rewriter,my.getLoc(), my.getTargets());
    rewriter.replaceOpWithNewOp<quake::MzOp>(
        my, my.getResultTypes(), my.getTargets(), my.getRegisterNameAttr());
    return success();
  }
};

class VeqSizeOpRewrite : public OpConversionPattern<quake::VeqSizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::VeqSizeOp vecsize, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = vecsize->getLoc();
    auto i64Ty = rewriter.getI64Type();
    auto ptrI64Ty = cudaq::cc::PointerType::get(i64Ty);
    auto sizeptr = cudaq::cc::ComputePtrOp::create(rewriter, 
        loc, ptrI64Ty, adaptor.getVeq(), ArrayRef<cudaq::cc::ComputePtrArg>{1});
    rewriter.replaceOpWithNewOp<cudaq::cc::LoadOp>(vecsize, sizeptr);
    return success();
  }
};

} // namespace

void cudaq::opt::populateQuakeToCCPatterns(TypeConverter &converter,
                                           RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  patterns.insert<AllocaOpRewrite, ConcatOpRewrite, DeallocOpRewrite,
                  DiscriminateOpRewrite, ExtractRefOpRewrite, VeqSizeOpRewrite,
                  MzOpRewrite, ResetRewrite, SubveqOpRewrite,
                  GenericRewrite<quake::HOp>, GenericRewrite<quake::PhasedRxOp>,
                  GenericRewrite<quake::R1Op>, GenericRewrite<quake::RxOp>,
                  GenericRewrite<quake::RyOp>, GenericRewrite<quake::RzOp>,
                  GenericRewrite<quake::SOp>, GenericRewrite<quake::SwapOp>,
                  GenericRewrite<quake::TOp>, GenericRewrite<quake::U2Op>,
                  GenericRewrite<quake::U3Op>, GenericRewrite<quake::XOp>,
                  GenericRewrite<quake::YOp>, GenericRewrite<quake::ZOp>>(
      converter, context);
}

void cudaq::opt::populateQuakeToCCPrepPatterns(RewritePatternSet &patterns) {
  patterns.insert<MxToMzRewrite, MyToMzRewrite>(patterns.getContext());
}
