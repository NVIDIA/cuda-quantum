/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/CCToLLVM.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#define DEBUG_TYPE "cc-to-llvm"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns for CC dialect ops.
//===----------------------------------------------------------------------===//

class AddressOfOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::AddressOfOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // One-to-one conversion to llvm.addressof op.
  LogicalResult
  matchAndRewrite(cudaq::cc::AddressOfOp addr, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(addr.getType());
    auto name = addr.getGlobalName();
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(addr, type, name);
    return success();
  }
};

class AllocaOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::AllocaOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Convert each cc::AllocaOp to an LLVM::AllocaOp.
  LogicalResult
  matchAndRewrite(cudaq::cc::AllocaOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto toTy = LLVM::LLVMPointerType::get([&]() -> Type {
      if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(alloc.getElementType());
          arrTy && arrTy.isUnknownSize())
        return getTypeConverter()->convertType(arrTy.getElementType());
      return getTypeConverter()->convertType(alloc.getElementType());
    }());
    if (operands.empty()) {
      rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(
          alloc, toTy,
          ArrayRef<Value>{cudaq::opt::factory::genLlvmI32Constant(
              alloc.getLoc(), rewriter, 1)});
    } else {
      rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(alloc, toTy, operands);
    }
    return success();
  }
};

class CallableClosureOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::CallableClosureOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CallableClosureOp callable, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = callable.getLoc();
    auto operands = adaptor.getOperands();
    SmallVector<Type> resTy;
    for (std::size_t i = 0, N = callable.getResults().size(); i < N; ++i)
      resTy.push_back(getTypeConverter()->convertType(callable.getType(i)));
    auto *ctx = rewriter.getContext();
    auto tupleTy = LLVM::LLVMStructType::getLiteral(ctx, resTy);
    auto tuplePtrTy = cudaq::opt::factory::getPointerType(tupleTy);
    auto structTy = dyn_cast<LLVM::LLVMStructType>(operands[0].getType());
    if (!structTy)
      return failure();
    auto one = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    auto extract = rewriter.create<LLVM::ExtractValueOp>(
        loc, structTy.getBody()[1], operands[0], one);
    auto tupleVal = rewriter.create<LLVM::BitcastOp>(loc, tuplePtrTy, extract);
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(callable, tupleTy, tupleVal);
    return success();
  }
};

class CallableFuncOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::CallableFuncOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CallableFuncOp callable, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = callable.getLoc();
    auto operands = adaptor.getOperands();
    auto resTy = getTypeConverter()->convertType(callable.getType());
    auto structTy = dyn_cast<LLVM::LLVMStructType>(operands[0].getType());
    if (!structTy)
      return failure();
    auto *ctx = rewriter.getContext();
    auto zero = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{0});
    auto extract = rewriter.create<LLVM::ExtractValueOp>(
        loc, structTy.getBody()[0], operands[0], zero);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(callable, resTy, extract);
    return success();
  }
};

class CallCallableOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::CallCallableOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CallCallableOp call, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = call.getLoc();
    auto calleeFuncTy =
        cast<cudaq::cc::CallableType>(call.getCallee().getType())
            .getSignature();
    auto operands = adaptor.getOperands();
    auto *ctx = rewriter.getContext();
    auto structTy = dyn_cast<LLVM::LLVMStructType>(operands[0].getType());
    if (!structTy)
      return failure();
    auto ptr0Ty = structTy.getBody()[0];
    auto zero = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{0});
    auto rawFuncPtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, ptr0Ty, operands[0], zero);
    auto ptr1Ty = structTy.getBody()[1];
    auto one = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    auto rawTuplePtr =
        rewriter.create<LLVM::ExtractValueOp>(loc, ptr1Ty, operands[0], one);
    Type funcPtrTy = getTypeConverter()->convertType(calleeFuncTy);
    auto funcPtr = rewriter.create<LLVM::BitcastOp>(loc, funcPtrTy, rawFuncPtr);
    auto i64Ty = rewriter.getI64Type();
    auto zeroI64 = cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, 0);
    auto rawTupleVal =
        rewriter.create<LLVM::PtrToIntOp>(loc, i64Ty, rawTuplePtr);
    auto isNullptr = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::eq,
                                                   rawTupleVal, zeroI64);
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPos);
    auto *thenBlock = rewriter.createBlock(endBlock);
    auto *elseBlock = rewriter.createBlock(endBlock);
    SmallVector<Type> resultTy;
    auto llvmFuncTy = cast<LLVM::LLVMFunctionType>(
        cast<LLVM::LLVMPointerType>(funcPtrTy).getElementType());
    if (!isa<LLVM::LLVMVoidType>(llvmFuncTy.getReturnType())) {
      resultTy.push_back(llvmFuncTy.getReturnType());
      endBlock->addArgument(resultTy[0], loc);
    }
    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<LLVM::CondBrOp>(loc, isNullptr, thenBlock, elseBlock);
    rewriter.setInsertionPointToEnd(thenBlock);
    SmallVector<Value> arguments1 = {funcPtr};
    arguments1.append(operands.begin() + 1, operands.end());
    auto call1 = rewriter.create<LLVM::CallOp>(loc, resultTy, arguments1);
    rewriter.create<LLVM::BrOp>(loc, call1.getResults(), endBlock);
    rewriter.setInsertionPointToEnd(elseBlock);
    SmallVector<Type> argTys(operands.getTypes().begin(),
                             operands.getTypes().end());
    auto adjustedFuncTy =
        LLVM::LLVMFunctionType::get(llvmFuncTy.getReturnType(), argTys);
    auto adjustedFuncPtr = rewriter.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(adjustedFuncTy), funcPtr);
    SmallVector<Value> arguments2 = {adjustedFuncPtr};
    arguments2.append(operands.begin(), operands.end());
    auto call2 = rewriter.create<LLVM::CallOp>(loc, resultTy, arguments2);
    rewriter.create<LLVM::BrOp>(loc, call2.getResults(), endBlock);
    rewriter.replaceOp(call, endBlock->getArguments());
    return success();
  }
};

class CastOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::CastOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Convert each cc::CastOp to one of the flavors of LLVM casts.
  LogicalResult
  matchAndRewrite(cudaq::cc::CastOp cast, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto fromTy = operands[0].getType();
    auto toTy = getTypeConverter()->convertType(cast.getType());
    auto boilerplate = [&]<typename A>(A *) {
      rewriter.replaceOpWithNewOp<A>(cast, toTy, operands);
    };
    TypeSwitch<Type>(toTy)
        .Case([&](IntegerType toIntTy) {
          TypeSwitch<Type>(fromTy)
              .Case([&](IntegerType fromIntTy) {
                if (fromIntTy.getWidth() < toIntTy.getWidth()) {
                  if (cast.getSint())
                    boilerplate((LLVM::SExtOp *)nullptr);
                  else
                    boilerplate((LLVM::ZExtOp *)nullptr);
                } else {
                  boilerplate((LLVM::TruncOp *)nullptr);
                }
              })
              .Case([&](FloatType) {
                if (cast.getSint())
                  boilerplate((LLVM::FPToSIOp *)nullptr);
                else if (cast.getZint())
                  boilerplate((LLVM::FPToUIOp *)nullptr);
                else
                  boilerplate((LLVM::BitcastOp *)nullptr);
              })
              .Case([&](LLVM::LLVMPointerType) {
                boilerplate((LLVM::PtrToIntOp *)nullptr);
              });
        })
        .Case([&](FloatType toFloatTy) {
          TypeSwitch<Type>(fromTy)
              .Case([&](FloatType fromFloatTy) {
                if (fromFloatTy.getWidth() < toFloatTy.getWidth())
                  boilerplate((LLVM::FPExtOp *)nullptr);
                else
                  boilerplate((LLVM::FPTruncOp *)nullptr);
              })
              .Case([&](IntegerType) {
                if (cast.getSint())
                  boilerplate((LLVM::SIToFPOp *)nullptr);
                else if (cast.getZint())
                  boilerplate((LLVM::UIToFPOp *)nullptr);
                else
                  boilerplate((LLVM::BitcastOp *)nullptr);
              });
        })
        .Case([&](LLVM::LLVMPointerType toPtrTy) {
          TypeSwitch<Type>(fromTy)
              .Case([&](LLVM::LLVMPointerType) {
                boilerplate((LLVM::BitcastOp *)nullptr);
              })
              .Case([&](IntegerType) {
                boilerplate((LLVM::IntToPtrOp *)nullptr);
              });
        });
    return success();
  }
};

class ComputePtrOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::ComputePtrOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Convert each cc::ComputePtrOp to an LLVM::GEPOp.
  LogicalResult
  matchAndRewrite(cudaq::cc::ComputePtrOp cp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    bool dropFirst = false;
    auto toTy = getTypeConverter()->convertType(cp.getType());
    Value base = operands[0];
    if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType()))
      if (auto arrTy = dyn_cast<LLVM::LLVMArrayType>(ptrTy.getElementType())) {
        // Eliminate intermediate array type. Not needed in LLVM. (NB: for some
        // element types, the executable will crash.)
        auto ty = cudaq::opt::factory::getPointerType(arrTy.getElementType());
        base = rewriter.create<LLVM::BitcastOp>(cp.getLoc(), ty, base);
      }
    auto gepOpnds = interleaveConstantsAndOperands(
        operands.drop_front(),
        cp.getRawConstantIndices().drop_front(dropFirst ? 1 : 0));
    rewriter.replaceOpWithNewOp<LLVM::GEPOp>(cp, toTy, base, gepOpnds);
    return success();
  }

  static SmallVector<LLVM::GEPArg>
  interleaveConstantsAndOperands(ValueRange values,
                                 ArrayRef<std::int32_t> rawConsts) {
    SmallVector<LLVM::GEPArg> result;
    auto valIter = values.begin();
    for (auto rc : rawConsts) {
      if (rc == cudaq::cc::ComputePtrOp::kDynamicIndex)
        result.push_back(*valIter++);
      else
        result.push_back(rc);
    }
    return result;
  }
};

class ExtractValueOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::ExtractValueOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::ExtractValueOp extract, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto toTy = getTypeConverter()->convertType(extract.getType());
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        extract, toTy, adaptor.getContainer(), adaptor.getPosition());
    return success();
  }
};

class FuncToPtrOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::FuncToPtrOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // This becomes a bitcast op.
  LogicalResult
  matchAndRewrite(cudaq::cc::FuncToPtrOp ftp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto toTy = getTypeConverter()->convertType(ftp.getType());
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(ftp, toTy, operands);
    return success();
  }
};

class GlobalOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::GlobalOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Replace the cc.global with an llvm.global, updating the types, etc.
  LogicalResult
  matchAndRewrite(cudaq::cc::GlobalOp global, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = global.getLoc();
    auto ptrTy = cast<cudaq::cc::PointerType>(global.getType());
    auto eleTy = ptrTy.getElementType();
    Type type = getTypeConverter()->convertType(eleTy);
    auto name = global.getSymName();
    bool isReadOnly = global.getConstant();
    Attribute initializer = global.getValue().value_or(Attribute{});
    rewriter.create<mlir::LLVM::GlobalOp>(loc, type, isReadOnly,
                                          LLVM::Linkage::Private, name,
                                          initializer, /*alignment=*/0);
    rewriter.eraseOp(global);
    return success();
  }
};

class InsertValueOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::InsertValueOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::InsertValueOp insert, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto toTy = getTypeConverter()->convertType(insert.getType());
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        insert, toTy, adaptor.getContainer(), adaptor.getValue(),
        adaptor.getPosition());
    return success();
  }
};

class InstantiateCallableOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::InstantiateCallableOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::InstantiateCallableOp callable, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = callable.getLoc();
    auto operands = adaptor.getOperands();
    auto *ctx = rewriter.getContext();
    SmallVector<Type> tupleMemTys(adaptor.getOperands().getTypes().begin(),
                                  adaptor.getOperands().getTypes().end());
    auto tupleTy = LLVM::LLVMStructType::getLiteral(ctx, tupleMemTys);
    Value tmp;
    auto tupleArgTy = cudaq::opt::lambdaAsPairOfPointers(ctx);
    if (callable.getNoCapture()) {
      auto zero = cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, 0);
      tmp =
          rewriter.create<LLVM::IntToPtrOp>(loc, tupleArgTy.getBody()[1], zero);
    } else {
      Value tupleVal = rewriter.create<LLVM::UndefOp>(loc, tupleTy);
      std::int64_t offsetVal = 0;
      for (auto op : operands) {
        auto offset =
            DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{offsetVal});
        tupleVal = rewriter.create<LLVM::InsertValueOp>(loc, tupleTy, tupleVal,
                                                        op, offset);
        offsetVal++;
      }
      auto tuplePtrTy = cudaq::opt::factory::getPointerType(tupleTy);
      tmp = cudaq::opt::factory::createLLVMTemporary(loc, rewriter, tuplePtrTy);
      rewriter.create<LLVM::StoreOp>(loc, tupleVal, tmp);
    }
    Value tupleArg = rewriter.create<LLVM::UndefOp>(loc, tupleArgTy);
    auto module = callable->getParentOfType<ModuleOp>();
    auto *calledFuncOp = module.lookupSymbol(callable.getCallee());
    auto sigTy = [&]() -> Type {
      if (auto calledFunc = dyn_cast<func::FuncOp>(calledFuncOp))
        return getTypeConverter()->convertType(calledFunc.getFunctionType());
      return cudaq::opt::factory::getPointerType(
          cast<LLVM::LLVMFuncOp>(calledFuncOp).getFunctionType());
    }();
    auto tramp = rewriter.create<LLVM::AddressOfOp>(
        loc, sigTy, callable.getCallee().cast<FlatSymbolRefAttr>());
    auto trampoline =
        rewriter.create<LLVM::BitcastOp>(loc, tupleArgTy.getBody()[0], tramp);
    auto zeroA = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{0});
    tupleArg = rewriter.create<LLVM::InsertValueOp>(loc, tupleArgTy, tupleArg,
                                                    trampoline, zeroA);
    auto castTmp =
        rewriter.create<LLVM::BitcastOp>(loc, tupleArgTy.getBody()[1], tmp);
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(
        callable, tupleArgTy, tupleArg, castTmp,
        DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1}));
    return success();
  }
};

class LoadOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::LoadOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Convert each cc::LoadOp to an LLVM::LoadOp.
  LogicalResult
  matchAndRewrite(cudaq::cc::LoadOp load, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    Type toTy = getTypeConverter()->convertType(load.getType());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(load, toTy, operands);
    return success();
  }
};

class SizeOfOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::SizeOfOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Use the GEP approach for now. LLVM is planning to remove support for this
  // at some point. See: https://github.com/llvm/llvm-project/issues/71507
  LogicalResult
  matchAndRewrite(cudaq::cc::SizeOfOp sizeOfOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputTy = sizeOfOp.getInputType();
    auto resultTy = sizeOfOp.getType();
    if (quake::isQuakeType(inputTy) || cudaq::cc::isDynamicType(inputTy)) {
      // Types that cannot be reified produce the poison op.
      rewriter.replaceOpWithNewOp<cudaq::cc::PoisonOp>(sizeOfOp, resultTy);
      return success();
    }
    auto loc = sizeOfOp.getLoc();
    // TODO: replace this with some target-specific memory layout computation
    // when we upgrade to a newer MLIR.
    auto zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    auto ptrTy = cudaq::cc::PointerType::get(inputTy);
    auto nullCast = rewriter.create<cudaq::cc::CastOp>(loc, ptrTy, zero);
    Value nextPtr = rewriter.create<cudaq::cc::ComputePtrOp>(
        loc, ptrTy, nullCast, ArrayRef<cudaq::cc::ComputePtrArg>{1});
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(sizeOfOp, resultTy, nextPtr);
    return success();
  }
};

class StdvecDataOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::StdvecDataOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::StdvecDataOp data, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto resTy = getTypeConverter()->convertType(data.getType());
    auto ctx = data.getContext();
    auto zero = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{0});
    auto structTy = dyn_cast<LLVM::LLVMStructType>(operands[0].getType());
    if (!structTy)
      return data.emitError("stdvec_data must have a struct as argument.");
    auto extract = rewriter.create<LLVM::ExtractValueOp>(
        data.getLoc(), structTy.getBody()[0], operands[0], zero);
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(data, resTy, extract);
    return success();
  }
};

class StdvecInitOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::StdvecInitOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::StdvecInitOp init, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto resTy = getTypeConverter()->convertType(init.getType());
    auto ctx = init.getContext();
    auto zero = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{0});
    auto loc = init.getLoc();
    Value val = rewriter.create<LLVM::UndefOp>(loc, resTy);
    auto structTy = dyn_cast<LLVM::LLVMStructType>(resTy);
    if (!structTy)
      return init.emitError("stdvec_init must have a struct as argument.");
    auto cast = rewriter.create<LLVM::BitcastOp>(loc, structTy.getBody()[0],
                                                 operands[0]);
    val = rewriter.create<LLVM::InsertValueOp>(loc, val, cast, zero);
    auto one = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(init, val, operands[1],
                                                     one);
    return success();
  }
};

class StdvecSizeOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::StdvecSizeOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::StdvecSizeOp size, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto resTy = getTypeConverter()->convertType(size.getType());
    auto ctx = size.getContext();
    auto one = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(size, resTy, operands[0],
                                                      one);
    return success();
  }
};

class CreateStringLiteralOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::CreateStringLiteralOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::CreateStringLiteralOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(cudaq::cc::CreateStringLiteralOp stringLiteralOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = stringLiteralOp.getLoc();
    auto parentModule = stringLiteralOp->getParentOfType<ModuleOp>();
    StringRef stringLiteral = stringLiteralOp.getStringLiteral();

    // Write to the module body
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(parentModule.getBody());

    // Create the register name global
    auto builder = cudaq::IRBuilder::atBlockEnd(parentModule.getBody());
    auto slGlobal =
        builder.genCStringLiteralAppendNul(loc, parentModule, stringLiteral);

    // Shift back to the function
    rewriter.restoreInsertionPoint(insertPoint);

    // Get the string address
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
        stringLiteralOp,
        cudaq::opt::factory::getPointerType(slGlobal.getType()),
        slGlobal.getSymName());

    return success();
  }
};

class StoreOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::StoreOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Convert each cc::StoreOp to an LLVM::StoreOp.
  LogicalResult
  matchAndRewrite(cudaq::cc::StoreOp store, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(store, operands[0], operands[1]);
    return success();
  }
};

class PoisonOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::PoisonOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::PoisonOp poison, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(poison.getType());
    // FIXME: This should use PoisonOp, obviously, when we upgrade MLIR.
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(poison, resTy);
    return success();
  }
};

class UndefOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::UndefOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::UndefOp undef, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(undef.getType());
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(undef, resTy);
    return success();
  }
};
} // namespace

void cudaq::opt::populateCCToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns.insert<AddressOfOpPattern, AllocaOpPattern, CallableClosureOpPattern,
                  CallableFuncOpPattern, CallCallableOpPattern, CastOpPattern,
                  ComputePtrOpPattern, CreateStringLiteralOpPattern,
                  ExtractValueOpPattern, FuncToPtrOpPattern, GlobalOpPattern,
                  InsertValueOpPattern, InstantiateCallableOpPattern,
                  LoadOpPattern, PoisonOpPattern, SizeOfOpPattern,
                  StdvecDataOpPattern, StdvecInitOpPattern, StdvecSizeOpPattern,
                  StoreOpPattern, UndefOpPattern>(typeConverter);
}
