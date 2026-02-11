/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/CCToLLVM.h"
#include "CodeGenOps.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
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
    Type type = getTypeConverter()->convertType(alloc.getElementType());
    Value size = adaptor.getSeqSize();
    if (!size)
      size = cudaq::opt::factory::genLlvmI32Constant(alloc.getLoc(), rewriter, 1);
    rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(alloc, getPtrType(), type, size);
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
    auto tuplePtrTy = getPtrType();
    auto structTy = dyn_cast<LLVM::LLVMStructType>(operands[0].getType());
    if (!structTy)
      return failure();
    auto one = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    auto extract = LLVM::ExtractValueOp::create(rewriter, loc,
        structTy.getBody()[1], operands[0], one);
    auto tupleVal =
        LLVM::BitcastOp::create(rewriter, loc, tuplePtrTy, extract);
    auto loadOp =
        LLVM::LoadOp::create(rewriter, loc, tupleTy, tupleVal);
    // In LLVM 22, replaceOp strictly requires the same number of results.
    // The LoadOp returns a single struct value; extract each field to match
    // the multiple results of CallableClosureOp.
    SmallVector<Value> results;
    for (std::size_t i = 0, N = callable.getResults().size(); i < N; ++i) {
      auto idx = DenseI64ArrayAttr::get(
          ctx, ArrayRef<std::int64_t>{static_cast<int64_t>(i)});
      results.push_back(LLVM::ExtractValueOp::create(
          rewriter, loc, resTy[i], loadOp.getResult(), idx));
    }
    rewriter.replaceOp(callable, results);
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
    auto extract = LLVM::ExtractValueOp::create(rewriter, loc, structTy.getBody()[0], operands[0], zero);
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
    // Get the mlir::FunctionType signature from the callable
    auto calleeFuncTy =
        cast<cudaq::cc::CallableType>(call.getCallee().getType())
            .getSignature();
    auto operands = adaptor.getOperands();
    auto *ctx = rewriter.getContext();
    auto structTy = dyn_cast<LLVM::LLVMStructType>(operands[0].getType());
    if (!structTy)
      return failure();

    // Extract raw function pointer (first element of callable struct)
    auto ptr0Ty = structTy.getBody()[0];
    auto zero = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{0});
    auto rawFuncPtr =
        LLVM::ExtractValueOp::create(rewriter, loc, ptr0Ty, operands[0], zero);

    // Extract raw tuple pointer (second element of callable struct)
    auto ptr1Ty = structTy.getBody()[1];
    auto one = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    auto rawTuplePtr =
        LLVM::ExtractValueOp::create(rewriter, loc, ptr1Ty, operands[0], one);

    // Build the LLVM function type by converting the signature's types
    // individually (since convertType on FunctionType returns ptr with opaque
    // pointers)
    SmallVector<Type> llvmArgTys;
    for (Type argTy : calleeFuncTy.getInputs())
      llvmArgTys.push_back(getTypeConverter()->convertType(argTy));

    Type llvmRetTy;
    if (calleeFuncTy.getNumResults() == 0)
      llvmRetTy = LLVM::LLVMVoidType::get(ctx);
    else if (calleeFuncTy.getNumResults() == 1)
      llvmRetTy = getTypeConverter()->convertType(calleeFuncTy.getResult(0));
    else {
      // Multiple results - pack into a struct
      SmallVector<Type> llvmResultTys;
      for (Type resTy : calleeFuncTy.getResults())
        llvmResultTys.push_back(getTypeConverter()->convertType(resTy));
      llvmRetTy = LLVM::LLVMStructType::getLiteral(ctx, llvmResultTys);
    }
    auto llvmFuncTy = LLVM::LLVMFunctionType::get(llvmRetTy, llvmArgTys);

    // Check if tuple pointer is null (determines direct vs closure call)
    auto i64Ty = rewriter.getI64Type();
    auto zeroI64 = cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, 0);
    auto rawTupleVal =
        LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, rawTuplePtr);
    auto isNullptr = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, rawTupleVal, zeroI64);

    // Create control flow blocks
    auto *initBlock = rewriter.getInsertionBlock();
    auto initPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(initBlock, initPos);
    auto *thenBlock = rewriter.createBlock(endBlock);
    auto *elseBlock = rewriter.createBlock(endBlock);

    SmallVector<Type> resultTy;
    if (!isa<LLVM::LLVMVoidType>(llvmFuncTy.getReturnType())) {
      resultTy.push_back(llvmFuncTy.getReturnType());
      endBlock->addArgument(resultTy[0], loc);
    }

    rewriter.setInsertionPointToEnd(initBlock);
    LLVM::CondBrOp::create(rewriter, loc, isNullptr, thenBlock, elseBlock);

    // Then block: tuple is null, call function directly with remaining operands
    rewriter.setInsertionPointToEnd(thenBlock);
    // For indirect calls, callee_operands contains: function_ptr followed by args
    SmallVector<Value> calleeOps1 = {rawFuncPtr};
    calleeOps1.append(operands.begin() + 1, operands.end());
    auto call1 = LLVM::CallOp::create(
        rewriter, loc, resultTy,
        /*var_callee_type=*/nullptr,
        /*callee=*/FlatSymbolRefAttr{},
        /*callee_operands=*/calleeOps1,
        /*fastmathFlags=*/LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::none),
        /*CConv=*/LLVM::CConvAttr::get(ctx, LLVM::cconv::CConv::C),
        /*TailCallKind=*/LLVM::TailCallKindAttr::get(ctx, LLVM::tailcallkind::TailCallKind::None),
        /*memory_effects=*/nullptr,
        /*convergent=*/nullptr,
        /*no_unwind=*/nullptr,
        /*will_return=*/nullptr,
        /*op_bundle_operands=*/{},
        /*op_bundle_tags=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*no_inline=*/nullptr,
        /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr,
        /*noalias_scopes=*/nullptr,
        /*tbaa=*/nullptr);
    LLVM::BrOp::create(rewriter, loc, call1.getResults(), endBlock);

    // Else block: tuple is not null, call with callable struct as first arg
    rewriter.setInsertionPointToEnd(elseBlock);
    // With opaque pointers, no bitcast needed - rawFuncPtr is already ptr
    SmallVector<Value> calleeOps2 = {rawFuncPtr};
    calleeOps2.append(operands.begin(), operands.end());
    auto call2 = LLVM::CallOp::create(
        rewriter, loc, resultTy,
        /*var_callee_type=*/nullptr,
        /*callee=*/FlatSymbolRefAttr{},
        /*callee_operands=*/calleeOps2,
        /*fastmathFlags=*/LLVM::FastmathFlagsAttr::get(ctx, LLVM::FastmathFlags::none),
        /*CConv=*/LLVM::CConvAttr::get(ctx, LLVM::cconv::CConv::C),
        /*TailCallKind=*/LLVM::TailCallKindAttr::get(ctx, LLVM::tailcallkind::TailCallKind::None),
        /*memory_effects=*/nullptr,
        /*convergent=*/nullptr,
        /*no_unwind=*/nullptr,
        /*will_return=*/nullptr,
        /*op_bundle_operands=*/{},
        /*op_bundle_tags=*/nullptr,
        /*arg_attrs=*/nullptr,
        /*res_attrs=*/nullptr,
        /*no_inline=*/nullptr,
        /*always_inline=*/nullptr,
        /*inline_hint=*/nullptr,
        /*access_groups=*/nullptr,
        /*alias_scopes=*/nullptr,
        /*noalias_scopes=*/nullptr,
        /*tbaa=*/nullptr);
    LLVM::BrOp::create(rewriter, loc, call2.getResults(), endBlock);

    rewriter.replaceOp(call, endBlock->getArguments());
    return success();
  }
};

class CallIndirectCallableOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::CallIndirectCallableOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CallIndirectCallableOp call, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = call.getLoc();
    auto parentModule = call->getParentOfType<ModuleOp>();
    auto funcPtrTy = getTypeConverter()->convertType(
        cast<cudaq::cc::IndirectCallableType>(call.getCallee().getType())
            .getSignature());
    auto ptrTy = getPtrType();
    LLVM::LLVMFunctionType funcTy;
    // FIXME
    assert(false);
    auto i64Ty = rewriter.getI64Type(); // intptr_t
    FlatSymbolRefAttr funSymbol = cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::runtime::getLinkableKernelDeviceSide, ptrTy, {i64Ty},
        parentModule);

    // Use the runtime helper function to convert the key to a pointer to the
    // function that was intended to be called. This can only be functional if
    // the runtime support has been linked into the executable and the
    // device-side functions are located in the same address space as well. None
    // of these functions should be expected to reside on remote hardware.
    // Therefore, this will likely only be useful in a simulation target.
    auto lookee = LLVM::CallOp::create(rewriter, loc, ptrTy, funSymbol, ValueRange{adaptor.getCallee()});
    auto lookup =
        LLVM::BitcastOp::create(rewriter, loc, funcPtrTy, lookee.getResult());

    // Call the function that was just found in the map.
    SmallVector<Value> args = {lookup.getResult()};
    args.append(adaptor.getArgs().begin(), adaptor.getArgs().end());
    if (isa<LLVM::LLVMVoidType>(funcTy.getReturnType()))
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(call, TypeRange{}, args);
    else
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(call, funcTy.getReturnType(),
                                                args);
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
              .Case([&](LLVM::LLVMFunctionType) {
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

  LogicalResult
  matchAndRewrite(cudaq::cc::ComputePtrOp cpOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the CC element type before conversion
    auto ccPtrTy = cast<cudaq::cc::PointerType>(cpOp.getBase().getType());
    Type ccEleTy = ccPtrTy.getElementType();

    // The first operand is the base pointer.
    if (cpOp.llvmNormalForm()) {
      // In this case, the `cc.compute_ptr` has already been converted such that
      // it corresponds 1:1 with the C-like semantics of LLVM's getelementptr
      // operation. Specifically, a pointer to a scalar type is overloaded to
      // possibly be the same as a pointer to an array with unknown bound.
      // All operands except the first are indices.
      // Extract inner element type from CC array type before conversion
      ccEleTy = cast<cudaq::cc::ArrayType>(ccEleTy).getElementType();
      auto newOpnds = interleaveConstantsAndOperands(
          adaptor.getDynamicIndices(), cpOp.getRawConstantIndices());
      // Convert to LLVM type after extracting the element type
      Type eleTy = getTypeConverter()->convertType(ccEleTy);
      // Rewrite the ComputePtrOp as a LLVM::GEPOp.
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(cpOp, getPtrType(), eleTy, adaptor.getBase(), newOpnds);
    } else {
      // If the `cc.compute_ptr` operation has a base argument that is not in
      // LLVM normal form, we implicitly assume that pointer's element type
      // should have been an `cc.array<T x ?>` instead of `T`. We therefore add
      // an explicit `0` as the first index. This converts the strong semantics
      // of `cc.compute_ptr` (which does not allow indexing out of bounds in
      // objects) to the weaker semantics of C/LLVM, which do implicitly allow
      // freely indexing out of bounds.
      SmallVector<std::int32_t> constIndices = {0};
      constIndices.append(cpOp.getRawConstantIndices().begin(),
                          cpOp.getRawConstantIndices().end());
      auto newOpnds =
          interleaveConstantsAndOperands(adaptor.getDynamicIndices(), constIndices);
      // Convert to LLVM type
      Type eleTy = getTypeConverter()->convertType(ccEleTy);
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(cpOp, getPtrType(), eleTy, adaptor.getBase(), newOpnds);
    }
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
    if (extract.indicesAreConstant()) {
      auto toTy = getTypeConverter()->convertType(extract.getType());
      SmallVector<std::int64_t> position{
          adaptor.getRawConstantIndices().begin(),
          adaptor.getRawConstantIndices().end()};
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
          extract, toTy, adaptor.getAggregate(), position);
    } else {
      extract.emitOpError(
          "nyi: conversion of extract_value with dynamic indices");
    }
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
    mlir::LLVM::GlobalOp::create(rewriter, loc, type, isReadOnly,
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
      Value zero = cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, 0);
      tmp =
          LLVM::IntToPtrOp::create(rewriter, loc, tupleArgTy.getBody()[1], zero);
    } else {
      Value tupleVal = LLVM::UndefOp::create(rewriter, loc, tupleTy);
      std::int64_t offsetVal = 0;
      for (auto op : operands) {
        auto offset =
            DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{offsetVal});
        tupleVal = LLVM::InsertValueOp::create(rewriter, loc, tupleTy, tupleVal,
                                                        op, offset);
        offsetVal++;
      }
      auto tuplePtrTy = getPtrType();
      tmp = cudaq::opt::factory::createLLVMTemporary(loc, rewriter, tuplePtrTy);
      LLVM::StoreOp::create(rewriter, loc, tupleVal, tmp);
    }
    Value tupleArg = LLVM::UndefOp::create(rewriter, loc, tupleArgTy);
    auto sigTy = getPtrType();
    auto tramp = LLVM::AddressOfOp::create(rewriter, loc, sigTy, cast<FlatSymbolRefAttr>(callable.getCallee()));
    auto trampoline =
        LLVM::BitcastOp::create(rewriter, loc, tupleArgTy.getBody()[0], tramp);
    auto zeroA = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{0});
    tupleArg = LLVM::InsertValueOp::create(rewriter, loc, tupleArgTy, tupleArg,
                                                    trampoline, zeroA);
    auto castTmp =
        LLVM::BitcastOp::create(rewriter, loc, tupleArgTy.getBody()[1], tmp);
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
    // We rely on MLIR here, they are using the GEP approach for now. LLVM is
    // planning to remove support for this at some point.
    // See: https://github.com/llvm/llvm-project/issues/71507 and
    //      https://github.com/llvm/llvm-project/issues/96047
    auto sizeOp = getSizeInBytes(loc, inputTy, rewriter);
    rewriter.replaceOp(sizeOfOp, sizeOp);
    return success();
  }
};

class OffsetOfOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::OffsetOfOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  // Use the GEP approach for now. LLVM is planning to remove support for this
  // at some point. See: https://github.com/llvm/llvm-project/issues/71507
  LogicalResult
  matchAndRewrite(cudaq::cc::OffsetOfOp offsetOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputTy = offsetOp.getInputType();
    SmallVector<cudaq::cc::ComputePtrArg> args;
    for (std::int32_t i : offsetOp.getConstantIndices())
      args.push_back(i);
    auto resultTy = offsetOp.getType();
    auto loc = offsetOp.getLoc();
    // TODO: replace this with some target-specific memory layout computation
    // when we upgrade to a newer MLIR.
    auto zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
    auto ptrTy = cudaq::cc::PointerType::get(inputTy);
    auto nul = cudaq::cc::CastOp::create(rewriter, loc, ptrTy, zero);
    Value nextPtr = cudaq::cc::ComputePtrOp::create(rewriter, loc, ptrTy, nul, args);
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(offsetOp, resultTy, nextPtr);
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
    auto extract = LLVM::ExtractValueOp::create(rewriter, data.getLoc(), structTy.getBody()[0], operands[0], zero);
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
    Value val = LLVM::UndefOp::create(rewriter, loc, resTy);
    auto structTy = dyn_cast<LLVM::LLVMStructType>(resTy);
    if (!structTy)
      return init.emitError("stdvec_init must have a struct as argument.");
    auto cast = LLVM::BitcastOp::create(rewriter, loc, structTy.getBody()[0],
                                                 operands[0]);
    val = LLVM::InsertValueOp::create(rewriter, loc, val, cast, zero);
    auto one = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    if (operands.size() == 2) {
      rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(init, val, operands[1],
                                                       one);
    } else {
      std::int64_t arrSize =
          llvm::cast<cudaq::cc::ArrayType>(
              llvm::cast<cudaq::cc::PointerType>(init.getBuffer().getType())
                  .getElementType())
              .getSize();
      auto i64Ty = rewriter.getI64Type();
      Value len = LLVM::ConstantOp::create(rewriter,
          loc, i64Ty, IntegerAttr::get(i64Ty, arrSize));
      rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(init, val, len, one);
    }
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
        getPtrType(),
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

class VarargCallPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::VarargCallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::VarargCallOp vcall, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> types;
    for (auto ty : vcall.getResultTypes())
      types.push_back(getTypeConverter()->convertType(ty));

    // For vararg calls, we need to set the var_callee_type attribute.
    // Look up the callee function to get its type.
    auto module = vcall->getParentOfType<ModuleOp>();
    auto calleeName = vcall.getCallee();
    TypeAttr varCalleeType;
    if (auto func = module.lookupSymbol<LLVM::LLVMFuncOp>(calleeName)) {
      varCalleeType = TypeAttr::get(func.getFunctionType());
    }

    auto callOp = rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        vcall, types, calleeName, adaptor.getArgs());
    if (varCalleeType)
      callOp.setVarCalleeTypeAttr(varCalleeType);
    return success();
  }
};

class NoInlineCallPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::NoInlineCallOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::NoInlineCallOp nicall, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> types;
    for (auto ty : nicall.getResultTypes())
      types.push_back(getTypeConverter()->convertType(ty));
    rewriter.replaceOpWithNewOp<func::CallOp>(nicall, types, nicall.getCallee(),
                                              adaptor.getArgs());
    return success();
  }
};
} // namespace

void cudaq::opt::populateCCToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  patterns
      .insert<AddressOfOpPattern, AllocaOpPattern, CallableClosureOpPattern,
              CallableFuncOpPattern, CallCallableOpPattern,
              CallIndirectCallableOpPattern, CastOpPattern, ComputePtrOpPattern,
              CreateStringLiteralOpPattern, ExtractValueOpPattern,
              FuncToPtrOpPattern, GlobalOpPattern, InsertValueOpPattern,
              InstantiateCallableOpPattern, LoadOpPattern, NoInlineCallPattern,
              OffsetOfOpPattern, PoisonOpPattern, SizeOfOpPattern,
              StdvecDataOpPattern, StdvecInitOpPattern, StdvecSizeOpPattern,
              StoreOpPattern, UndefOpPattern, VarargCallPattern>(typeConverter);
}
