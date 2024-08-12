/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CodeGenDialect.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/CCToLLVM.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToLibm/ComplexToLibm.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CCTOLLVM
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-to-llvm-pass"

using namespace mlir;

LLVM::LLVMStructType cudaq::opt::lambdaAsPairOfPointers(MLIRContext *context) {
  auto ptrTy = cudaq::opt::factory::getPointerType(context);
  SmallVector<Type> pairOfPointers = {ptrTy, ptrTy};
  return LLVM::LLVMStructType::getLiteral(context, pairOfPointers);
}

void cudaq::opt::populateCCTypeConversions(LLVMTypeConverter *converter) {
  converter->addConversion([](cc::CallableType type) {
    return lambdaAsPairOfPointers(type.getContext());
  });
  converter->addConversion([converter](cc::SpanLikeType type) {
    auto eleTy = converter->convertType(type.getElementType());
    return factory::stdVectorImplType(eleTy);
  });
  converter->addConversion([converter](cc::PointerType type) {
    auto eleTy = type.getElementType();
    if (isa<NoneType>(eleTy))
      return factory::getPointerType(type.getContext());
    eleTy = converter->convertType(eleTy);
    if (auto arrTy = dyn_cast<cc::ArrayType>(eleTy)) {
      // If array has a static size, it becomes an LLVMArrayType. Otherwise, we
      // end up here.
      assert(arrTy.isUnknownSize());
      return factory::getPointerType(
          converter->convertType(arrTy.getElementType()));
    }
    return factory::getPointerType(eleTy);
  });
  converter->addConversion([converter](cc::ArrayType type) -> Type {
    auto eleTy = converter->convertType(type.getElementType());
    if (type.isUnknownSize())
      return type;
    return LLVM::LLVMArrayType::get(eleTy, type.getSize());
  });
  converter->addConversion(
      [](cc::StateType type) { return factory::stateImplType(type); });
  converter->addConversion([converter](cc::StructType type) -> Type {
    SmallVector<Type> members;
    for (auto t : type.getMembers())
      members.push_back(converter->convertType(t));
    return LLVM::LLVMStructType::getLiteral(type.getContext(), members,
                                            type.getPacked());
  });
}

std::size_t cudaq::opt::getDataSize(llvm::DataLayout &dataLayout, Type ty) {
  LLVMTypeConverter converter(ty.getContext());
  cudaq::opt::populateCCTypeConversions(&converter);
  auto llvmDialectTy = converter.convertType(ty);
  llvm::LLVMContext context;
  LLVM::TypeToLLVMIRTranslator translator(context);
  auto llvmTy = translator.translateType(llvmDialectTy);
  return dataLayout.getTypeAllocSize(llvmTy);
}

std::size_t cudaq::opt::getDataOffset(llvm::DataLayout &dataLayout, Type ty,
                                      std::size_t off) {
  LLVMTypeConverter converter(ty.getContext());
  cudaq::opt::populateCCTypeConversions(&converter);
  auto llvmDialectTy = converter.convertType(ty);
  llvm::LLVMContext context;
  LLVM::TypeToLLVMIRTranslator translator(context);
  auto llvmTy = cast<llvm::StructType>(translator.translateType(llvmDialectTy));
  return dataLayout.getStructLayout(llvmTy)->getElementOffset(off);
}

namespace {
struct CCToLLVM : public cudaq::opt::impl::CCToLLVMBase<CCToLLVM> {
  using CCToLLVMBase::CCToLLVMBase;

  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    LLVMTypeConverter ccTypeConverter{context};

    cudaq::opt::populateCCTypeConversions(&ccTypeConverter);
    populateComplexToLibmConversionPatterns(patterns, 1);
    populateComplexToLLVMConversionPatterns(ccTypeConverter, patterns);
    arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(ccTypeConverter, patterns);
    populateMathToLLVMConversionPatterns(ccTypeConverter, patterns);

    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(ccTypeConverter, patterns);
    populateFuncToLLVMConversionPatterns(ccTypeConverter, patterns);
    cudaq::opt::populateCCToLLVMPatterns(ccTypeConverter, patterns);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      LLVM_DEBUG(getOperation().dump());
      signalPassFailure();
    }
  }
};
} // namespace
