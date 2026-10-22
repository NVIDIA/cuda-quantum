/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// CuDensityMatToLLVM: lower cudm dialect ops to llvm.call sequences
// targeting the libcudm-runtime C ABI.

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

using namespace mlir;

namespace {

// All cudm types lower to !llvm.ptr (opaque handles)
class CudmTypeConverter : public TypeConverter {
public:
  CudmTypeConverter(MLIRContext *ctx) {
    auto ptrTy = LLVM::LLVMPointerType::get(ctx);
    addConversion([](Type t) { return t; });
    addConversion([ptrTy](cudm::HandleType) -> Type { return ptrTy; });
    addConversion([ptrTy](cudm::StateType) -> Type { return ptrTy; });
    addConversion([ptrTy](cudm::WorkspaceType) -> Type { return ptrTy; });
    addConversion([ptrTy](cudm::ElementaryOpType) -> Type { return ptrTy; });
    addConversion([ptrTy](cudm::OpTermType) -> Type { return ptrTy; });
    addConversion([ptrTy](cudm::OperatorType) -> Type { return ptrTy; });
    addConversion([ptrTy](cudm::ExpectationType) -> Type { return ptrTy; });
  }
};

// Helper: get or insert an extern func decl returning i32
static LLVM::LLVMFuncOp getOrInsertRuntimeFn(ModuleOp module, OpBuilder &b,
                                             StringRef name, Type retTy,
                                             ArrayRef<Type> argTys) {
  if (auto fn = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return fn;
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  auto fnTy = LLVM::LLVMFunctionType::get(retTy, argTys);
  return LLVM::LLVMFuncOp::create(b, module.getLoc(), name, fnTy);
}

// Helpers for common LLVM types
static Type i32(MLIRContext *c) { return IntegerType::get(c, 32); }
static Type i64(MLIRContext *c) { return IntegerType::get(c, 64); }
static Type f64(MLIRContext *c) { return Float64Type::get(c); }
static Type ptr(MLIRContext *c) { return LLVM::LLVMPointerType::get(c); }

// ---- InitHandleOp -> llvm.call @cudm_init ----
struct InitHandleLowering : public OpConversionPattern<cudm::InitHandleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::InitHandleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    // Allocate stack space for handle pointer, call cudm_init(&handle)
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                        rewriter.getI64IntegerAttr(1));
    auto handleSlot =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), ptr(ctx), one);
    auto fn = getOrInsertRuntimeFn(module, rewriter, "cudm_init", i32(ctx),
                                   {ptr(ctx)});
    LLVM::CallOp::create(rewriter, loc, fn, ValueRange{handleSlot});
    auto handle = LLVM::LoadOp::create(rewriter, loc, ptr(ctx), handleSlot);
    rewriter.replaceOp(op, handle.getResult());
    return success();
  }
};

// ---- DestroyHandleOp -> llvm.call @cudm_destroy ----
struct DestroyHandleLowering
    : public OpConversionPattern<cudm::DestroyHandleOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::DestroyHandleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto fn = getOrInsertRuntimeFn(module, rewriter, "cudm_destroy", i32(ctx),
                                   {ptr(ctx)});
    LLVM::CallOp::create(rewriter, op.getLoc(), fn, adaptor.getHandle());
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- CreateStateOp -> llvm.call @cudm_state_alloc ----
struct CreateStateLowering : public OpConversionPattern<cudm::CreateStateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::CreateStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto modeExtents = op.getModeExtents();
    int32_t numModes = modeExtents.size();
    int32_t purity = static_cast<int32_t>(op.getPurity());
    int32_t dataType = static_cast<int32_t>(op.getDataType());

    // Allocate stack arrays for mode_extents
    auto numModesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(numModes));
    auto extentsPtr =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), i64(ctx), numModesVal);
    for (int i = 0; i < numModes; i++) {
      auto idx = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                          rewriter.getI64IntegerAttr(i));
      auto elemPtr = LLVM::GEPOp::create(rewriter, loc, ptr(ctx), i64(ctx),
                                         extentsPtr, ValueRange{idx});
      auto val = LLVM::ConstantOp::create(
          rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(modeExtents[i]));
      LLVM::StoreOp::create(rewriter, loc, val, elemPtr);
    }

    // &state
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                        rewriter.getI64IntegerAttr(1));
    auto stateSlot =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), ptr(ctx), one);

    auto numModesI32 = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(numModes));
    auto purityI32 = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(purity));
    auto dtypeI32 = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(dataType));

    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "cudm_state_alloc", i32(ctx),
        {ptr(ctx), ptr(ctx), ptr(ctx), i32(ctx), i32(ctx), i32(ctx)});
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{adaptor.getHandle(), stateSlot, extentsPtr,
                                    numModesI32, purityI32, dtypeI32});
    auto state = LLVM::LoadOp::create(rewriter, loc, ptr(ctx), stateSlot);
    rewriter.replaceOp(op, state.getResult());
    return success();
  }
};

// ---- DestroyStateOp -> llvm.call @cudm_state_destroy ----
struct DestroyStateLowering : public OpConversionPattern<cudm::DestroyStateOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::DestroyStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto fn = getOrInsertRuntimeFn(module, rewriter, "cudm_state_destroy",
                                   i32(ctx), {ptr(ctx)});
    LLVM::CallOp::create(rewriter, op.getLoc(), fn, adaptor.getState());
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- CreateWorkspaceOp -> llvm.call @cudm_workspace_create ----
struct CreateWorkspaceLowering
    : public OpConversionPattern<cudm::CreateWorkspaceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::CreateWorkspaceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto one = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                        rewriter.getI64IntegerAttr(1));
    auto wsSlot =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), ptr(ctx), one);
    auto fn = getOrInsertRuntimeFn(module, rewriter, "cudm_workspace_create",
                                   i32(ctx), {ptr(ctx), ptr(ctx)});
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{adaptor.getHandle(), wsSlot});
    auto ws = LLVM::LoadOp::create(rewriter, loc, ptr(ctx), wsSlot);
    rewriter.replaceOp(op, ws.getResult());
    return success();
  }
};

// ---- DestroyWorkspaceOp -> llvm.call @cudm_workspace_destroy ----
struct DestroyWorkspaceLowering
    : public OpConversionPattern<cudm::DestroyWorkspaceOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::DestroyWorkspaceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto fn = getOrInsertRuntimeFn(module, rewriter, "cudm_workspace_destroy",
                                   i32(ctx), {ptr(ctx)});
    LLVM::CallOp::create(rewriter, op.getLoc(), fn, adaptor.getWorkspace());
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- CreateOperatorOp -> llvm.call @cudm_operator_create ----
struct CreateOperatorLowering
    : public OpConversionPattern<cudm::CreateOperatorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::CreateOperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();
    auto modeExtents = op.getModeExtents();
    int32_t numModes = modeExtents.size();

    auto numModesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(numModes));
    auto extentsPtr =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), i64(ctx), numModesVal);
    for (int i = 0; i < numModes; i++) {
      auto idx = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                          rewriter.getI64IntegerAttr(i));
      auto elemPtr = LLVM::GEPOp::create(rewriter, loc, ptr(ctx), i64(ctx),
                                         extentsPtr, ValueRange{idx});
      auto val = LLVM::ConstantOp::create(
          rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(modeExtents[i]));
      LLVM::StoreOp::create(rewriter, loc, val, elemPtr);
    }

    auto one = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                        rewriter.getI64IntegerAttr(1));
    auto opSlot =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), ptr(ctx), one);
    auto numModesI32 = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(numModes));
    auto fn =
        getOrInsertRuntimeFn(module, rewriter, "cudm_operator_create", i32(ctx),
                             {ptr(ctx), ptr(ctx), ptr(ctx), i32(ctx)});
    LLVM::CallOp::create(
        rewriter, loc, fn,
        ValueRange{adaptor.getHandle(), opSlot, extentsPtr, numModesI32});
    auto result = LLVM::LoadOp::create(rewriter, loc, ptr(ctx), opSlot);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

// ---- DestroyOperatorOp -> llvm.call @cudm_operator_destroy ----
struct DestroyOperatorLowering
    : public OpConversionPattern<cudm::DestroyOperatorOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::DestroyOperatorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto fn = getOrInsertRuntimeFn(module, rewriter, "cudm_operator_destroy",
                                   i32(ctx), {ptr(ctx)});
    LLVM::CallOp::create(rewriter, op.getLoc(), fn, adaptor.getOp());
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- CreateElementaryOpOp -> llvm.call @cudm_elementary_op_create ----
struct CreateElementaryOpLowering
    : public OpConversionPattern<cudm::CreateElementaryOpOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::CreateElementaryOpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto one = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                        rewriter.getI64IntegerAttr(1));
    auto elemSlot =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), ptr(ctx), one);

    auto tensorConstant = op.getTensorData().getDefiningOp<arith::ConstantOp>();
    if (!tensorConstant)
      return rewriter.notifyMatchFailure(
          op, "elementary operator data must be an arith constant");
    auto tensorValues =
        dyn_cast<DenseFPElementsAttr>(tensorConstant.getValue());
    if (!tensorValues)
      return rewriter.notifyMatchFailure(
          op, "elementary operator data must contain floating-point values");

    const auto numValues = tensorValues.getNumElements();
    auto numValuesConstant = LLVM::ConstantOp::create(
        rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(numValues));
    auto tensorData = LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), f64(ctx),
                                             numValuesConstant);
    int64_t index = 0;
    for (const auto &value : tensorValues.getValues<APFloat>()) {
      auto indexConstant = LLVM::ConstantOp::create(
          rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(index++));
      auto elementPointer =
          LLVM::GEPOp::create(rewriter, loc, ptr(ctx), f64(ctx), tensorData,
                              ValueRange{indexConstant});
      auto elementValue = LLVM::ConstantOp::create(
          rewriter, loc, f64(ctx),
          rewriter.getF64FloatAttr(value.convertToDouble()));
      LLVM::StoreOp::create(rewriter, loc, elementValue, elementPointer);
    }

    auto modeExtents = op.getModeExtents();
    auto numModes = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(modeExtents.size())));
    auto extentCount = LLVM::ConstantOp::create(
        rewriter, loc, i64(ctx),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(modeExtents.size())));
    auto extents =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), i64(ctx), extentCount);
    for (auto [i, extent] : llvm::enumerate(modeExtents)) {
      auto indexValue = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                                 rewriter.getI64IntegerAttr(i));
      auto element =
          LLVM::GEPOp::create(rewriter, loc, ptr(ctx), i64(ctx), extents,
                              ValueRange{indexValue.getResult()});
      auto extentValue = LLVM::ConstantOp::create(
          rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(extent));
      LLVM::StoreOp::create(rewriter, loc, extentValue, element);
    }
    auto dataType = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getDataType())));

    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "cudm_elementary_op_create", i32(ctx),
        {ptr(ctx), ptr(ctx), ptr(ctx), i64(ctx), ptr(ctx), i32(ctx), i32(ctx)});
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{adaptor.getHandle(), elemSlot, tensorData,
                                    numValuesConstant, extents, numModes,
                                    dataType});
    auto elem = LLVM::LoadOp::create(rewriter, loc, ptr(ctx), elemSlot);
    rewriter.replaceOp(op, elem.getResult());
    return success();
  }
};

// ---- DestroyElementaryOpOp -> llvm.call @cudm_elementary_op_destroy ----
struct DestroyElementaryOpLowering
    : public OpConversionPattern<cudm::DestroyElementaryOpOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::DestroyElementaryOpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "cudm_elementary_op_destroy", i32(ctx), {ptr(ctx)});
    LLVM::CallOp::create(rewriter, op.getLoc(), fn, adaptor.getElemOp());
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- CreateOpTermOp -> llvm.call @cudm_op_term_create ----
struct CreateOpTermLowering : public OpConversionPattern<cudm::CreateOpTermOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::CreateOpTermOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto one = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                        rewriter.getI64IntegerAttr(1));
    auto termSlot =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), ptr(ctx), one);
    auto modeExtents = op.getModeExtents();
    auto numModes = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(modeExtents.size())));
    auto extentCount = LLVM::ConstantOp::create(
        rewriter, loc, i64(ctx),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(modeExtents.size())));
    auto extents =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), i64(ctx), extentCount);
    for (auto [i, extent] : llvm::enumerate(modeExtents)) {
      auto indexValue = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                                 rewriter.getI64IntegerAttr(i));
      auto element =
          LLVM::GEPOp::create(rewriter, loc, ptr(ctx), i64(ctx), extents,
                              ValueRange{indexValue.getResult()});
      auto extentValue = LLVM::ConstantOp::create(
          rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(extent));
      LLVM::StoreOp::create(rewriter, loc, extentValue, element);
    }
    auto fn =
        getOrInsertRuntimeFn(module, rewriter, "cudm_op_term_create", i32(ctx),
                             {ptr(ctx), ptr(ctx), ptr(ctx), i32(ctx)});
    LLVM::CallOp::create(
        rewriter, loc, fn,
        ValueRange{adaptor.getHandle(), termSlot, extents, numModes});
    auto term = LLVM::LoadOp::create(rewriter, loc, ptr(ctx), termSlot);
    rewriter.replaceOp(op, term.getResult());
    return success();
  }
};

struct DestroyOpTermLowering
    : public OpConversionPattern<cudm::DestroyOpTermOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::DestroyOpTermOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto fn = getOrInsertRuntimeFn(module, rewriter, "cudm_op_term_destroy",
                                   i32(ctx), {ptr(ctx)});
    LLVM::CallOp::create(rewriter, op.getLoc(), fn, adaptor.getOpTerm());
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- AppendElementaryProductOp -> llvm.call @cudm_op_term_append ----
struct AppendElementaryProductLowering
    : public OpConversionPattern<cudm::AppendElementaryProductOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::AppendElementaryProductOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto coeffReal = LLVM::ConstantOp::create(rewriter, loc, f64(ctx),
                                              op.getCoeffRealAttr());
    auto coeffImag = LLVM::ConstantOp::create(rewriter, loc, f64(ctx),
                                              op.getCoeffImagAttr());

    int32_t callbackKind = 0;
    ArrayRef<double> callbackParameters;
    if (auto kind = op->getAttrOfType<IntegerAttr>("cudm.callback_kind"))
      callbackKind = static_cast<int32_t>(kind.getInt());
    if (auto parameters =
            op->getAttrOfType<DenseF64ArrayAttr>("cudm.callback_params"))
      callbackParameters = parameters.asArrayRef();
    auto callbackKindValue = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(callbackKind));
    auto callbackCountValue = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx),
        rewriter.getI32IntegerAttr(callbackParameters.size()));
    auto callbackAllocationCount =
        LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                 rewriter.getI64IntegerAttr(std::max<size_t>(
                                     1, callbackParameters.size())));
    auto callbackData = LLVM::AllocaOp::create(
        rewriter, loc, ptr(ctx), f64(ctx), callbackAllocationCount);
    for (auto [i, parameter] : llvm::enumerate(callbackParameters)) {
      auto index = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                            rewriter.getI64IntegerAttr(i));
      auto element =
          LLVM::GEPOp::create(rewriter, loc, ptr(ctx), f64(ctx), callbackData,
                              ValueRange{index.getResult()});
      auto value = LLVM::ConstantOp::create(
          rewriter, loc, f64(ctx), rewriter.getF64FloatAttr(parameter));
      LLVM::StoreOp::create(rewriter, loc, value, element);
    }

    int32_t numElems = adaptor.getElemOps().size();
    auto numElemsVal = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(numElems));

    auto numElemsI64 = LLVM::ConstantOp::create(
        rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(numElems));
    auto elemOps =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), ptr(ctx), numElemsI64);
    auto modes =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), i32(ctx), numElemsI64);
    auto duality =
        LLVM::AllocaOp::create(rewriter, loc, ptr(ctx), i32(ctx), numElemsI64);
    auto modeValues = op.getModesActedOn();
    auto dualityValues = op.getDuality();
    if (modeValues.size() != static_cast<size_t>(numElems) ||
        dualityValues.size() != static_cast<size_t>(numElems))
      return rewriter.notifyMatchFailure(
          op, "mode and duality arrays must match elementary operator count");
    for (int32_t i = 0; i < numElems; ++i) {
      auto index = LLVM::ConstantOp::create(rewriter, loc, i64(ctx),
                                            rewriter.getI64IntegerAttr(i));
      auto opElement =
          LLVM::GEPOp::create(rewriter, loc, ptr(ctx), ptr(ctx), elemOps,
                              ValueRange{index.getResult()});
      LLVM::StoreOp::create(rewriter, loc, adaptor.getElemOps()[i], opElement);
      auto modeElement =
          LLVM::GEPOp::create(rewriter, loc, ptr(ctx), i32(ctx), modes,
                              ValueRange{index.getResult()});
      auto modeValue = LLVM::ConstantOp::create(
          rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(modeValues[i]));
      LLVM::StoreOp::create(rewriter, loc, modeValue, modeElement);
      auto dualityElement =
          LLVM::GEPOp::create(rewriter, loc, ptr(ctx), i32(ctx), duality,
                              ValueRange{index.getResult()});
      auto dualityValue = LLVM::ConstantOp::create(
          rewriter, loc, i32(ctx),
          rewriter.getI32IntegerAttr(dualityValues[i]));
      LLVM::StoreOp::create(rewriter, loc, dualityValue, dualityElement);
    }

    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "cudm_op_term_append", i32(ctx),
        {ptr(ctx), ptr(ctx), ptr(ctx), ptr(ctx), ptr(ctx), i32(ctx), f64(ctx),
         f64(ctx), i32(ctx), ptr(ctx), i32(ctx)});
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{adaptor.getHandle(), adaptor.getOpTerm(),
                                    elemOps, modes, duality, numElemsVal,
                                    coeffReal, coeffImag, callbackKindValue,
                                    callbackData, callbackCountValue});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- OperatorAppendTermOp -> llvm.call @cudm_operator_append ----
struct OperatorAppendTermLowering
    : public OpConversionPattern<cudm::OperatorAppendTermOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::OperatorAppendTermOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    auto duality =
        LLVM::ConstantOp::create(rewriter, loc, i32(ctx), op.getDualityAttr());
    auto coeffReal = LLVM::ConstantOp::create(rewriter, loc, f64(ctx),
                                              op.getCoeffRealAttr());
    auto coeffImag = LLVM::ConstantOp::create(rewriter, loc, f64(ctx),
                                              op.getCoeffImagAttr());

    auto fn = getOrInsertRuntimeFn(
        module, rewriter, "cudm_operator_append", i32(ctx),
        {ptr(ctx), ptr(ctx), ptr(ctx), i32(ctx), f64(ctx), f64(ctx)});
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{adaptor.getHandle(), adaptor.getOp(),
                                    adaptor.getTerm(), duality, coeffReal,
                                    coeffImag});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---- EvolveOp -> one runtime integration call + host result capture ----
struct EvolveLowering : public OpConversionPattern<cudm::EvolveOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(cudm::EvolveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto *ctx = rewriter.getContext();
    auto loc = op.getLoc();

    double tStart = op.getTStart().convertToDouble();
    double tEnd = op.getTEnd().convertToDouble();
    int64_t numSteps = op.getNumSteps();
    int32_t integrator = static_cast<int32_t>(op.getIntegrator());

    auto fn =
        getOrInsertRuntimeFn(module, rewriter, "cudm_evolve", i32(ctx),
                             {ptr(ctx), ptr(ctx), ptr(ctx), ptr(ctx), ptr(ctx),
                              f64(ctx), f64(ctx), i64(ctx), i32(ctx)});

    auto integratorVal = LLVM::ConstantOp::create(
        rewriter, loc, i32(ctx), rewriter.getI32IntegerAttr(integrator));
    auto startVal = LLVM::ConstantOp::create(rewriter, loc, f64(ctx),
                                             rewriter.getF64FloatAttr(tStart));
    auto endVal = LLVM::ConstantOp::create(rewriter, loc, f64(ctx),
                                           rewriter.getF64FloatAttr(tEnd));
    auto stepsVal = LLVM::ConstantOp::create(
        rewriter, loc, i64(ctx), rewriter.getI64IntegerAttr(numSteps));
    LLVM::CallOp::create(rewriter, loc, fn,
                         ValueRange{adaptor.getHandle(), adaptor.getOp(),
                                    adaptor.getStateIn(), adaptor.getStateOut(),
                                    adaptor.getWorkspace(), startVal, endVal,
                                    stepsVal, integratorVal});

    auto capture = getOrInsertRuntimeFn(module, rewriter, "cudm_state_capture",
                                        i32(ctx), {ptr(ctx)});
    LLVM::CallOp::create(rewriter, loc, capture, adaptor.getStateOut());

    // The result is state_out (same pointer)
    rewriter.replaceOp(op, adaptor.getStateOut());
    return success();
  }
};

// ---- The pass itself ----
struct CuDensityMatToLLVMPass
    : public PassWrapper<CuDensityMatToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CuDensityMatToLLVMPass)

  StringRef getArgument() const final { return "cudm-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower cudm dialect ops to LLVM IR call sequences targeting "
           "libcudm-runtime";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    CudmTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<cudm::CuDensityMatDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<InitHandleLowering, DestroyHandleLowering, CreateStateLowering,
                 DestroyStateLowering, CreateWorkspaceLowering,
                 DestroyWorkspaceLowering, CreateOperatorLowering,
                 DestroyOperatorLowering, CreateElementaryOpLowering,
                 DestroyElementaryOpLowering, CreateOpTermLowering,
                 DestroyOpTermLowering, AppendElementaryProductLowering,
                 OperatorAppendTermLowering, EvolveLowering>(typeConverter,
                                                             ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace cudm {

std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass() {
  return std::make_unique<CuDensityMatToLLVMPass>();
}

} // namespace cudm
