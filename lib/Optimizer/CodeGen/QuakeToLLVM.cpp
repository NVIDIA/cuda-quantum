/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/QuakeToLLVM.h"
#include "CodeGenOps.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

#define DEBUG_TYPE "quake-to-llvm"

using namespace mlir;

namespace {
//===----------------------------------------------------------------------===//
// Conversion patterns for Quake dialect ops.
//===----------------------------------------------------------------------===//

/// Lowers Quake AllocaOp to QIR function call in LLVM.
class AllocaOpRewrite : public ConvertOpToLLVMPattern<quake::AllocaOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::AllocaOp alloca, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = alloca->getLoc();
    auto parentModule = alloca->getParentOfType<ModuleOp>();
    auto *context = parentModule->getContext();

    // If this alloc is just returning a qubit
    if (auto resultType =
            dyn_cast_if_present<quake::RefType>(alloca.getResult().getType())) {

      StringRef qirQubitAllocate = cudaq::opt::QIRQubitAllocate;
      auto qubitType = cudaq::opt::getQubitType(context);
      FlatSymbolRefAttr symbolRef =
          cudaq::opt::factory::createLLVMFunctionSymbol(
              qirQubitAllocate, qubitType, {}, parentModule);

      rewriter.replaceOpWithNewOp<LLVM::CallOp>(alloca, qubitType, symbolRef,
                                                std::nullopt);
      return success();
    }

    // Create a QIR call to allocate the qubits.
    StringRef qir_qubit_array_allocate = cudaq::opt::QIRArrayQubitAllocateArray;
    auto array_qbit_type = cudaq::opt::getArrayType(context);
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qir_qubit_array_allocate, array_qbit_type, {rewriter.getI64Type()},
        parentModule);

    // AllocaOp could have a size operand, or the size could
    // be compile time known and encoded in the veq return type.
    Value sizeOperand;
    if (adaptor.getOperands().empty()) {
      auto type = alloca.getResult().getType().cast<quake::VeqType>();
      auto constantSize = type.getSize();
      sizeOperand =
          rewriter.create<arith::ConstantIntOp>(loc, constantSize, 64);
    } else {
      sizeOperand = adaptor.getOperands().front();
      if (sizeOperand.getType().cast<IntegerType>().getWidth() < 64) {
        sizeOperand = rewriter.create<LLVM::ZExtOp>(loc, rewriter.getI64Type(),
                                                    sizeOperand);
      }
    }

    // Replace the AllocaOp with the QIR call.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(alloca, array_qbit_type,
                                              symbolRef, sizeOperand);
    return success();
  }
};

// Lower codegen.qmem_raii to a QIR function to allocate the qubits with the
// provided state vector.
class QmemRAIIOpRewrite
    : public ConvertOpToLLVMPattern<cudaq::codegen::RAIIOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(cudaq::codegen::RAIIOp raii, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = raii->getLoc();
    auto parentModule = raii->getParentOfType<ModuleOp>();
    auto array_qbit_type = cudaq::opt::getArrayType(rewriter.getContext());

    // Get the CC Pointer for the state
    auto ccState = adaptor.getInitState();

    // Inspect the element type of the complex data, need to
    // know if its f32 or f64
    StringRef functionName;
    Type eleTy = raii.getInitElementType();
    if (auto elePtrTy = dyn_cast<cudaq::cc::PointerType>(eleTy))
      eleTy = elePtrTy.getElementType();
    if (auto arrayTy = dyn_cast<cudaq::cc::ArrayType>(eleTy))
      eleTy = arrayTy.getElementType();
    bool fromComplex = false;
    if (auto complexTy = dyn_cast<ComplexType>(eleTy)) {
      fromComplex = true;
      eleTy = complexTy.getElementType();
    }
    if (isa<cudaq::cc::StateType>(eleTy))
      functionName = cudaq::opt::QIRArrayQubitAllocateArrayWithCudaqStatePtr;
    if (eleTy == rewriter.getF64Type())
      functionName =
          fromComplex ? cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex64
                      : cudaq::opt::QIRArrayQubitAllocateArrayWithStateFP64;
    if (eleTy == rewriter.getF32Type())
      functionName =
          fromComplex ? cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex32
                      : cudaq::opt::QIRArrayQubitAllocateArrayWithStateFP32;
    if (functionName.empty())
      return raii.emitOpError("invalid type on initialize state operation, "
                              "must be complex floating point.");

    // Get the size of the qubit register
    Type allocTy = adaptor.getAllocType();
    auto allocSize = adaptor.getAllocSize();
    Value sizeOperand;
    auto i64Ty = rewriter.getI64Type();
    if (allocSize) {
      sizeOperand = allocSize;
      auto sizeTy = cast<IntegerType>(sizeOperand.getType());
      if (sizeTy.getWidth() < 64)
        sizeOperand = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, sizeOperand);
      else if (sizeTy.getWidth() > 64)
        sizeOperand = rewriter.create<LLVM::TruncOp>(loc, i64Ty, sizeOperand);
    } else {
      auto type = cast<quake::VeqType>(allocTy);
      auto constantSize = type.getSize();
      sizeOperand =
          rewriter.create<arith::ConstantIntOp>(loc, constantSize, 64);
    }

    // Create QIR allocation with initializer function.
    auto *ctx = rewriter.getContext();
    auto ptrTy = cudaq::opt::factory::getPointerType(ctx);

    FlatSymbolRefAttr raiiSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            functionName, array_qbit_type, {i64Ty, ptrTy}, parentModule);

    // Call the allocation function
    Value castedInitState =
        rewriter.create<LLVM::BitcastOp>(loc, ptrTy, ccState);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        raii, array_qbit_type, raiiSymbolRef,
        ArrayRef<Value>{sizeOperand, castedInitState});
    return success();
  }
};

/// Lower Quake Dealloc Ops to QIR function calls.
class DeallocOpRewrite : public ConvertOpToLLVMPattern<quake::DeallocOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::DeallocOp dealloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentModule = dealloc->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    auto retType = LLVM::LLVMVoidType::get(context);

    // Could be a dealloc on a ref or a veq
    StringRef qirQuantumDeallocateFunc;
    Type operandType, qType = dealloc.getOperand().getType();
    if (qType.isa<quake::VeqType>()) {
      qirQuantumDeallocateFunc = cudaq::opt::QIRArrayQubitReleaseArray;
      operandType = cudaq::opt::getArrayType(context);
    } else {
      qirQuantumDeallocateFunc = cudaq::opt::QIRArrayQubitReleaseQubit;
      operandType = cudaq::opt::getQubitType(context);
    }

    FlatSymbolRefAttr deallocSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            qirQuantumDeallocateFunc, retType, {operandType}, parentModule);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(dealloc, ArrayRef<Type>({}),
                                              deallocSymbolRef,
                                              adaptor.getOperands().front());
    return success();
  }
};

// Convert a quake.concat op to QIR code.
//
// This implementation could be improved for efficiency in some cases. For
// example, if the size (in qubits) for all the arguments is constant, then the
// result Array could be allocated and each qubit written into it with no need
// for further reallocations, etc. (These opportunities are left as TODOs.) In
// general, quake.concat does not guarantee that the sizes of the input Veq is
// a compile-time constant however.
class ConcatOpRewrite : public ConvertOpToLLVMPattern<quake::ConcatOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::ConcatOp concat, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentModule = concat->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    auto arrType = cudaq::opt::getArrayType(context);
    auto loc = concat.getLoc();

    StringRef qirArrayConcatName = cudaq::opt::QIRArrayConcatArray;
    FlatSymbolRefAttr concatFunc =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            qirArrayConcatName, arrType, {arrType, arrType}, parentModule);

    if (adaptor.getOperands().empty()) {
      rewriter.eraseOp(concat);
      return success();
    }

    auto qirArrayTy = cudaq::opt::getArrayType(context);
    auto i8PtrTy = cudaq::opt::factory::getPointerType(context);
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRArrayCreateArray, qirArrayTy,
        {rewriter.getI32Type(), rewriter.getI64Type()}, parentModule);
    FlatSymbolRefAttr getSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::QIRArrayGetElementPtr1d, i8PtrTy,
            {qirArrayTy, rewriter.getIntegerType(64)}, parentModule);
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    // FIXME: 8 bytes is assumed to be the sizeof(char*) on the target machine.
    Value eight = rewriter.create<arith::ConstantIntOp>(loc, 8, 32);
    // Function to convert a QIR Qubit value to an Array value.
    auto wrapQubitInArray = [&](Value v) -> Value {
      if (v.getType() != cudaq::opt::getQubitType(context))
        return v;
      auto createCall = rewriter.create<LLVM::CallOp>(
          loc, qirArrayTy, symbolRef, ArrayRef<Value>{eight, one});
      auto result = createCall.getResult();
      auto call = rewriter.create<LLVM::CallOp>(loc, i8PtrTy, getSymbolRef,
                                                ArrayRef<Value>{result, zero});
      Value pointer = rewriter.create<LLVM::BitcastOp>(
          loc, cudaq::opt::factory::getPointerType(i8PtrTy), call.getResult());
      auto cast = rewriter.create<LLVM::BitcastOp>(loc, i8PtrTy, v);
      rewriter.create<LLVM::StoreOp>(loc, cast, pointer);
      return result;
    };

    // Loop over all the arguments to the concat operator and glue them
    // together, converting Qubit values to Array values as needed.
    auto frontArr = wrapQubitInArray(adaptor.getOperands().front());
    for (auto oper : adaptor.getOperands().drop_front(1)) {
      auto backArr = wrapQubitInArray(oper);
      auto glue = rewriter.create<LLVM::CallOp>(
          loc, qirArrayTy, concatFunc, ArrayRef<Value>{frontArr, backArr});
      frontArr = glue.getResult();
    }
    rewriter.replaceOp(concat, frontArr);
    return success();
  }
};

class DiscriminateOpPattern
    : public ConvertOpToLLVMPattern<quake::DiscriminateOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::DiscriminateOp discr, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto m = discr.getMeasurement();
    rewriter.replaceOp(discr, m);
    return success();
  }
};

/// Convert a ExtractRefOp to the respective QIR.
class ExtractQubitOpRewrite
    : public ConvertOpToLLVMPattern<quake::ExtractRefOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::ExtractRefOp extract, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = extract->getLoc();
    auto parentModule = extract->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    auto qir_array_get_element_ptr_1d = cudaq::opt::QIRArrayGetElementPtr1d;

    auto array_qbit_type = cudaq::opt::getArrayType(context);
    auto qbit_element_ptr_type =
        LLVM::LLVMPointerType::get(rewriter.getI8Type());

    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qir_array_get_element_ptr_1d, qbit_element_ptr_type,
        {array_qbit_type, rewriter.getI64Type()}, parentModule);

    Value idx_operand;
    auto i64Ty = rewriter.getI64Type();
    if (extract.hasConstantIndex()) {
      idx_operand = rewriter.create<arith::ConstantIntOp>(
          loc, extract.getConstantIndex(), i64Ty);
    } else {
      idx_operand = adaptor.getOperands()[1];

      if (idx_operand.getType().isIntOrFloat() &&
          idx_operand.getType().cast<IntegerType>().getWidth() < 64)
        idx_operand = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, idx_operand);
    }

    auto get_qbit_qir_call = rewriter.create<LLVM::CallOp>(
        loc, qbit_element_ptr_type, symbolRef,
        llvm::ArrayRef({adaptor.getOperands().front(), idx_operand}));

    auto bitcast = rewriter.create<LLVM::BitcastOp>(
        loc, LLVM::LLVMPointerType::get(cudaq::opt::getQubitType(context)),
        get_qbit_qir_call.getResult());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(
        extract, cudaq::opt::getQubitType(context), bitcast.getResult());
    return success();
  }
};

class SubveqOpRewrite : public ConvertOpToLLVMPattern<quake::SubVeqOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::SubVeqOp subveq, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subveq->getLoc();
    auto parentModule = subveq->getParentOfType<ModuleOp>();
    auto *context = parentModule->getContext();
    constexpr auto rtSubveqFuncName = cudaq::opt::QIRArraySlice;
    auto arrayTy = cudaq::opt::getArrayType(context);
    auto resultTy = arrayTy;

    auto i32Ty = rewriter.getI32Type();
    auto i64Ty = rewriter.getI64Type();
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        rtSubveqFuncName, arrayTy, {arrayTy, i32Ty, i64Ty, i64Ty, i64Ty},
        parentModule);

    Value lowArg = adaptor.getOperands()[1];
    Value highArg = adaptor.getOperands()[2];
    auto extend = [&](Value &v) -> Value {
      if (v.getType().isa<IntegerType>() &&
          v.getType().cast<IntegerType>().getWidth() < 64)
        return rewriter.create<LLVM::ZExtOp>(loc, i64Ty, v);
      return v;
    };
    lowArg = extend(lowArg);
    highArg = extend(highArg);
    Value inArr = adaptor.getOperands()[0];
    auto one32 = rewriter.create<arith::ConstantIntOp>(loc, 1, i32Ty);
    auto one64 = rewriter.create<arith::ConstantIntOp>(loc, 1, i64Ty);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        subveq, resultTy, symbolRef,
        ValueRange{inArr, one32, lowArg, one64, highArg});
    return success();
  }
};

/// Lower the quake.reset op to QIR
class ResetRewrite : public ConvertOpToLLVMPattern<quake::ResetOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::ResetOp instOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentModule = instOp->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    std::string instName = instOp->getName().stripDialect().str();

    // Get the reset QIR function name
    auto qirFunctionName = cudaq::opt::QIRQISPrefix + instName;

    // Create the qubit pointer type
    auto qirQubitPointerType = cudaq::opt::getQubitType(context);

    // Get the function reference for the reset function
    auto qirFunctionSymbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, LLVM::LLVMVoidType::get(context),
        {qirQubitPointerType}, parentModule);

    // Replace the quake op with the new call op.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        instOp, TypeRange{}, qirFunctionSymbolRef, adaptor.getOperands());
    return success();
  }
};

/// Lower exp_pauli(f64, veq, cc.string) to __quantum__qis__exp_pauli
class ExpPauliRewrite : public ConvertOpToLLVMPattern<quake::ExpPauliOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::ExpPauliOp instOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instOp->getLoc();
    auto parentModule = instOp->getParentOfType<ModuleOp>();
    auto *context = rewriter.getContext();
    std::string qirQisPrefix{cudaq::opt::QIRQISPrefix};
    auto qirFunctionName = qirQisPrefix + "exp_pauli";
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
        {rewriter.getF64Type(), cudaq::opt::getArrayType(context),
         cudaq::opt::factory::getPointerType(context)},
        parentModule);
    SmallVector<Value> operands = adaptor.getOperands();
    // First need to check the type of the Pauli word. We expect
    // a pauli_word directly `{i8*,i64}` or a string literal
    // `ptr<i8>`. If it is a string literal, we need to map it to
    // a pauli word.
    auto pauliWord = operands.back();
    if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(pauliWord.getType())) {
      // Make sure we have the right types to extract the
      // length of the string literal
      auto ptrEleTy = ptrTy.getElementType();
      auto innerArrTy = dyn_cast<LLVM::LLVMArrayType>(ptrEleTy);
      if (!innerArrTy)
        return instOp.emitError(
            "exp_pauli string literal expected to be ptr<array<i8 x N.>.");

      // Get the number of elements in the provided string literal
      auto numElements = innerArrTy.getNumElements() - 1;

      // Remove the old operand
      operands.pop_back();

      // We must create the {i8*, i64} struct from the string literal
      SmallVector<Type> structTys{
          LLVM::LLVMPointerType::get(rewriter.getI8Type()),
          rewriter.getI64Type()};
      auto structTy = LLVM::LLVMStructType::getLiteral(context, structTys);

      // Allocate the char span struct
      Value alloca = cudaq::opt::factory::createLLVMTemporary(
          loc, rewriter, LLVM::LLVMPointerType::get(structTy));

      // We'll need these constants
      auto zero = cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, 0);
      auto one = cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, 1);
      auto size =
          cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, numElements);

      // Set the string literal data
      auto strPtr = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getI8Type()), alloca,
          ValueRange{zero, zero});
      auto castedPauli = rewriter.create<LLVM::BitcastOp>(
          loc, cudaq::opt::factory::getPointerType(context), pauliWord);
      rewriter.create<LLVM::StoreOp>(loc, castedPauli, strPtr);

      // Set the integer length
      auto intPtr = rewriter.create<LLVM::GEPOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getI64Type()), alloca,
          ValueRange{zero, one});
      rewriter.create<LLVM::StoreOp>(loc, size, intPtr);

      // Cast to raw opaque pointer
      auto castedStore = rewriter.create<LLVM::BitcastOp>(
          loc, cudaq::opt::factory::getPointerType(context), alloca);
      operands.push_back(castedStore);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{}, symbolRef,
                                                operands);
      return success();
    }

    // Here we know we have a pauli word expressed as `{i8*, i64}`.
    // Allocate a stack slot for it and store what we have to that pointer,
    // pass the pointer to NVQIR
    Value alloca = cudaq::opt::factory::createLLVMTemporary(
        loc, rewriter, LLVM::LLVMPointerType::get(pauliWord.getType()));
    rewriter.create<LLVM::StoreOp>(loc, pauliWord, alloca);
    auto castedPauli = rewriter.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(context), alloca);
    operands.pop_back();
    operands.push_back(castedPauli);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{}, symbolRef,
                                              operands);
    return success();
  }
};

template <typename OP>
class ConvertOpWithControls : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewriteWithControls(OP instOp, typename Base::OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const {
    auto numControls = instOp.getControls().size();
    auto loc = instOp->getLoc();
    auto parentModule = instOp->template getParentOfType<ModuleOp>();
    auto *context = parentModule->getContext();
    std::string qirQisPrefix{cudaq::opt::QIRQISPrefix};
    std::string instName = instOp->getName().stripDialect().str();

    // Handle the case where we have and S or T gate,
    // but the adjoint has been requested.
    std::vector<std::string> filterNames{"s", "t"};
    if (std::find(filterNames.begin(), filterNames.end(), instName) !=
        filterNames.end())
      if (instOp.isAdj())
        instName = instName + "dg";

    // Convert the ctrl bits to an Array
    auto qirFunctionName = qirQisPrefix + instName + "__ctl";

    // Useful types we'll need
    auto qirArrayType = cudaq::opt::getArrayType(context);
    auto qirQubitPointerType = cudaq::opt::getQubitType(context);
    auto i64Type = rewriter.getI64Type();

    // __quantum__qis__NAME__ctl(Array*, Qubit*) Type
    SmallVector<Type> argTys = {qirArrayType, qirQubitPointerType};
    auto numTargetOperands = instOp.getTargets().size();
    if (numTargetOperands < 1 || numTargetOperands > 2)
      return failure();
    if (numTargetOperands == 2)
      argTys.push_back(qirQubitPointerType);
    auto instOpQISFunctionType =
        LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context), argTys);

    // Get the function pointer for the ctrl operation
    auto qirFunctionSymbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, LLVM::LLVMVoidType::get(context), argTys,
        parentModule);

    // Get the first control
    auto control = instOp.getControls().front();
    auto instOperands = adaptor.getOperands();
    if (numControls == 1 && isa<quake::VeqType>(control.getType())) {
      // Operands are already an Array* and Qubit*.
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          instOp, TypeRange{}, qirFunctionSymbolRef, instOperands);
      return success();
    }

    // Here we know we have multiple controls, we have to check if we have all
    // refs or a mix of ref / veq. If the latter we'll use a different runtime
    // function.
    FlatSymbolRefAttr applyMultiControlFunction;
    SmallVector<Value> args;
    Value ctrlOpPointer = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(instOpQISFunctionType),
        qirFunctionSymbolRef);
    Value numControlOperands =
        rewriter.create<LLVM::ConstantOp>(loc, i64Type, numControls);
    args.push_back(numControlOperands);

    // Check if all controls are qubit types, if so retain existing
    // functionality.
    auto allControlsAreQubits = [&]() {
      for (auto c : adaptor.getControls())
        if (c.getType() != qirQubitPointerType)
          return false;
      return true;
    }();
    if (allControlsAreQubits && numTargetOperands == 1) {
      // Conditionally use the `invokeWithControlQubits` runtime function. This
      // function is used instead of the more general one because it is used by
      // a peephole optimization.
      applyMultiControlFunction = cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::NVQIRInvokeWithControlBits,
          LLVM::LLVMVoidType::get(context),
          {i64Type, LLVM::LLVMPointerType::get(instOpQISFunctionType)},
          parentModule, true);
    } else {
      // Otherwise use the general function, which can handle registers of
      // qubits and multiple target qubits. Get symbol for the
      // `invokeWithControlRegisterOrQubits` runtime function.
      applyMultiControlFunction = cudaq::opt::factory::createLLVMFunctionSymbol(
          cudaq::opt::NVQIRInvokeWithControlRegisterOrBits,
          LLVM::LLVMVoidType::get(context),
          {i64Type, LLVM::LLVMPointerType::get(i64Type), i64Type,
           LLVM::LLVMPointerType::get(instOpQISFunctionType)},
          parentModule, true);

      // The total number of control qubits may be more than the number of
      // control operands, e.g. if ctrls = `{veq<2>, ref}` then there are 3
      // controls even though there are 2 operands. The i64 array encoding is
      // used to track the control operands. If control operand is a `ref` it
      // has a $0$ size. If control operand is a `veq<N>` it has size $N$. The
      // array created is the number of control operands in size, where the
      // $k$-th element is $N$ if the $k$-th control operand has type `veq<N>`,
      // and $0$ otherwise.
      Value isArrayAndLengthArr =
          cudaq::opt::factory::packIsArrayAndLengthArray(
              loc, rewriter, parentModule, numControls, adaptor.getControls());
      args.push_back(isArrayAndLengthArr);
      args.push_back(
          rewriter.create<LLVM::ConstantOp>(loc, i64Type, numTargetOperands));
    }
    args.push_back(ctrlOpPointer);
    args.append(instOperands.begin(), instOperands.end());

    // Call our utility function.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{},
                                              applyMultiControlFunction, args);

    return success();
  }
};

/// Lower single target Quantum ops with no parameter to QIR:
/// h, x, y, z, s, t
template <typename OP>
class OneTargetRewrite : public ConvertOpWithControls<OP> {
public:
  using Base = ConvertOpWithControls<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto numControls = instOp.getControls().size();
    auto parentModule = instOp->template getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    std::string qirQisPrefix{cudaq::opt::QIRQISPrefix};
    std::string instName = instOp->getName().stripDialect().str();

    if (numControls != 0) {
      // Handle the cases with controls.
      return Base::matchAndRewriteWithControls(instOp, adaptor, rewriter);
    }

    // There are no control bits, so call the function directly.
    auto qirFunctionName =
        qirQisPrefix + instName + (instOp.getIsAdj() ? "__adj" : "");
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
        {cudaq::opt::getQubitType(context)}, parentModule);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{}, symbolRef,
                                              adaptor.getOperands());
    return success();
  }
};

/// Lower single target Quantum ops with one parameter to QIR:
/// rx, ry, rz, p
template <typename OP>
class OneTargetOneParamRewrite : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = instOp->getName().stripDialect().str();
    auto numControls = instOp.getControls().size();
    auto instOperands = adaptor.getOperands();

    auto loc = instOp.getLoc();
    ModuleOp parentModule = instOp->template getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    std::string qirQisPrefix = cudaq::opt::QIRQISPrefix;
    auto qirFunctionName = qirQisPrefix + instName;

    auto qubitIndexType = cudaq::opt::getQubitType(context);
    auto qubitArrayType = cudaq::opt::getArrayType(context);
    auto paramType = FloatType::getF64(context);

    SmallVector<Value> funcArgs;
    auto castToDouble = [&](Value v) {
      if (v.getType().getIntOrFloatBitWidth() < 64)
        v = rewriter.create<arith::ExtFOp>(loc, rewriter.getF64Type(), v);
      return v;
    };
    Value val = instOp.getIsAdj()
                    ? rewriter.create<arith::NegFOp>(loc, instOperands[0])
                    : instOperands[0];
    funcArgs.push_back(castToDouble(val));

    // If no controls, then this is easy
    if (numControls == 0) {
      FlatSymbolRefAttr symbolRef =
          cudaq::opt::factory::createLLVMFunctionSymbol(
              qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
              {paramType, qubitIndexType}, parentModule);

      funcArgs.push_back(adaptor.getTargets().front());

      // Create the CallOp for this quantum instruction
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, ArrayRef<Type>{},
                                                symbolRef, funcArgs);
      return success();
    }

    qirFunctionName += "__ctl";

    // __quantum__qis__NAME__ctl(double, Array*, Qubit*) Type
    auto instOpQISFunctionType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context),
        {paramType, qubitArrayType, qubitIndexType});

    // Get function pointer to ctrl operation
    FlatSymbolRefAttr instSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
            {paramType, qubitArrayType, qubitIndexType}, parentModule);

    // We have >= 1 control, is the first a veq or a ref?
    auto control = *instOp.getControls().begin();
    Type type = control.getType();
    // If type is a VeqType, then we're good, just forward to the call op
    if (numControls == 1 && type.isa<quake::VeqType>()) {

      // Add the control array to the args.
      funcArgs.push_back(adaptor.getControls().front());

      // Add the target op
      funcArgs.push_back(adaptor.getTargets().front());

      // Here we already have and Array*, Qubit*
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{},
                                                instSymbolRef, funcArgs);
      return success();
    }

    // The remaining scenarios are best handled with the
    // invokeRotationWithControlQubits function.

    Value ctrlOpPointer = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(instOpQISFunctionType), instSymbolRef);

    // Get symbol for
    // void invokeRotationWithControlQubits(double param, const std::size_t
    // numControlOperands, i64* isArrayAndLength, void (*QISFunction)(Array*,
    // Qubit*), Qubit*, ...);
    auto i64Type = rewriter.getI64Type();
    auto applyMultiControlFunction =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::NVQIRInvokeRotationWithControlBits,
            LLVM::LLVMVoidType::get(context),
            {paramType, i64Type, LLVM::LLVMPointerType::get(i64Type),
             LLVM::LLVMPointerType::get(instOpQISFunctionType)},
            parentModule, true);

    // Create an integer array where the kth element is N if the kth
    // control operand is a veq<N>, and 0 otherwise.
    Value isArrayAndLengthArr = cudaq::opt::factory::packIsArrayAndLengthArray(
        loc, rewriter, parentModule, numControls, adaptor.getControls());

    funcArgs.push_back(
        cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, numControls));
    funcArgs.push_back(isArrayAndLengthArr);
    funcArgs.push_back(ctrlOpPointer);
    funcArgs.append(instOperands.begin(), instOperands.end());

    // Call our utility function.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        instOp, TypeRange{}, applyMultiControlFunction, funcArgs);

    return success();
  }
};

/// Lower single target Quantum ops with two parameters to QIR: u2
template <typename OP>
class OneTargetTwoParamRewrite : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = instOp->getName().stripDialect().str();
    auto numControls = instOp.getControls().size();
    auto loc = instOp->getLoc();
    ModuleOp parentModule = instOp->template getParentOfType<ModuleOp>();
    auto *context = instOp.getContext();
    auto qirFunctionName = std::string(cudaq::opt::QIRQISPrefix) + instName;

    SmallVector<Type> tmpArgTypes;
    auto qubitIndexType = cudaq::opt::getQubitType(context);

    auto paramType = FloatType::getF64(context);
    tmpArgTypes.push_back(paramType);
    tmpArgTypes.push_back(paramType);
    tmpArgTypes.push_back(qubitIndexType);

    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
        std::move(tmpArgTypes), parentModule);

    SmallVector<Value> funcArgs;
    auto castToDouble = [&](Value v) {
      if (v.getType().getIntOrFloatBitWidth() < 64)
        v = rewriter.create<arith::ExtFOp>(loc, rewriter.getF64Type(), v);
      return v;
    };
    Value v = adaptor.getOperands()[0];
    v = instOp.getIsAdj() ? rewriter.create<arith::NegFOp>(loc, v) : v;
    funcArgs.push_back(castToDouble(v));
    v = adaptor.getOperands()[1];
    v = instOp.getIsAdj() ? rewriter.create<arith::NegFOp>(loc, v) : v;
    funcArgs.push_back(castToDouble(v));

    // TODO: What about the control qubits?
    if (numControls != 0)
      return instOp.emitError("unsupported controlled op " + instName +
                              " with " + std::to_string(numControls) +
                              " ctrl qubits");

    funcArgs.push_back(adaptor.getOperands()[numControls + 2]);

    // Create the CallOp for this quantum instruction
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{}, symbolRef,
                                              funcArgs);
    return success();
  }
};

/// Lower single target Quantum ops with three parameters to QIR: u3
template <typename OP>
class OneTargetThreeParamRewrite : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = instOp->getName().stripDialect().str();
    auto numControls = instOp.getControls().size();
    auto instOperands = adaptor.getOperands();

    auto loc = instOp.getLoc();
    ModuleOp parentModule = instOp->template getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    std::string qirQisPrefix = cudaq::opt::QIRQISPrefix;
    auto qirFunctionName = qirQisPrefix + instName;

    auto qubitIndexType = cudaq::opt::getQubitType(context);
    auto qubitArrayType = cudaq::opt::getArrayType(context);
    auto paramType = FloatType::getF64(context);

    SmallVector<Value> funcArgs;
    auto castToDouble = [&](Value v) {
      if (v.getType().getIntOrFloatBitWidth() < 64)
        v = rewriter.create<arith::ExtFOp>(loc, rewriter.getF64Type(), v);
      return v;
    };
    // 3 parameters
    for (int i = 0; i < 3; i++) {
      Value val = instOp.getIsAdj()
                      ? rewriter.create<arith::NegFOp>(loc, instOperands[i])
                      : instOperands[i];
      funcArgs.push_back(castToDouble(val));
    }

    // If no controls, then this is simple
    if (numControls == 0) {
      FlatSymbolRefAttr symbolRef =
          cudaq::opt::factory::createLLVMFunctionSymbol(
              qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
              {paramType, paramType, paramType, qubitIndexType}, parentModule);

      funcArgs.push_back(adaptor.getTargets().front());

      // Create the CallOp for this quantum instruction
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, ArrayRef<Type>{},
                                                symbolRef, funcArgs);
      return success();
    }

    qirFunctionName += "__ctl";

    // __quantum__qis__u3__ctl(double, double, double, Array*, Qubit*) Type
    auto instOpQISFunctionType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context),
        {paramType, paramType, paramType, qubitArrayType, qubitIndexType});

    // Get function pointer to ctrl operation
    FlatSymbolRefAttr instSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
            {paramType, paramType, paramType, qubitArrayType, qubitIndexType},
            parentModule);

    // We have >= 1 control, is the first a veq or a ref?
    auto control = *instOp.getControls().begin();
    Type type = control.getType();
    // If type is a VeqType, then we're good, just forward to the call op
    if (numControls == 1 && type.isa<quake::VeqType>()) {

      // Add the control array to the args.
      funcArgs.push_back(adaptor.getControls().front());

      // Add the target op
      funcArgs.push_back(adaptor.getTargets().front());

      // Here we already have and Array*, Qubit*
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{},
                                                instSymbolRef, funcArgs);
      return success();
    }

    // The remaining scenarios are best handled with the
    // invokeU3RotationWithControlQubits function.

    Value ctrlOpPointer = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(instOpQISFunctionType), instSymbolRef);

    // Get symbol for void invokeU3RotationWithControlQubits(double theta,
    // double phi, double lambda, const std::size_t numControlOperands, i64*
    // isArrayAndLength, void (*QISFunction)(double, double, double, Array*,
    // Qubit*), Qubit*, ...);
    auto i64Type = rewriter.getI64Type();
    auto applyMultiControlFunction =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::NVQIRInvokeU3RotationWithControlBits,
            LLVM::LLVMVoidType::get(context),
            {paramType, paramType, paramType, i64Type,
             LLVM::LLVMPointerType::get(i64Type),
             LLVM::LLVMPointerType::get(instOpQISFunctionType)},
            parentModule, true);

    // Create an integer array where the kth element is N if the kth
    // control operand is a veq<N>, and 0 otherwise.
    Value isArrayAndLengthArr = cudaq::opt::factory::packIsArrayAndLengthArray(
        loc, rewriter, parentModule, numControls, adaptor.getControls());

    funcArgs.push_back(
        cudaq::opt::factory::genLlvmI64Constant(loc, rewriter, numControls));
    funcArgs.push_back(isArrayAndLengthArr);
    funcArgs.push_back(ctrlOpPointer);
    funcArgs.append(instOperands.begin(), instOperands.end());

    // Call our utility function.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        instOp, TypeRange{}, applyMultiControlFunction, funcArgs);

    return success();
  }
};

/// Lower two-target Quantum ops with no parameter to QIR:
/// swap
template <typename OP>
class TwoTargetRewrite : public ConvertOpWithControls<OP> {
public:
  using Base = ConvertOpWithControls<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto numControls = instOp.getControls().size();

    if (numControls != 0)
      return Base::matchAndRewriteWithControls(instOp, adaptor, rewriter);

    auto instName = instOp->getName().stripDialect().str();
    ModuleOp parentModule = instOp->template getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto qirFunctionName = std::string(cudaq::opt::QIRQISPrefix) + instName;

    auto qubitIndexType = cudaq::opt::getQubitType(context);
    SmallVector<Type> tmpArgTypes = {qubitIndexType, qubitIndexType};

    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
        std::move(tmpArgTypes), parentModule);

    // FIXME: Do we want to use any fast math flags?
    // TODO: Looks like bugs in the MLIR cmake files don't install this any
    // longer.
    // auto fastMathFlags = LLVM::FMFAttr::get(context, {});
    // Create the CallOp for this quantum instruction
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        instOp, TypeRange{}, symbolRef,
        adaptor.getOperands() /*, fastMathFlags*/);
    return success();
  }
};

/// Lowers Quake MeasureOp to respective QIR function.
/// mx, my, mz
template <typename OP>
class MeasureRewrite : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

  unsigned &measureCounter;

  MeasureRewrite(LLVMTypeConverter &typeConverter, unsigned &c)
      : Base(typeConverter), measureCounter(c) {}

  LogicalResult
  matchAndRewrite(OP measure, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = measure->getLoc();
    auto parentModule = measure->template getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    std::string qFunctionName = cudaq::opt::QIRMeasure;
    Attribute regName = measure.getRegisterNameAttr();
    std::vector<Type> funcTypes{cudaq::opt::getQubitType(context)};
    std::vector<Value> args{adaptor.getOperands().front()};

    bool appendName;
    if (regName && !regName.cast<StringAttr>().getValue().empty()) {
      // Change the function name
      qFunctionName += "__to__register";
      // Append a string type argument
      funcTypes.push_back(LLVM::LLVMPointerType::get(rewriter.getI8Type()));
      appendName = true;
    } else {
      // If no register name is supplied, make one up. Zero pad the counter so
      // that sequential measurements contain alphabetically sorted register
      // names.
      char regNameCounter[16]; // sized big enough for "r" + 5 digits
      if (measureCounter > 99999) {
        emitError(loc,
                  "Too many unnamed measurements. Name your measurements by "
                  "saving them to variables, like `auto result = mz(q)`");
        return failure();
      }
      std::snprintf(regNameCounter, sizeof(regNameCounter), "r%05d",
                    measureCounter++);
      regName = rewriter.getStringAttr(regNameCounter);
      appendName = false;
    }
    // Get the name
    auto regNameAttr = regName.cast<StringAttr>();
    auto regNameStr = regNameAttr.getValue().str();
    std::string regNameGlobalStr = regNameStr;

    // Write to the module body
    auto insertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(parentModule.getBody());

    // Create the register name global
    auto builder = cudaq::IRBuilder::atBlockEnd(parentModule.getBody());
    auto regNameGlobal =
        builder.genCStringLiteralAppendNul(loc, parentModule, regNameStr);

    // Shift back to the function
    rewriter.restoreInsertionPoint(insertPoint);

    // Get the string address and bit cast
    auto regNameRef = rewriter.create<LLVM::AddressOfOp>(
        loc, cudaq::opt::factory::getPointerType(regNameGlobal.getType()),
        regNameGlobal.getSymName());
    auto castedRegNameRef = rewriter.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(context), regNameRef);

    // Append to the args list
    if (appendName)
      args.push_back(castedRegNameRef);

    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qFunctionName, cudaq::opt::getResultType(context),
        llvm::ArrayRef(funcTypes), parentModule);

    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, cudaq::opt::getResultType(context), symbolRef, ValueRange{args});
    if (regName)
      callOp->setAttr("registerName", regName);
    auto i1Ty = rewriter.getI1Type();
    auto i1PtrTy = LLVM::LLVMPointerType::get(i1Ty);
    auto cast =
        rewriter.create<LLVM::BitcastOp>(loc, i1PtrTy, callOp.getResult());
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(measure, i1Ty, cast);

    return success();
  }
};

/// Convert a MX operation to a sequence H; MZ.
class MxToMz : public OpConversionPattern<quake::MxOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::MxOp mx, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<quake::HOp>(mx.getLoc(), adaptor.getTargets());
    rewriter.replaceOpWithNewOp<quake::MzOp>(mx, mx.getResultTypes(),
                                             adaptor.getTargets(),
                                             mx.getRegisterNameAttr());
    return success();
  }
};

/// Convert a MY operation to a sequence S; H; MZ.
class MyToMz : public OpConversionPattern<quake::MyOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::MyOp my, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.create<quake::SOp>(my.getLoc(), true, ValueRange{}, ValueRange{},
                                adaptor.getTargets());
    rewriter.create<quake::HOp>(my.getLoc(), adaptor.getTargets());
    rewriter.replaceOpWithNewOp<quake::MzOp>(my, my.getResultTypes(),
                                             adaptor.getTargets(),
                                             my.getRegisterNameAttr());
    return success();
  }
};

class GetVeqSizeOpRewrite : public OpConversionPattern<quake::VeqSizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::VeqSizeOp vecsize, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = vecsize->getLoc();
    auto parentModule = vecsize->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto qFunctionName = cudaq::opt::QIRArrayGetSize;

    auto symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qFunctionName, rewriter.getI64Type(),
        {cudaq::opt::getArrayType(context)}, parentModule);

    auto c = rewriter.create<LLVM::CallOp>(loc, rewriter.getI64Type(),
                                           symbolRef, adaptor.getOperands());
    vecsize->getResult(0).replaceAllUsesWith(c->getResult(0));
    rewriter.eraseOp(vecsize);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Other conversion patterns.
//===----------------------------------------------------------------------===//

/// Converts returning a Result* to returning a bit. QIR expects
/// __quantum__qis__mz(Qubit*) to return a Result*, and CUDA-Q expects
/// mz to return a bool. In the library we let Result = bool, so Result* is
/// a bool*. Here we bitcast the Result* to a bool* and then load it and
/// replace its use with that loaded bool.
class ReturnBitRewrite : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp ret, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = ret->getLoc();
    auto parentModule = ret->getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();

    // If we are returning a llvm.ptr<Result> then we've really
    // been asked to return a bit, set that up here
    if (ret.getNumOperands() == 1 && adaptor.getOperands().front().getType() ==
                                         cudaq::opt::getResultType(context)) {

      // Bitcast the produced value, which corresponds to the value in
      // ret.operands()[0], from llvm.ptr<Result> to llvm.ptr<i1>. There is a
      // big assumption here, which is that the operation that produced the
      // llvm.ptr<Result> type returns only one value. Given that this should
      // be a call to __quantum__qis__mz(Qubit*) and that in the LLVM dialect,
      // functions always have a single result, this should be fine. If things
      // change, we will need to update this.
      auto bitcast = rewriter.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getI1Type()),
          adaptor.getOperands().front());

      // Load the bool
      auto loadBit = rewriter.create<LLVM::LoadOp>(loc, rewriter.getI1Type(),
                                                   bitcast.getResult());

      // Replace all uses of the llvm.ptr<Result> with the i1, which includes
      // the return op. Do not replace its use in the bitcast.
      adaptor.getOperands().front().replaceAllUsesExcept(loadBit.getResult(),
                                                         bitcast);
      return success();
    }
    return failure();
  }
};

/// In case we still have a RelaxSizeOp, we can just remove it, since QIR works
/// on `Array*` for all sized veqs.
class RemoveRelaxSizeRewrite : public OpConversionPattern<quake::RelaxSizeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(quake::RelaxSizeOp relax, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(relax, relax.getInputVec());
    return success();
  }
};

/// NOTE: This class currently targets simulator backends. On hardware targets
/// the custom operations ought to be decomposed by a separate pass and should
/// never reach here.
class CustomUnitaryOpRewrite
    : public ConvertOpToLLVMPattern<quake::CustomUnitarySymbolOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  /// Function to convert a QIR Qubit value to an Array value.
  /// TODO: Refactor to reuse code from 'ConcatOpRewrite' (line#247)
  Value wrapQubitInArray(Location &loc, ConversionPatternRewriter &rewriter,
                         ModuleOp parentModule, Value v) const {
    auto context = rewriter.getContext();
    auto qirArrayTy = cudaq::opt::getArrayType(context);
    auto ptrTy = cudaq::opt::factory::getPointerType(context);
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRArrayCreateArray, qirArrayTy,
        {rewriter.getI32Type(), rewriter.getI64Type()}, parentModule);
    FlatSymbolRefAttr getSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::QIRArrayGetElementPtr1d, ptrTy,
            {qirArrayTy, rewriter.getIntegerType(64)}, parentModule);
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 64);
    // FIXME: 8 bytes is assumed to be the sizeof(char*) on the target machine.
    Value eight = rewriter.create<arith::ConstantIntOp>(loc, 8, 32);
    if (v.getType() != cudaq::opt::getQubitType(context))
      return v;
    auto createCall = rewriter.create<LLVM::CallOp>(
        loc, qirArrayTy, symbolRef, ArrayRef<Value>{eight, one});
    auto result = createCall.getResult();
    auto call = rewriter.create<LLVM::CallOp>(loc, ptrTy, getSymbolRef,
                                              ArrayRef<Value>{result, zero});
    Value pointer = rewriter.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(ptrTy), call.getResult());
    auto cast = rewriter.create<LLVM::BitcastOp>(loc, ptrTy, v);
    rewriter.create<LLVM::StoreOp>(loc, cast, pointer);
    return result;
  }

  LogicalResult
  matchAndRewrite(quake::CustomUnitarySymbolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto parentModule = op->getParentOfType<ModuleOp>();
    auto context = op->getContext();
    auto typeConverter = this->getTypeConverter();

    auto numParameters = op.getParameters().size();
    if (numParameters)
      op.emitOpError("Parameterized custom operations not yet supported.");

    auto arrType = cudaq::opt::getArrayType(context);
    auto qirArrayTy = cudaq::opt::getArrayType(context);
    FlatSymbolRefAttr concatFunc =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::QIRArrayConcatArray, arrType, {arrType, arrType},
            parentModule);

    // targets
    auto targetArr = wrapQubitInArray(loc, rewriter, parentModule,
                                      adaptor.getTargets().front());
    for (auto oper : adaptor.getTargets().drop_front(1)) {
      auto backArr = wrapQubitInArray(loc, rewriter, parentModule, oper);
      auto glue = rewriter.create<LLVM::CallOp>(
          loc, qirArrayTy, concatFunc, ArrayRef<Value>{targetArr, backArr});
      targetArr = glue.getResult();
    }

    // controls
    auto controls = op.getControls();
    Value controlArr;
    if (controls.empty()) {
      // make an empty array
      Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      Value zero32 = rewriter.create<arith::ConstantIntOp>(loc, 8, 32);
      FlatSymbolRefAttr symbolRef =
          cudaq::opt::factory::createLLVMFunctionSymbol(
              cudaq::opt::QIRArrayCreateArray,
              cudaq::opt::getArrayType(context),
              {rewriter.getI32Type(), rewriter.getI64Type()}, parentModule);
      controlArr = rewriter
                       .create<LLVM::CallOp>(
                           loc, TypeRange{cudaq::opt::getArrayType(context)},
                           symbolRef, ValueRange{zero32, zero})
                       .getResult();
    } else {
      controlArr = wrapQubitInArray(loc, rewriter, parentModule,
                                    adaptor.getControls().front());
      for (auto oper : adaptor.getControls().drop_front(1)) {
        auto backArr = wrapQubitInArray(loc, rewriter, parentModule, oper);
        auto glue = rewriter.create<LLVM::CallOp>(
            loc, qirArrayTy, concatFunc, ArrayRef<Value>{controlArr, backArr});
        controlArr = glue.getResult();
      }
    }

    // Fetch the unitary matrix generator for this custom operation
    auto sref = op.getGenerator();
    StringRef generatorName = sref.getRootReference();
    auto globalOp =
        parentModule.lookupSymbol<cudaq::cc::GlobalOp>(generatorName);
    if (!globalOp)
      return op.emitOpError("global not found for custom op");

    auto complex64Ty =
        typeConverter->convertType(ComplexType::get(rewriter.getF64Type()));
    auto complex64PtrTy = LLVM::LLVMPointerType::get(complex64Ty);
    Type type = getTypeConverter()->convertType(globalOp.getType());
    auto addrOp = rewriter.create<LLVM::AddressOfOp>(loc, type, generatorName);
    auto unitaryData =
        rewriter.create<LLVM::BitcastOp>(loc, complex64PtrTy, addrOp);

    auto qirFunctionName =
        std::string{cudaq::opt::QIRCustomOp} + (op.isAdj() ? "__adj" : "");

    FlatSymbolRefAttr customSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            qirFunctionName, LLVM::LLVMVoidType::get(context),
            {complex64PtrTy, cudaq::opt::getArrayType(context),
             cudaq::opt::getArrayType(context)},
            parentModule);

    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, TypeRange{}, customSymbolRef,
        ValueRange{unitaryData, controlArr, targetArr});

    return success();
  }
};
} // namespace

void cudaq::opt::populateQuakeToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                             RewritePatternSet &patterns,
                                             unsigned &measureCounter) {
  auto *context = patterns.getContext();
  patterns.insert<GetVeqSizeOpRewrite, RemoveRelaxSizeRewrite, MxToMz, MyToMz,
                  ReturnBitRewrite>(context);
  patterns.insert<
      AllocaOpRewrite, ConcatOpRewrite, CustomUnitaryOpRewrite,
      DeallocOpRewrite, DiscriminateOpPattern, ExtractQubitOpRewrite,
      ExpPauliRewrite, OneTargetRewrite<quake::HOp>,
      OneTargetRewrite<quake::XOp>, OneTargetRewrite<quake::YOp>,
      OneTargetRewrite<quake::ZOp>, OneTargetRewrite<quake::SOp>,
      OneTargetRewrite<quake::TOp>, OneTargetOneParamRewrite<quake::R1Op>,
      OneTargetTwoParamRewrite<quake::PhasedRxOp>,
      OneTargetOneParamRewrite<quake::RxOp>,
      OneTargetOneParamRewrite<quake::RyOp>,
      OneTargetOneParamRewrite<quake::RzOp>,
      OneTargetTwoParamRewrite<quake::U2Op>,
      OneTargetThreeParamRewrite<quake::U3Op>, QmemRAIIOpRewrite, ResetRewrite,
      SubveqOpRewrite, TwoTargetRewrite<quake::SwapOp>>(typeConverter);
  patterns.insert<MeasureRewrite<quake::MzOp>>(typeConverter, measureCounter);
}
