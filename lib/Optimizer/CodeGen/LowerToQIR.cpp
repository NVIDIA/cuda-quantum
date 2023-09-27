/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#pragma GCC diagnostic ignored "-Wrestrict"
#endif
#include "PassDetails.h"
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif

#include "cudaq/Optimizer/Builder/CUDAQBuilder.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Peephole.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

static LLVM::LLVMStructType lambdaAsPairOfPointers(MLIRContext *context) {
  auto ptrTy =
      cudaq::opt::factory::getPointerType(IntegerType::get(context, 8));
  SmallVector<Type> pairOfPointers = {ptrTy, ptrTy};
  return LLVM::LLVMStructType::getLiteral(context, pairOfPointers);
}

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
            alloca.getResult().getType().dyn_cast_or_null<quake::RefType>()) {

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

    rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(
        dealloc, ArrayRef<Type>({}), deallocSymbolRef,
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
    auto i8PtrTy =
        cudaq::opt::factory::getPointerType(IntegerType::get(context, 8));
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        cudaq::opt::QIRArrayCreateArray, qirArrayTy,
        {rewriter.getIntegerType(32), rewriter.getIntegerType(64)},
        parentModule);
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
        LLVM::LLVMPointerType::get(rewriter.getIntegerType(8));

    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qir_array_get_element_ptr_1d, qbit_element_ptr_type,
        {array_qbit_type, rewriter.getIntegerType(64)}, parentModule);

    Value idx_operand;
    auto i64Ty = rewriter.getI64Type();
    if (extract.hasConstantIndex()) {
      idx_operand = rewriter.create<arith::ConstantIntOp>(
          loc, extract.getConstantIndex(), i64Ty);
    } else {
      idx_operand = adaptor.getOperands()[1];

      if (idx_operand.getType().isIntOrFloat() &&
          idx_operand.getType().cast<IntegerType>().getWidth() < 64) {
        idx_operand = rewriter.create<LLVM::ZExtOp>(loc, i64Ty, idx_operand);
      }
      if (idx_operand.getType().isa<IndexType>()) {
        idx_operand =
            rewriter.create<arith::IndexCastOp>(loc, i64Ty, idx_operand)
                .getResult();
      }
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

    auto i32Ty = rewriter.getIntegerType(32);
    auto i64Ty = rewriter.getIntegerType(64);
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        rtSubveqFuncName, arrayTy, {arrayTy, i32Ty, i64Ty, i64Ty, i64Ty},
        parentModule);

    Value lowArg = adaptor.getOperands()[1];
    Value highArg = adaptor.getOperands()[2];
    auto extend = [&](Value &v) -> Value {
      if (v.getType().isa<IntegerType>() &&
          v.getType().cast<IntegerType>().getWidth() < 64)
        return rewriter.create<LLVM::ZExtOp>(loc, i64Ty, v);
      if (v.getType().isa<IndexType>())
        return rewriter.create<arith::IndexCastOp>(loc, i64Ty, v);
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
    std::string qirQisPrefix(cudaq::opt::QIRQISPrefix);
    std::string instName = instOp->getName().stripDialect().str();

    // Get the reset QIR function name
    auto qirFunctionName = qirQisPrefix + instName;

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
    std::string qirQisPrefix(cudaq::opt::QIRQISPrefix);
    auto qirFunctionName = qirQisPrefix + "exp_pauli";
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
        {rewriter.getF64Type(), cudaq::opt::getArrayType(context),
         cudaq::opt::factory::getPointerType(context)},
        parentModule);
    SmallVector<Value> operands = adaptor.getOperands();
    // Make sure to drop any length information from the type of the Pauli word.
    auto pauliWord = operands.back();
    operands.pop_back();
    auto castedPauli = rewriter.create<LLVM::BitcastOp>(
        loc, cudaq::opt::factory::getPointerType(context), pauliWord);
    operands.push_back(castedPauli);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{}, symbolRef,
                                              operands);
    return success();
  }
};

/// Lower single target Quantum ops with no parameter to QIR:
/// h, x, y, z, s, t
template <typename OP>
class OneTargetRewrite : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto numControls = instOp.getControls().size();
    auto loc = instOp->getLoc();
    auto parentModule = instOp->template getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    std::string qirQisPrefix(cudaq::opt::QIRQISPrefix);
    std::string instName = instOp->getName().stripDialect().str();

    if (numControls == 0) {
      // There are no control bits, so call the function directly.
      auto qirFunctionName =
          qirQisPrefix + instName + (instOp.getIsAdj() ? "__adj" : "");
      FlatSymbolRefAttr symbolRef =
          cudaq::opt::factory::createLLVMFunctionSymbol(
              qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
              {cudaq::opt::getQubitType(context)}, parentModule);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{}, symbolRef,
                                                adaptor.getOperands());
      return success();
    }

    // Convert the ctrl bits to an Array
    auto qirFunctionName = qirQisPrefix + instName + "__ctl";
    auto negateFunctionName = qirQisPrefix + "x";
    auto negatedQubitCtrls = instOp.getNegatedQubitControls();

    // Useful types we'll need
    auto qirArrayType = cudaq::opt::getArrayType(context);
    auto qirQubitPointerType = cudaq::opt::getQubitType(context);
    auto i64Type = rewriter.getI64Type();

    // __quantum__qis__NAME__ctl(Array*, Qubit*) Type
    auto instOpQISFunctionType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(context), {qirArrayType, qirQubitPointerType});

    // Get the function pointer for the ctrl operation
    auto qirFunctionSymbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        qirFunctionName, LLVM::LLVMVoidType::get(context),
        {qirArrayType, qirQubitPointerType}, parentModule);

    // Get the first control's type
    auto control = *instOp.getControls().begin();
    Type type = control.getType();
    auto instOperands = adaptor.getOperands();
    if (numControls == 1 && type.isa<quake::VeqType>()) {
      if (negatedQubitCtrls)
        return instOp.emitError("unsupported controlled op " + instName +
                                " with vector of ctrl qubits");
      // Operands are already an Array* and Qubit*.
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(
          instOp, TypeRange{}, qirFunctionSymbolRef, instOperands);
      return success();
    }

    // Get symbol for
    // void invokeWithControlQubits(const std::size_t nControls, void
    // (*QISFunction)(Array*, Qubit*), Qubit*, ...);
    auto applyMultiControlFunction =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::NVQIRInvokeWithControlBits,
            LLVM::LLVMVoidType::get(context),
            {i64Type, LLVM::LLVMPointerType::get(instOpQISFunctionType)},
            parentModule, true);

    Value ctrlOpPointer = rewriter.create<LLVM::AddressOfOp>(
        loc, LLVM::LLVMPointerType::get(instOpQISFunctionType),
        qirFunctionSymbolRef);
    auto arraySize =
        rewriter.create<LLVM::ConstantOp>(loc, i64Type, numControls);
    // This will need the numControls, function pointer, and all Qubit*
    // operands
    SmallVector<Value> args = {arraySize, ctrlOpPointer};
    FlatSymbolRefAttr negateFuncRef;
    if (negatedQubitCtrls) {
      negateFuncRef = cudaq::opt::factory::createLLVMFunctionSymbol(
          negateFunctionName,
          /*return type=*/LLVM::LLVMVoidType::get(context),
          {cudaq::opt::getQubitType(context)}, parentModule);
      for (auto v : llvm::enumerate(instOperands)) {
        if ((v.index() < numControls) && (*negatedQubitCtrls)[v.index()])
          rewriter.create<LLVM::CallOp>(loc, TypeRange{}, negateFuncRef,
                                        v.value());
        args.push_back(v.value());
      }
    } else {
      args.append(instOperands.begin(), instOperands.end());
    }

    // Call our utility function.
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{},
                                              applyMultiControlFunction, args);

    if (negatedQubitCtrls)
      for (auto v : llvm::enumerate(instOperands))
        if ((v.index() < numControls) && (*negatedQubitCtrls)[v.index()])
          rewriter.create<LLVM::CallOp>(loc, TypeRange{}, negateFuncRef,
                                        v.value());

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

    // TODO: handle general control ops. For now, only allow Rn with 1
    // control
    if (numControls > 1)
      return instOp.emitError("unsupported controlled op " + instName +
                              " with " + std::to_string(numControls) +
                              " ctrl qubits");

    auto loc = instOp.getLoc();
    ModuleOp parentModule = instOp->template getParentOfType<ModuleOp>();
    auto context = parentModule->getContext();
    auto qirFunctionName = std::string(cudaq::opt::QIRQISPrefix) + instName;

    SmallVector<Type> tmpArgTypes;
    auto qubitIndexType = cudaq::opt::getQubitType(context);
    auto qubitArrayType = cudaq::opt::getArrayType(context);
    auto paramType = FloatType::getF64(context);
    tmpArgTypes.push_back(paramType);

    SmallVector<Value> funcArgs;
    auto castToDouble = [&](Value v) {
      if (v.getType().getIntOrFloatBitWidth() < 64)
        v = rewriter.create<arith::ExtFOp>(loc, rewriter.getF64Type(), v);
      return v;
    };
    Value v =
        instOp.getIsAdj()
            ? rewriter.create<arith::NegFOp>(loc, adaptor.getOperands()[0])
            : adaptor.getOperands()[0];
    funcArgs.push_back(castToDouble(v));

    // If no controls, then this is easy
    if (numControls == 0) {
      tmpArgTypes.push_back(qubitIndexType);
      FlatSymbolRefAttr symbolRef =
          cudaq::opt::factory::createLLVMFunctionSymbol(
              qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
              std::move(tmpArgTypes), parentModule);

      funcArgs.push_back(adaptor.getTargets().front());

      // Create the CallOp for this quantum instruction
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, ArrayRef<Type>{},
                                                symbolRef, funcArgs);
      return success();
    }

    // We have 1 control, is it a veq or a ref?
    auto control = *instOp.getControls().begin();
    qirFunctionName += "__ctl";

    // All signatures will take an Array* for the controls
    tmpArgTypes.push_back(qubitArrayType);

    // If type is a VeqType, then we're good, just forward to the call op
    Type type = control.getType();
    if (type.isa<quake::VeqType>()) {

      // Add the control array to the args.
      funcArgs.push_back(adaptor.getControls().front());

      // This is a single target op, add that type
      tmpArgTypes.push_back(qubitIndexType);

      FlatSymbolRefAttr instSymbolRef =
          cudaq::opt::factory::createLLVMFunctionSymbol(
              qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
              std::move(tmpArgTypes), parentModule);

      // Add the target op
      funcArgs.push_back(adaptor.getTargets().front());

      // Here we already have and Array*, Qubit*
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{},
                                                instSymbolRef, funcArgs);
      return success();
    }

    // If the control is a qubit, we need to pack it into an Array*
    FlatSymbolRefAttr packingSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::NVQIRPackSingleQubitInArray,
            /*return type=*/qubitArrayType, {qubitIndexType}, parentModule);
    // Pack the qubit into the array
    Value result =
        rewriter
            .create<LLVM::CallOp>(loc, qubitArrayType, packingSymbolRef,
                                  adaptor.getControls().front())
            .getResult();
    // The array result is what we want for the function args
    funcArgs.push_back(result);

    // This is a single target op, add that type
    tmpArgTypes.push_back(qubitIndexType);

    FlatSymbolRefAttr instSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            qirFunctionName, /*return type=*/LLVM::LLVMVoidType::get(context),
            std::move(tmpArgTypes), parentModule);

    // Add the target op
    funcArgs.push_back(adaptor.getTargets().front());

    // Here we already have and Array*, Qubit*
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(instOp, TypeRange{},
                                              instSymbolRef, funcArgs);

    // We need to release the control Array.
    FlatSymbolRefAttr releaseSymbolRef =
        cudaq::opt::factory::createLLVMFunctionSymbol(
            cudaq::opt::NVQIRReleasePackedQubitArray,
            /*return type=*/LLVM::LLVMVoidType::get(context), {qubitArrayType},
            parentModule);
    Value ctrlArray = funcArgs[1];
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, releaseSymbolRef,
                                  ctrlArray);

    return success();
  }
};

/// Lower single target Quantum ops with two parameters to QIR:
/// u2, u3
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

/// Lower two-target Quantum ops with no parameter to QIR:
/// swap
template <typename OP>
class TwoTargetRewrite : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto instName = instOp->getName().stripDialect().str();
    auto numControls = instOp.getControls().size();

    // TODO: handle general control ops.
    if (numControls != 0)
      return instOp.emitError("unsupported controlled op " + instName +
                              " with " + std::to_string(numControls) +
                              " ctrl qubits");

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
    Attribute regName = measure->getAttr("registerName");
    std::vector<Type> funcTypes{cudaq::opt::getQubitType(context)};
    std::vector<Value> args{adaptor.getOperands().front()};

    bool appendName;
    if (regName && !regName.cast<StringAttr>().getValue().empty()) {
      // Change the function name
      qFunctionName += "__to__register";
      // Append a string type argument
      funcTypes.push_back(
          LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));
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
    auto i1Ty = rewriter.getIntegerType(1);
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
// Conversion patterns for CC dialect ops.
//===----------------------------------------------------------------------===//

class AllocaOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::AllocaOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::AllocaOp>;
  using Base::Base;

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
          ArrayRef<Value>{cudaq::opt::factory::genI32Constant(alloc.getLoc(),
                                                              rewriter, 1)});
    } else {
      rewriter.replaceOpWithNewOp<LLVM::AllocaOp>(alloc, toTy, operands);
    }
    return success();
  }
};

class CallableClosureOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::CallableClosureOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::CallableClosureOp>;
  using Base::Base;

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
    assert(structTy);
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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::CallableFuncOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(cudaq::cc::CallableFuncOp callable, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = callable.getLoc();
    auto operands = adaptor.getOperands();
    auto resTy = getTypeConverter()->convertType(callable.getType());
    auto structTy = dyn_cast<LLVM::LLVMStructType>(operands[0].getType());
    assert(structTy);
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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::CallCallableOp>;
  using Base::Base;

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
    auto zeroAttr = rewriter.getI64IntegerAttr(0);
    auto i64Ty = rewriter.getI64Type();
    auto zeroI64 = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, zeroAttr);
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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::CastOp>;
  using Base::Base;

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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::ComputePtrOp>;
  using Base::Base;

  // Convert each cc::ComputePtrOp to an LLVM::GEPOp.
  LogicalResult
  matchAndRewrite(cudaq::cc::ComputePtrOp cp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto toTy = getTypeConverter()->convertType(cp.getType());
    Value base = operands[0];
    if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(base.getType()))
      if (auto arrTy = dyn_cast<LLVM::LLVMArrayType>(ptrTy.getElementType())) {
        // Eliminate intermediate array type. Not needed in LLVM. (NB: for some
        // element types, the executable will crash.)
        auto ty = cudaq::opt::factory::getPointerType(arrTy.getElementType());
        base = rewriter.create<LLVM::BitcastOp>(cp.getLoc(), ty, base);
      }
    auto gepOpnds = interleaveConstantsAndOperands(operands.drop_front(),
                                                   cp.getRawConstantIndices());
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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::ExtractValueOp>;
  using Base::Base;

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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::FuncToPtrOp>;
  using Base::Base;

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

class InsertValueOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::InsertValueOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::InsertValueOp>;
  using Base::Base;

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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::InstantiateCallableOp>;
  using Base::Base;

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
    auto i64Ty = rewriter.getI64Type();
    auto tupleArgTy = lambdaAsPairOfPointers(ctx);
    if (callable.getNoCapture()) {
      auto zeroAttr = rewriter.getI64IntegerAttr(0);
      auto zero = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, zeroAttr);
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
      auto oneAttr = rewriter.getI64IntegerAttr(1);
      Value one = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, oneAttr);
      auto tuplePtrTy = cudaq::opt::factory::getPointerType(tupleTy);
      tmp = rewriter.create<LLVM::AllocaOp>(loc, tuplePtrTy, one);
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
    auto oneA = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{1});
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(callable, tupleArgTy,
                                                     tupleArg, castTmp, oneA);
    return success();
  }
};

class LoadOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::LoadOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::LoadOp>;
  using Base::Base;

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

class StdvecDataOpPattern
    : public ConvertOpToLLVMPattern<cudaq::cc::StdvecDataOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::StdvecDataOp>;
  using Base::Base;

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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::StdvecInitOp>;
  using Base::Base;

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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::StdvecSizeOp>;
  using Base::Base;

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
  using Base = ConvertOpToLLVMPattern<cudaq::cc::StoreOp>;
  using Base::Base;

  // Convert each cc::StoreOp to an LLVM::StoreOp.
  LogicalResult
  matchAndRewrite(cudaq::cc::StoreOp store, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(store, operands[0], operands[1]);
    return success();
  }
};

class UndefOpPattern : public ConvertOpToLLVMPattern<cudaq::cc::UndefOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::UndefOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(cudaq::cc::UndefOp undef, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resTy = getTypeConverter()->convertType(undef.getType());
    rewriter.replaceOpWithNewOp<LLVM::UndefOp>(undef, resTy);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Other conversion patterns.
//===----------------------------------------------------------------------===//

/// Converts returning a Result* to returning a bit. QIR expects
/// __quantum__qis__mz(Qubit*) to return a Result*, and CUDA Quantum expects
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

//===----------------------------------------------------------------------===//
// Code generation: converts the Quake IR to QIR.
//===----------------------------------------------------------------------===//

/// Convert Quake dialect to LLVM-IR and QIR.
class QuakeToQIRRewrite : public cudaq::opt::QuakeToQIRBase<QuakeToQIRRewrite> {
public:
  QuakeToQIRRewrite() = default;

  /// Measurement counter for unnamed measurements. Resets every module.
  unsigned measureCounter = 0;

  ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto *context = &getContext();
    LLVMConversionTarget target{*context};
    LLVMTypeConverter typeConverter(&getContext());
    initializeTypeConversions(typeConverter);
    RewritePatternSet patterns(context);

    populateAffineToStdConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);

    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    patterns.insert<GetVeqSizeOpRewrite, MxToMz, MyToMz, ReturnBitRewrite>(
        context);
    patterns
        .insert<AllocaOpRewrite, AllocaOpPattern, CallableClosureOpPattern,
                CallableFuncOpPattern, CallCallableOpPattern, CastOpPattern,
                ComputePtrOpPattern, ConcatOpRewrite, DeallocOpRewrite,
                CreateStringLiteralOpPattern, ExtractQubitOpRewrite,
                ExtractValueOpPattern, FuncToPtrOpPattern, InsertValueOpPattern,
                InstantiateCallableOpPattern, LoadOpPattern, ExpPauliRewrite,
                OneTargetRewrite<quake::HOp>, OneTargetRewrite<quake::XOp>,
                OneTargetRewrite<quake::YOp>, OneTargetRewrite<quake::ZOp>,
                OneTargetRewrite<quake::SOp>, OneTargetRewrite<quake::TOp>,
                OneTargetOneParamRewrite<quake::R1Op>,
                OneTargetTwoParamRewrite<quake::PhasedRxOp>,
                OneTargetOneParamRewrite<quake::RxOp>,
                OneTargetOneParamRewrite<quake::RyOp>,
                OneTargetOneParamRewrite<quake::RzOp>,
                OneTargetTwoParamRewrite<quake::U2Op>,
                OneTargetTwoParamRewrite<quake::U3Op>, ResetRewrite,
                StdvecDataOpPattern, StdvecInitOpPattern, StdvecSizeOpPattern,
                StoreOpPattern, SubveqOpRewrite,
                TwoTargetRewrite<quake::SwapOp>, UndefOpPattern>(typeConverter);
    patterns.insert<MeasureRewrite<quake::MzOp>>(typeConverter, measureCounter);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    if (failed(applyFullConversion(getModule(), target, std::move(patterns))))
      signalPassFailure();
  }

  void initializeTypeConversions(LLVMTypeConverter &typeConverter) {
    typeConverter.addConversion([](quake::VeqType type) {
      return cudaq::opt::getArrayType(type.getContext());
    });
    typeConverter.addConversion([](quake::RefType type) {
      return cudaq::opt::getQubitType(type.getContext());
    });
    typeConverter.addConversion([](cudaq::cc::CallableType type) {
      return lambdaAsPairOfPointers(type.getContext());
    });
    typeConverter.addConversion([](cudaq::cc::StdvecType type) {
      return cudaq::opt::factory::stdVectorImplType(type.getElementType());
    });
    typeConverter.addConversion([&typeConverter](cudaq::cc::PointerType type) {
      auto eleTy = type.getElementType();
      if (isa<NoneType>(eleTy))
        return cudaq::opt::factory::getPointerType(type.getContext());
      eleTy = typeConverter.convertType(eleTy);
      if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(eleTy)) {
        assert(arrTy.isUnknownSize());
        return cudaq::opt::factory::getPointerType(
            typeConverter.convertType(arrTy.getElementType()));
      }
      return cudaq::opt::factory::getPointerType(eleTy);
    });
    typeConverter.addConversion(
        [&typeConverter](cudaq::cc::ArrayType type) -> Type {
          auto eleTy = typeConverter.convertType(type.getElementType());
          if (type.isUnknownSize())
            return type;
          return LLVM::LLVMArrayType::get(eleTy, type.getSize());
        });
    typeConverter.addConversion(
        [&typeConverter](cudaq::cc::StructType type) -> Type {
          SmallVector<Type> members;
          for (auto t : type.getMembers())
            members.push_back(typeConverter.convertType(t));
          return LLVM::LLVMStructType::getLiteral(type.getContext(), members);
        });
  }
};

} // namespace

std::unique_ptr<Pass> cudaq::opt::createConvertToQIRPass() {
  return std::make_unique<QuakeToQIRRewrite>();
}
