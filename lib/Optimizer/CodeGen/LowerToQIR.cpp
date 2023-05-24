/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/CUDAQBuilder.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Peephole.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
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

/// Lowers Quake AllocaOp to QIR function call in LLVM.
class AllocaOpLowering : public ConvertOpToLLVMPattern<quake::AllocaOp> {
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
        qir_qubit_array_allocate, array_qbit_type,
        {rewriter.getIntegerType(64)}, parentModule);

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
class DeallocOpLowering : public ConvertOpToLLVMPattern<quake::DeallocOp> {
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
class ConcatOpLowering : public ConvertOpToLLVMPattern<quake::ConcatOp> {
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
class ExtractQubitOpLowering
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

class SubvecOpLowering : public ConvertOpToLLVMPattern<quake::SubVecOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(quake::SubVecOp subvec, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subvec->getLoc();
    auto parentModule = subvec->getParentOfType<ModuleOp>();
    auto *context = parentModule->getContext();
    constexpr auto rtSubvecFuncName = cudaq::opt::QIRArraySlice;
    auto arrayTy = cudaq::opt::getArrayType(context);
    auto resultTy = arrayTy;

    auto i32Ty = rewriter.getIntegerType(32);
    auto i64Ty = rewriter.getIntegerType(64);
    FlatSymbolRefAttr symbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
        rtSubvecFuncName, arrayTy, {arrayTy, i32Ty, i64Ty, i64Ty, i64Ty},
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
        subvec, resultTy, symbolRef,
        ValueRange{inArr, one32, lowArg, one64, highArg});
    return success();
  }
};

/// Lower the quake.reset op to QIR
template <typename ResetOpType>
class ResetLowering : public ConvertOpToLLVMPattern<ResetOpType> {
public:
  using Base = ConvertOpToLLVMPattern<ResetOpType>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(ResetOpType instOp, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentModule = instOp->template getParentOfType<ModuleOp>();
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

/// Lower single target Quantum ops with no parameter to QIR:
/// h, x, y, z, s, t
template <typename OP>
class OneTargetLowering : public ConvertOpToLLVMPattern<OP> {
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
    if (!instOp->template hasTrait<cudaq::Hermitian>() && instOp.getIsAdj())
      instName += "dg";

    if (numControls == 0) {
      // There are no control bits, so call the function directly.
      auto qirFunctionName = qirQisPrefix + instName;
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
class OneTargetOneParamLowering : public ConvertOpToLLVMPattern<OP> {
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
class OneTargetTwoParamLowering : public ConvertOpToLLVMPattern<OP> {
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
class TwoTargetLowering : public ConvertOpToLLVMPattern<OP> {
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
class MeasureLowering : public ConvertOpToLLVMPattern<OP> {
public:
  using Base = ConvertOpToLLVMPattern<OP>;
  using Base::Base;

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

    // Call a different measure function if we have a classical register name
    if (regName) {
      // Get the name
      auto regNameAttr = regName.cast<StringAttr>();
      auto regNameStr = regNameAttr.getValue().str();
      std::string regNameGlobalStr = regNameStr;

      // Change the function name
      qFunctionName += "__to__register";

      // Append a string type argument
      funcTypes.push_back(
          LLVM::LLVMPointerType::get(rewriter.getIntegerType(8)));

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
      args.push_back(castedRegNameRef);
    }

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

class GetVeqSizeOpLowering : public OpConversionPattern<quake::VeqSizeOp> {
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

class FuncToPtrOpLowering
    : public ConvertOpToLLVMPattern<cudaq::cc::FuncToPtrOp> {
public:
  using Base = ConvertOpToLLVMPattern<cudaq::cc::FuncToPtrOp>;
  using Base::Base;

  // This becomes a bitcast op.
  LogicalResult
  matchAndRewrite(cudaq::cc::FuncToPtrOp ftp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto operands = adaptor.getOperands();
    auto toTy = ftp.getType();
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(ftp, toTy, operands);
    return success();
  }
};

class StdvecInitOpLowering
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

class StdvecDataOpLowering
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

class StdvecSizeOpLowering
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

class UndefOpLowering : public ConvertOpToLLVMPattern<cudaq::cc::UndefOp> {
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

class CallableFuncOpLowering
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

class CallableClosureOpLowering
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

class InstantiateCallableOpLowering
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
    auto i64Ty = rewriter.getI64Type();
    Value one = rewriter.create<LLVM::ConstantOp>(loc, i64Ty, oneAttr);
    auto tuplePtrTy = cudaq::opt::factory::getPointerType(tupleTy);
    auto tmp = rewriter.create<LLVM::AllocaOp>(loc, tuplePtrTy, one);
    rewriter.create<LLVM::StoreOp>(loc, tupleVal, tmp);
    auto tupleArgTy = lambdaAsPairOfPointers(ctx);
    Value tupleArg = rewriter.create<LLVM::UndefOp>(loc, tupleArgTy);
    auto module = callable->getParentOfType<ModuleOp>();
    auto calledFunc = module.lookupSymbol<func::FuncOp>(callable.getCallee());
    Type sigTy = getTypeConverter()->convertType(calledFunc.getFunctionType());
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

/// Convert Quake dialect to LLVM-IR and QIR.
class QuakeToQIRLowering
    : public cudaq::opt::QuakeToQIRBase<QuakeToQIRLowering> {
public:
  QuakeToQIRLowering() = default;
  ModuleOp getModule() { return getOperation(); }
  void runOnOperation() override final {
    auto *context = getModule().getContext();
    LLVMConversionTarget target{*context};
    LLVMTypeConverter typeConverter(&getContext());
    typeConverter.addConversion(
        [&](quake::VeqType type) { return cudaq::opt::getArrayType(context); });
    typeConverter.addConversion(
        [&](quake::RefType type) { return cudaq::opt::getQubitType(context); });
    typeConverter.addConversion([&](cudaq::cc::LambdaType type) {
      return lambdaAsPairOfPointers(type.getContext());
    });
    typeConverter.addConversion([&](cudaq::cc::StdvecType type) {
      return cudaq::opt::factory::stdVectorImplType(type.getElementType());
    });
    RewritePatternSet patterns(context);

    populateAffineToStdConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    populateMathToLLVMConversionPatterns(typeConverter, patterns);
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);

    populateSCFToControlFlowConversionPatterns(patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    patterns.insert<GetVeqSizeOpLowering, MxToMz, MyToMz, ReturnBitRewrite>(
        context);
    patterns.insert<
        AllocaOpLowering, CallableClosureOpLowering, CallableFuncOpLowering,
        ConcatOpLowering, DeallocOpLowering, ExtractQubitOpLowering,
        FuncToPtrOpLowering, InstantiateCallableOpLowering,
        MeasureLowering<quake::MzOp>, OneTargetLowering<quake::HOp>,
        OneTargetLowering<quake::XOp>, OneTargetLowering<quake::YOp>,
        OneTargetLowering<quake::ZOp>, OneTargetLowering<quake::SOp>,
        OneTargetLowering<quake::TOp>, ResetLowering<quake::ResetOp>,
        OneTargetOneParamLowering<quake::R1Op>,
        OneTargetOneParamLowering<quake::RxOp>,
        OneTargetOneParamLowering<quake::RyOp>,
        OneTargetOneParamLowering<quake::RzOp>,
        OneTargetTwoParamLowering<quake::U2Op>,
        OneTargetTwoParamLowering<quake::U3Op>,
        TwoTargetLowering<quake::SwapOp>, StdvecDataOpLowering,
        StdvecInitOpLowering, StdvecSizeOpLowering, SubvecOpLowering,
        UndefOpLowering>(typeConverter);

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<ModuleOp>();

    if (failed(applyFullConversion(getModule(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> cudaq::opt::createConvertToQIRPass() {
  return std::make_unique<QuakeToQIRLowering>();
}
