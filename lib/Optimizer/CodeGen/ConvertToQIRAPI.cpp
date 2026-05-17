/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CodeGenOps.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/CodeGen/QuakeToExecMgr.h"
#include "cudaq/Optimizer/Transforms/Passes.h" // for GlobalizeArrayValues
#include "nlohmann/json.hpp"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "convert-to-qir-api"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUAKETOQIRAPI
#define GEN_PASS_DEF_QUAKETOQIRAPIPREP
#define GEN_PASS_DEF_QUAKETOQIRAPIFINAL
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

// Attribute name used to mark kernels that have been processed.
static constexpr const char FuncIsQIRAPI[] = "qir-api";
static constexpr const char InitialArgTypesAttrName[] = "initial_arg_types";

//===----------------------------------------------------------------------===//

static std::string getGateName(Operation *op) {
  return op->getName().stripDialect().str();
}

static std::string getGateFunctionPrefix(Operation *op) {
  return cudaq::opt::QIRQISPrefix + getGateName(op);
}

// The transport triple is a colon separated string that determines the profile,
// optional version, and an optional list of extensions being used.
inline static void splitTransportTriple(SmallVectorImpl<StringRef> &results,
                                        StringRef transportLayer) {
  transportLayer.split(results, ":");
}

constexpr std::array<std::string_view, 2> filterAdjointNames = {"s", "t"};

static SmallVector<Value> filterArgs(Operation *op, ValueRange adaptedArgs) {
  auto arrAttr = op->getAttrOfType<ArrayAttr>(InitialArgTypesAttrName);
  if (!arrAttr)
    return {};
  SmallVector<Value> result;
  assert(arrAttr.size() == adaptedArgs.size());
  for (auto [tyAttr, argval] : llvm::zip(arrAttr, adaptedArgs))
    if (cudaq::quake::isQuantumValueType(cast<TypeAttr>(tyAttr).getValue()))
      result.push_back(argval);
  return result;
}

template <typename OP>
std::pair<std::string, bool> generateGateFunctionName(OP op) {
  auto prefix = getGateFunctionPrefix(op.getOperation());
  auto gateName = getGateName(op.getOperation());
  if (op.isAdj()) {
    if (std::find(filterAdjointNames.begin(), filterAdjointNames.end(),
                  gateName) != filterAdjointNames.end()) {
      if (!op.getControls().empty())
        return {prefix + "dg__ctl", false};
      return {prefix + "__adj", false};
    }
  }
  if (!op.getControls().empty())
    return {prefix + "__ctl", false};
  return {prefix, true};
}

static Value createGlobalCString(Operation *op, Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 StringRef regName) {
  cudaq::IRBuilder irb(rewriter.getContext());
  auto mod = op->getParentOfType<ModuleOp>();
  auto nameObj = irb.genCStringLiteralAppendNul(loc, mod, regName);
  Value nameVal = cudaq::cc::AddressOfOp::create(
      rewriter, loc, cudaq::cc::PointerType::get(nameObj.getType()),
      nameObj.getName());
  auto cstrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
  return cudaq::cc::CastOp::create(rewriter, loc, cstrTy, nameVal);
}

/// Use modifier class classes to specialize the QIR API to a particular flavor
/// of QIR. For example, the names of the actual functions in "full QIR" are
/// different than the names used by the other API flavors.
namespace {

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

/// Type converter for converting quake dialect to one of the QIR APIs. This
/// class is used for conversions as well as instantiating QIR types in
/// conversion patterns.

struct QIRAPITypeConverter : public TypeConverter {
  using TypeConverter::convertType;

  QIRAPITypeConverter(bool useOpaque) : useOpaque(useOpaque) {
    addConversion([&](Type ty) { return ty; });
    addConversion([&](FunctionType ft) { return convertFunctionType(ft); });
    addConversion([&](cudaq::cc::PointerType ty) {
      return cudaq::cc::PointerType::get(convertType(ty.getElementType()));
    });
    addConversion([&](cudaq::cc::CallableType ty) {
      auto newSig = cast<FunctionType>(convertType(ty.getSignature()));
      return cudaq::cc::CallableType::get(newSig);
    });
    addConversion([&](cudaq::cc::IndirectCallableType ty) {
      auto newSig = cast<FunctionType>(convertType(ty.getSignature()));
      return cudaq::cc::IndirectCallableType::get(newSig);
    });
    addConversion([&](cudaq::quake::VeqType ty) {
      return getArrayType(ty.getContext());
    });
    addConversion([&](cudaq::quake::RefType ty) {
      return getQubitType(ty.getContext());
    });
    addConversion([&](cudaq::quake::WireType ty) {
      return getQubitType(ty.getContext());
    });
    addConversion([&](cudaq::quake::ControlType ty) {
      return getQubitType(ty.getContext());
    });
    addConversion([&](cudaq::quake::CableType ty) {
      return getArrayType(ty.getContext());
    });
    addConversion([&](cudaq::quake::MeasureType ty) {
      return getResultType(ty.getContext());
    });
    addConversion(
        [&](cudaq::quake::StruqType ty) { return convertStruqType(ty); });
    // `!cc.measure_handle` is the IR alias of `cudaq::measure_handle`. The
    // QIR API models classical measurement results as `Result*` (legacy
    // `quake.measure`) or as opaque `i64` payloads (handle form). The
    // measurement / discriminate patterns bridge `Result*` to/from `i64`
    // when the original op carried a handle.
    addConversion([](cudaq::cc::MeasureHandleType ty) -> Type {
      return IntegerType::get(ty.getContext(), 64);
    });
    // Recursively convert handle / quake types nested in CC array and
    // stdvec types so that container-shaped function signatures, allocations,
    // and pointers see consistent post-conversion element types.
    addConversion([&](cudaq::cc::ArrayType ty) -> Type {
      Type newEleTy = convertType(ty.getElementType());
      if (newEleTy == ty.getElementType())
        return ty;
      if (ty.isUnknownSize())
        return cudaq::cc::ArrayType::get(newEleTy);
      return cudaq::cc::ArrayType::get(ty.getContext(), newEleTy, ty.getSize());
    });
    addConversion([&](cudaq::cc::StdvecType ty) -> Type {
      Type newEleTy = convertType(ty.getElementType());
      if (newEleTy == ty.getElementType())
        return ty;
      return cudaq::cc::StdvecType::get(ty.getContext(), newEleTy);
    });
  }

  Type convertFunctionType(FunctionType ty) {
    SmallVector<Type> args;
    if (failed(convertTypes(ty.getInputs(), args)))
      return {};
    SmallVector<Type> res;
    if (failed(convertTypes(ty.getResults(), res)))
      return {};
    return FunctionType::get(ty.getContext(), args, res);
  }

  Type convertStruqType(cudaq::quake::StruqType ty) {
    SmallVector<Type> mems;
    mems.reserve(ty.getNumMembers());
    if (failed(convertTypes(ty.getMembers(), mems)))
      return {};
    return cudaq::cc::StructType::get(ty.getContext(), mems);
  }

  Type getQubitType(MLIRContext *ctx) {
    return cudaq::cg::getQubitType(ctx, useOpaque);
  }
  Type getArrayType(MLIRContext *ctx) {
    return cudaq::cg::getArrayType(ctx, useOpaque);
  }
  Type getResultType(MLIRContext *ctx) {
    return cudaq::cg::getResultType(ctx, useOpaque);
  }

  bool useOpaque;
};
} // namespace

namespace {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

template <typename M>
struct AllocaOpToCallsRewrite
    : public OpConversionPattern<cudaq::quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::AllocaOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // If this alloc is just returning a qubit
    if (auto resultType =
            dyn_cast_if_present<cudaq::quake::RefType>(alloc.getType())) {
      StringRef qirQubitAllocate = cudaq::opt::QIRQubitAllocate;
      Type qubitTy = M::getQubitType(rewriter.getContext());

      rewriter.replaceOpWithNewOp<func::CallOp>(alloc, TypeRange{qubitTy},
                                                qirQubitAllocate, ValueRange{});
      return success();
    }

    // Create a QIR call to allocate the qubits.
    StringRef qirQubitArrayAllocate = cudaq::opt::QIRArrayQubitAllocateArray;
    Type arrayQubitTy = M::getArrayType(rewriter.getContext());

    // AllocaOp could have a size operand, or the size could be compile time
    // known and encoded in the veq return type.
    Value sizeOperand;
    auto loc = alloc.getLoc();
    if (adaptor.getOperands().empty()) {
      auto type = cast<cudaq::quake::VeqType>(alloc.getType());
      if (!type.hasSpecifiedSize())
        return failure();
      auto constantSize = type.getSize();
      sizeOperand =
          arith::ConstantIntOp::create(rewriter, loc, constantSize, 64);
    } else {
      sizeOperand = adaptor.getOperands().front();
      auto sizeOpTy = cast<IntegerType>(sizeOperand.getType());
      if (sizeOpTy.getWidth() < 64)
        sizeOperand = cudaq::cc::CastOp::create(
            rewriter, loc, rewriter.getI64Type(), sizeOperand,
            cudaq::cc::CastOpMode::Unsigned);
      else if (sizeOpTy.getWidth() > 64)
        sizeOperand = cudaq::cc::CastOp::create(
            rewriter, loc, rewriter.getI64Type(), sizeOperand);
    }

    // Replace the AllocaOp with the QIR call.
    rewriter.replaceOpWithNewOp<func::CallOp>(alloc, TypeRange{arrayQubitTy},
                                              qirQubitArrayAllocate,
                                              ValueRange{sizeOperand});
    return success();
  }
};

template <typename M>
struct NullWireOpToCallsRewrite
    : public OpConversionPattern<cudaq::quake::NullWireOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::NullWireOp nullwire, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef qirQubitAllocate = cudaq::opt::QIRQubitAllocate;
    Type qubitTy = M::getQubitType(rewriter.getContext());

    rewriter.replaceOpWithNewOp<func::CallOp>(nullwire, TypeRange{qubitTy},
                                              qirQubitAllocate, ValueRange{});
    return success();
  }
};

template <typename M>
struct NullCableOpToCallsRewrite
    : public OpConversionPattern<cudaq::quake::NullCableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::NullCableOp nullcable, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Create a QIR call to allocate the qubits.
    StringRef qirQubitArrayAllocate = cudaq::opt::QIRArrayQubitAllocateArray;
    Type arrayQubitTy = M::getArrayType(rewriter.getContext());

    // NullCableOp must have a constant size encoded in the `!quake.cable`
    // return type.
    auto loc = nullcable.getLoc();
    cudaq::quake::CableType type = nullcable.getType();
    auto constantSize = type.getSize();
    Value sizeOperand =
        arith::ConstantIntOp::create(rewriter, loc, constantSize, 64);

    // Replace the NullCableOp with the QIR call.
    rewriter.replaceOpWithNewOp<func::CallOp>(
        nullcable, TypeRange{arrayQubitTy}, qirQubitArrayAllocate,
        ValueRange{sizeOperand});
    return success();
  }
};

template <typename M>
struct AllocaOpToIntRewrite
    : public OpConversionPattern<cudaq::quake::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  // Precondition: every allocation must have been annotated with a starting
  // index by the preparation phase.
  LogicalResult
  matchAndRewrite(cudaq::quake::AllocaOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!alloc->hasAttr(cudaq::opt::StartingOffsetAttrName))
      return alloc.emitOpError("allocation must be annotated.");

    auto loc = alloc.getLoc();
    // If this alloc is just returning a qubit, so just replace it with the
    // attribute.
    Type ty = alloc.getType();
    if (!ty)
      return alloc.emitOpError("quake alloca is malformed");
    auto startingOffsetAttr =
        alloc->getAttr(cudaq::opt::StartingOffsetAttrName);
    auto startingOffset = cast<IntegerAttr>(startingOffsetAttr).getInt();

    // In the case this is allocating a single qubit, we can just substitute
    // the startingIndex as the qubit value. Voila!
    if (auto resultType = dyn_cast<cudaq::quake::RefType>(ty)) {
      Value index =
          arith::ConstantIntOp::create(rewriter, loc, startingOffset, 64);
      auto qubitTy = M::getQubitType(rewriter.getContext());
      rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(alloc, qubitTy, index);
      return success();
    }

    auto veqTy = dyn_cast<cudaq::quake::VeqType>(ty);
    if (!veqTy)
      return alloc.emitOpError("quake alloca must be a veq");
    if (!veqTy.hasSpecifiedSize())
      return alloc.emitOpError("quake alloca must be a veq with constant size");

    // Otherwise, the allocation is of a sequence of qubits. Here, we allocate a
    // constant array value with the qubit integral values in an ascending
    // sequence. These will be accessed by extract_value or used collectively.
    auto *ctx = rewriter.getContext();
    const std::int64_t veqSize = veqTy.getSize();
    auto arrTy = cudaq::cc::ArrayType::get(ctx, rewriter.getI64Type(), veqSize);
    SmallVector<std::int64_t> data;
    for (std::int64_t i = 0; i < veqSize; ++i)
      data.emplace_back(startingOffset + i);
    auto arr = cudaq::cc::ConstantArrayOp::create(
        rewriter, loc, arrTy, rewriter.getI64ArrayAttr(data));
    Type qirArrTy = M::getArrayType(rewriter.getContext());
    rewriter.replaceOpWithNewOp<cudaq::codegen::MaterializeConstantArrayOp>(
        alloc, qirArrTy, arr);
    return success();
  }
};

template <typename M>
struct NullWireOpToIntRewrite
    : public OpConversionPattern<cudaq::quake::NullWireOp> {
  using OpConversionPattern::OpConversionPattern;

  // Precondition: every allocation must have been annotated with a starting
  // index by the preparation phase.
  LogicalResult
  matchAndRewrite(cudaq::quake::NullWireOp nullwire, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto startingOffsetAttr =
        nullwire->getAttr(cudaq::opt::StartingOffsetAttrName);
    if (!startingOffsetAttr)
      return nullwire.emitOpError("allocation must be annotated.");

    auto loc = nullwire.getLoc();
    auto startingOffset = cast<IntegerAttr>(startingOffsetAttr).getInt();

    // In this case this is allocating a single qubit, so we can just substitute
    // the startingIndex as the qubit value. Voila!
    Value index =
        arith::ConstantIntOp::create(rewriter, loc, startingOffset, 64);
    auto qubitTy = M::getQubitType(rewriter.getContext());
    rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(nullwire, qubitTy, index);
    return success();
  }
};

template <typename M>
struct NullCableOpToIntRewrite
    : public OpConversionPattern<cudaq::quake::NullCableOp> {
  using OpConversionPattern::OpConversionPattern;

  // Precondition: every allocation must have been annotated with a starting
  // index by the preparation phase.
  LogicalResult
  matchAndRewrite(cudaq::quake::NullCableOp nullcable, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto startingOffsetAttr =
        nullcable->getAttr(cudaq::opt::StartingOffsetAttrName);
    if (!startingOffsetAttr)
      return nullcable.emitOpError("allocation must be annotated.");

    auto loc = nullcable.getLoc();
    auto startingOffset = cast<IntegerAttr>(startingOffsetAttr).getInt();

    cudaq::quake::CableType cableTy = nullcable.getType();
    if (!cableTy)
      return nullcable.emitOpError("quake null_cable must be a cable");

    // Otherwise, the allocation is of a sequence of qubits. Here, we allocate a
    // constant array value with the qubit integral values in an ascending
    // sequence. These will be accessed by extract_value or used collectively.
    auto *ctx = rewriter.getContext();
    const std::int64_t cableSize = cableTy.getSize();
    auto arrTy =
        cudaq::cc::ArrayType::get(ctx, rewriter.getI64Type(), cableSize);
    SmallVector<std::int64_t> data;
    for (std::int64_t i = 0; i < cableSize; ++i)
      data.emplace_back(startingOffset + i);
    auto arr = cudaq::cc::ConstantArrayOp::create(
        rewriter, loc, arrTy, rewriter.getI64ArrayAttr(data));
    Type qirArrTy = M::getArrayType(rewriter.getContext());
    rewriter.replaceOpWithNewOp<cudaq::codegen::MaterializeConstantArrayOp>(
        nullcable, qirArrTy, arr);
    return success();
  }
};

template <typename OP>
Type getInitialType(OP op, unsigned off) {
  ArrayAttr initialArgs =
      op->template getAttrOfType<ArrayAttr>(InitialArgTypesAttrName);
  if (!initialArgs)
    return {};
  return cast<TypeAttr>(initialArgs[off]).getValue();
}

template <typename M>
struct ApplyNoiseOpRewrite
    : public OpConversionPattern<cudaq::quake::ApplyNoiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::ApplyNoiseOp noise, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = noise.getLoc();

    const unsigned paramOffset = noise.getKey() ? 1 : 0;
    if (!noise.getNoiseFunc()) {
      // This is the key-based variant. Call the generalized version of the
      // apply_kraus_channel helper function. Let it do all the conversions into
      // contiguous buffers for us, greatly simplifying codegen here.
      SmallVector<Value> args;
      const bool pushASpan =
          adaptor.getParameters().size() == 1 &&
          isa<cudaq::cc::StdvecType>(getInitialType(noise, paramOffset));
      const bool usingDouble = [&]() {
        if (adaptor.getParameters().empty())
          return true;
        Type param0Ty = getInitialType(noise, paramOffset);
        if (pushASpan)
          return cast<cudaq::cc::StdvecType>(param0Ty).getElementType() ==
                 rewriter.getF64Type();
        return cast<cudaq::cc::PointerType>(param0Ty).getElementType() ==
               rewriter.getF64Type();
      }();
      if (usingDouble) {
        auto code = static_cast<std::int64_t>(
            cudaq::opt::KrausChannelDataKind::DoubleKind);
        args.push_back(arith::ConstantIntOp::create(rewriter, loc, code, 64));
      } else {
        auto code = static_cast<std::int64_t>(
            cudaq::opt::KrausChannelDataKind::FloatKind);
        args.push_back(arith::ConstantIntOp::create(rewriter, loc, code, 64));
      }
      args.push_back(adaptor.getKey());
      if (pushASpan) {
        args.push_back(arith::ConstantIntOp::create(rewriter, loc, 1, 64));
        args.push_back(arith::ConstantIntOp::create(rewriter, loc, 0, 64));
      } else {
        args.push_back(arith::ConstantIntOp::create(rewriter, loc, 0, 64));
        auto numParams = std::distance(adaptor.getParameters().begin(),
                                       adaptor.getParameters().end());
        args.push_back(
            arith::ConstantIntOp::create(rewriter, loc, numParams, 64));
      }
      auto numTargets =
          std::distance(adaptor.getQubits().begin(), adaptor.getQubits().end());
      args.push_back(
          arith::ConstantIntOp::create(rewriter, loc, numTargets, 64));
      if (pushASpan) {
        Value stdvec = adaptor.getParameters()[0];
        auto stdvecTy =
            cast<cudaq::cc::StdvecType>(getInitialType(noise, paramOffset));
        auto dataTy = cudaq::cc::PointerType::get(
            cudaq::cc::ArrayType::get(stdvecTy.getElementType()));
        args.push_back(
            cudaq::cc::StdvecDataOp::create(rewriter, loc, dataTy, stdvec));
        args.push_back(cudaq::cc::StdvecSizeOp::create(
            rewriter, loc, rewriter.getI64Type(), stdvec));
      } else {
        args.append(adaptor.getParameters().begin(),
                    adaptor.getParameters().end());
      }
      args.append(adaptor.getQubits().begin(), adaptor.getQubits().end());

      rewriter.replaceOpWithNewOp<cudaq::cc::VarargCallOp>(
          noise, TypeRange{}, cudaq::opt::QISApplyKrausChannel, args);
      return success();
    }

    // This is a noise_func variant. Call the noise function. There are two
    // cases that must be considered.
    //
    // 1. The parameters to the Kraus channel are passed in an object of type
    // `std::vector<double>`. To do that requires a bunch of code to translate
    // the span of doubles on the device side into a `std::vector<double>` on
    // the stack for passing to the host-side function. It is ABSOLUTELY
    // CRITICAL that the host side NOT use move semantics or otherwise try to
    // claim ownership of the fake vector being passed back as that will crash
    // the executable. The host side should not modify the content of the vector
    // either. These assumptions are made in this code as the argument to the
    // host side is `const std::vector<double>&`. This code must also modify the
    // signature of the called function since the bridge will have assumed it
    // was a span. Again all of this chicanery is so we don't call the function
    // with the wrong data type and/or have the callee try to modify the vector.
    // Such actions will result in the executable CRASHING or giving WRONG
    // ANSWERS.
    //
    // 2. Easier by a jaw-dropping margin, just pass rvalue references to double
    // values, each individually, back to the host-side function. Since that's
    // already the case, we just append the operands.
    SmallVector<Value> args;
    if (adaptor.getParameters().size() == 1 &&
        isa<cudaq::cc::StdvecType>(getInitialType(noise, paramOffset))) {
      Value svp = adaptor.getParameters()[0];
      // Convert the device-side span back to a host-side vector so that C++
      // doesn't crash.
      auto stdvecTy =
          cast<cudaq::cc::StdvecType>(getInitialType(noise, paramOffset));
      auto *ctx = rewriter.getContext();
      auto ptrTy = cudaq::cc::PointerType::get(stdvecTy.getElementType());
      auto ptrArrTy = cudaq::cc::PointerType::get(
          cudaq::cc::ArrayType::get(stdvecTy.getElementType()));
      auto hostVecTy = cudaq::cc::ArrayType::get(ctx, ptrTy, 3);
      auto hostVec = cudaq::cc::AllocaOp::create(rewriter, loc, hostVecTy);
      Value startPtr =
          cudaq::cc::StdvecDataOp::create(rewriter, loc, ptrArrTy, svp);
      auto i64Ty = rewriter.getI64Type();
      Value len = cudaq::cc::StdvecSizeOp::create(rewriter, loc, i64Ty, svp);
      Value endPtr = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, ptrTy, startPtr,
          ArrayRef<cudaq::cc::ComputePtrArg>{len});
      Value castStartPtr =
          cudaq::cc::CastOp::create(rewriter, loc, ptrTy, startPtr);
      auto ptrPtrTy = cudaq::cc::PointerType::get(ptrTy);
      Value ptr0 = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, ptrPtrTy, hostVec,
          ArrayRef<cudaq::cc::ComputePtrArg>{0});
      cudaq::cc::StoreOp::create(rewriter, loc, castStartPtr, ptr0);
      Value ptr1 = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, ptrPtrTy, hostVec,
          ArrayRef<cudaq::cc::ComputePtrArg>{1});
      cudaq::cc::StoreOp::create(rewriter, loc, endPtr, ptr1);
      Value ptr2 = cudaq::cc::ComputePtrOp::create(
          rewriter, loc, ptrPtrTy, hostVec,
          ArrayRef<cudaq::cc::ComputePtrArg>{2});
      cudaq::cc::StoreOp::create(rewriter, loc, endPtr, ptr2);

      // N.B. This pointer must be treated as const by the C++ side and should
      // never have move semantics!
      args.push_back(hostVec);

      // Finally, we need to modify the called function's signature.
      auto module = noise->getParentOfType<ModuleOp>();
      auto funcTy = FunctionType::get(ctx, {}, {});
      auto [fn, flag] = cudaq::opt::factory::getOrAddFunc(
          loc, *noise.getNoiseFunc(), funcTy, module);
      funcTy = fn.getFunctionType();
      SmallVector<Type> inputTys{funcTy.getInputs().begin(),
                                 funcTy.getInputs().end()};
      inputTys[0] = hostVec.getType();
      auto newFuncTy = FunctionType::get(ctx, inputTys, funcTy.getResults());
      fn.setFunctionType(newFuncTy);
    } else {
      args.append(adaptor.getParameters().begin(),
                  adaptor.getParameters().end());
    }
    SmallVector<Value> qubits;
    SmallVector<Value> converted;
    Type qirArrTy = M::getArrayType(rewriter.getContext());
    SmallVector<Type> origQubitTys;
    for (auto [i, _] : llvm::enumerate(noise.getQubits()))
      origQubitTys.push_back(getInitialType(
          noise, paramOffset + adaptor.getParameters().size() + i));
    for (auto [qb, oa] : llvm::zip(adaptor.getQubits(), origQubitTys)) {
      if (isa<cudaq::quake::VeqType>(oa)) {
        auto svec = func::CallOp::create(rewriter, loc, qirArrTy,
                                         cudaq::opt::QISConvertArrayToStdvec,
                                         ValueRange{qb});
        qb = svec.getResult(0);
        converted.push_back(qb);
      }
      qubits.push_back(qb);
    }
    args.append(qubits.begin(), qubits.end());
    rewriter.replaceOpWithNewOp<func::CallOp>(noise, TypeRange{},
                                              *noise.getNoiseFunc(), args);
    for (auto v : converted)
      func::CallOp::create(rewriter, loc, TypeRange{},
                           cudaq::opt::QISFreeConvertedStdvec, ValueRange{v});
    return success();
  }
};

struct MaterializeConstantArrayOpRewrite
    : public OpRewritePattern<cudaq::codegen::MaterializeConstantArrayOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::codegen::MaterializeConstantArrayOp mca,
                                PatternRewriter &rewriter) const override {
    Value arr = mca.getConstArray();
    if (auto arrVal = arr.getDefiningOp<cudaq::cc::LoadOp>()) {
      Type ty = mca.getType();
      auto ptr = arrVal.getPtrvalue();
      rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(mca, ty, ptr);
      return success();
    }
    return failure();
  }
};

/// This helper base class provides shared functionality to convert single
/// qubits (`!quake.ref`) to vectors of qubits (`!quake.veq`) to satisfy the QIR
/// API.
template <typename M, typename OP>
struct QubitHelperConversionPattern : public OpConversionPattern<OP> {
  using Base = OpConversionPattern<OP>;
  using Base::Base;

  Value wrapQubitAsArray(Location loc, ConversionPatternRewriter &rewriter,
                         Value val, Type origTy) const {
    if (isa<cudaq::quake::VeqType>(origTy))
      return val;

    // Create a QIR array container of 1 element.
    auto ptrTy = cudaq::cc::PointerType::get(rewriter.getNoneType());
    Value sizeofPtrVal = cudaq::cc::SizeOfOp::create(
        rewriter, loc, rewriter.getI32Type(), ptrTy);
    Value one = arith::ConstantIntOp::create(rewriter, loc, 1, 64);
    Type arrayTy = M::getArrayType(rewriter.getContext());
    auto newArr = func::CallOp::create(rewriter, loc, TypeRange{arrayTy},
                                       cudaq::opt::QIRArrayCreateArray,
                                       ArrayRef<Value>{sizeofPtrVal, one});
    Value result = newArr.getResult(0);

    // Get a pointer to element 0.
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
    Type qubitTy = M::getQubitType(rewriter.getContext());
    auto ptrQubitTy = cudaq::cc::PointerType::get(qubitTy);
    auto elePtr = func::CallOp::create(rewriter, loc, TypeRange{ptrQubitTy},
                                       cudaq::opt::QIRArrayGetElementPtr1d,
                                       ArrayRef<Value>{result, zero});

    // Write the qubit into the array at position 0.
    auto castVal = cudaq::cc::CastOp::create(rewriter, loc, qubitTy, val);
    Value addr = elePtr.getResult(0);
    cudaq::cc::StoreOp::create(rewriter, loc, castVal, addr);

    return result;
  }
};

template <typename M>
struct ConcatOpRewrite
    : public QubitHelperConversionPattern<M, cudaq::quake::ConcatOp> {
  using Base = QubitHelperConversionPattern<M, cudaq::quake::ConcatOp>;
  using Base::Base;

  // For this rewrite, we walk the list of operands (if any) and for each
  // operand, $o$, we ensure $o$ is already of type QIR array or convert $o$ to
  // the array type using QIR functions. Then, we walk the list and pairwise
  // concatenate each operand. First, take $c$ to be $o_0$ and then update $c$
  // to be the concat of the previous $c$ and $o_i \forall i \in \{ 1..N \}$.
  // This algorithm will generate a linear number of concat calls for the number
  // of operands.
  LogicalResult
  matchAndRewrite(cudaq::quake::ConcatOp concat, Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().empty()) {
      rewriter.eraseOp(concat);
      return success();
    }

    auto loc = concat.getLoc();
    Type arrayTy = M::getArrayType(rewriter.getContext());
    Value firstOperand = adaptor.getOperands().front();
    Type firstTy = getInitialType(concat, 0);
    Value resultArray =
        Base::wrapQubitAsArray(loc, rewriter, firstOperand, firstTy);
    SmallVector<Type> origTys;
    for (auto [i, _] : llvm::enumerate(adaptor.getOperands().drop_front()))
      origTys.push_back(getInitialType(concat, i + 1));
    for (auto [next, origTy] :
         llvm::zip(adaptor.getOperands().drop_front(), origTys)) {
      Value wrapNext = Base::wrapQubitAsArray(loc, rewriter, next, origTy);
      auto appended = func::CallOp::create(
          rewriter, loc, arrayTy, cudaq::opt::QIRArrayConcatArray,
          ArrayRef<Value>{resultArray, wrapNext});
      resultArray = appended.getResult(0);
    }
    rewriter.replaceOp(concat, resultArray);
    return success();
  }
};

template <typename M>
struct BundleCableOpRewrite
    : public QubitHelperConversionPattern<M, cudaq::quake::BundleCableOp> {
  using Base = QubitHelperConversionPattern<M, cudaq::quake::BundleCableOp>;
  using Base::Base;

  // Note that this is the same as quake.concat.
  LogicalResult
  matchAndRewrite(cudaq::quake::BundleCableOp bundle, Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().empty()) {
      rewriter.eraseOp(bundle);
      return success();
    }

    auto loc = bundle.getLoc();
    Type arrayTy = M::getArrayType(rewriter.getContext());
    Value firstOperand = adaptor.getOperands().front();
    Type firstTy = getInitialType(bundle, 0);
    Value resultArray =
        Base::wrapQubitAsArray(loc, rewriter, firstOperand, firstTy);
    for (auto [i, next] : llvm::enumerate(adaptor.getOperands().drop_front())) {
      Type nextTy = getInitialType(bundle, i + 1);
      Value wrapNext = Base::wrapQubitAsArray(loc, rewriter, next, nextTy);
      auto appended = func::CallOp::create(
          rewriter, loc, arrayTy, cudaq::opt::QIRArrayConcatArray,
          ArrayRef<Value>{resultArray, wrapNext});
      resultArray = appended.getResult(0);
    }
    rewriter.replaceOp(bundle, resultArray);
    return success();
  }
};

struct DeallocOpRewrite : public OpConversionPattern<cudaq::quake::DeallocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::DeallocOp dealloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = dealloc.getReference().getType();
    StringRef qirFuncName = isa<cudaq::quake::VeqType>(ty)
                                ? cudaq::opt::QIRArrayQubitReleaseArray
                                : cudaq::opt::QIRArrayQubitReleaseQubit;
    rewriter.replaceOpWithNewOp<func::CallOp>(dealloc, TypeRange{}, qirFuncName,
                                              adaptor.getReference());
    return success();
  }
};

struct SinkOpRewrite : public OpConversionPattern<cudaq::quake::SinkOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::SinkOp sink, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ty = sink.getTarget().getType();
    StringRef qirFuncName = isa<cudaq::quake::CableType>(ty)
                                ? cudaq::opt::QIRArrayQubitReleaseArray
                                : cudaq::opt::QIRArrayQubitReleaseQubit;
    rewriter.replaceOpWithNewOp<func::CallOp>(sink, TypeRange{}, qirFuncName,
                                              adaptor.getTarget());
    return success();
  }
};

template <typename OP>
struct DeallocLikeErase : public OpConversionPattern<OP> {
  using Base = OpConversionPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP op, Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

using DeallocOpErase = DeallocLikeErase<cudaq::quake::DeallocOp>;
using SinkOpErase = DeallocLikeErase<cudaq::quake::SinkOp>;

struct DiscriminateOpRewrite
    : public OpConversionPattern<cudaq::quake::DiscriminateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::DiscriminateOp disc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = disc.getLoc();
    Value m = adaptor.getMeasurement();
    auto i1PtrTy = cudaq::cc::PointerType::get(rewriter.getI1Type());
    auto cast = cudaq::cc::CastOp::create(rewriter, loc, i1PtrTy, m);
    rewriter.replaceOpWithNewOp<cudaq::cc::LoadOp>(disc, cast);
    return success();
  }
};

// Supported QIR versions.
enum struct QirVersion { version_0_1, version_1_0 };

template <typename M>
struct DiscriminateOpToCallRewrite
    : public OpConversionPattern<cudaq::quake::DiscriminateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::DiscriminateOp disc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = disc.getLoc();
    // Handle-form callers feed `quake.discriminate` an `i64` payload (the
    // converted form of `!cc.measure_handle`). Restore the `Result*` view
    // expected by the QIR read-result functions.
    SmallVector<Value> operands{adaptor.getOperands().begin(),
                                adaptor.getOperands().end()};
    if (operands.size() == 1 && isa<IntegerType>(operands.front().getType())) {
      auto resultTy = M::getResultType(rewriter.getContext());
      operands.front() =
          cudaq::cc::CastOp::create(rewriter, loc, resultTy, operands.front());
    }
    if constexpr (M::discriminateToClassical) {
      if constexpr (M::qirVersion == QirVersion::version_1_0) {
        rewriter.replaceOpWithNewOp<func::CallOp>(
            disc, rewriter.getI1Type(), cudaq::opt::qir1_0::ReadResult,
            operands);
      } else {
        rewriter.replaceOpWithNewOp<func::CallOp>(
            disc, rewriter.getI1Type(), cudaq::opt::qir0_1::ReadResultBody,
            operands);
      }
    } else {
      // NB: the double cast here is to avoid folding the pointer casts.
      auto i64Ty = rewriter.getI64Type();
      auto unu = cudaq::cc::CastOp::create(rewriter, loc, i64Ty,
                                           adaptor.getOperands());
      auto ptrI1Ty = cudaq::cc::PointerType::get(rewriter.getI1Type());
      auto du = cudaq::cc::CastOp::create(rewriter, loc, ptrI1Ty, unu);
      rewriter.replaceOpWithNewOp<cudaq::cc::LoadOp>(disc, du);
    }
    return success();
  }

  const std::string version;
};

template <typename M>
struct ExtractRefOpRewrite
    : public OpConversionPattern<cudaq::quake::ExtractRefOp> {
  using OpConversionPattern::OpConversionPattern;

  // There are two cases depending on which flavor of QIR is being generated.
  // For full QIR, we need to generate calls to QIR functions to select the
  // qubit from a QIR array.
  // For the profile QIRs, we replace this with a `cc.extract_value` operation,
  // which will be canonicalized into a constant.
  LogicalResult
  matchAndRewrite(cudaq::quake::ExtractRefOp extract, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = extract.getLoc();
    auto veq = adaptor.getVeq();
    auto i64Ty = rewriter.getI64Type();

    Value index;
    if (!adaptor.getIndex()) {
      index = arith::ConstantIntOp::create(rewriter, loc,
                                           extract.getConstantIndex(), 64);
    } else {
      index = adaptor.getIndex();
      if (index.getType().isIntOrFloat()) {
        if (cast<IntegerType>(index.getType()).getWidth() < 64)
          index = cudaq::cc::CastOp::create(rewriter, loc, i64Ty, index,
                                            cudaq::cc::CastOpMode::Unsigned);
        else if (cast<IntegerType>(index.getType()).getWidth() > 64)
          index = cudaq::cc::CastOp::create(rewriter, loc, i64Ty, index);
      }
    }
    auto qubitTy = M::getQubitType(rewriter.getContext());

    if (auto mca =
            veq.getDefiningOp<cudaq::codegen::MaterializeConstantArrayOp>()) {
      // This is the profile QIR case.
      auto ext = cudaq::cc::ExtractValueOp::create(rewriter, loc, i64Ty,
                                                   mca.getConstArray(), index);
      rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(extract, qubitTy, ext);
      return success();
    }

    // Otherwise, this must be full QIR.
    auto call = func::CallOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(qubitTy),
        cudaq::opt::QIRArrayGetElementPtr1d, ArrayRef<Value>{veq, index});
    rewriter.replaceOpWithNewOp<cudaq::cc::LoadOp>(extract, call.getResult(0));
    return success();
  }
};

template <typename M>
struct SplitCableOpRewrite
    : public OpConversionPattern<cudaq::quake::SplitCableOp> {
  using OpConversionPattern::OpConversionPattern;

  // There are two cases depending on which flavor of QIR is being generated.
  // For full QIR, we need to generate calls to QIR functions to select the
  // qubit from a QIR array.
  // For the profile QIRs, we replace this with a `cc.extract_value` operation,
  // which will be canonicalized into a constant.
  LogicalResult
  matchAndRewrite(cudaq::quake::SplitCableOp split, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = split.getLoc();
    auto cab = adaptor.getCable();
    auto qubitTy = M::getQubitType(rewriter.getContext());

    SmallVector<Value> qubits;
    std::uint64_t size =
        cast<cudaq::quake::CableType>(getInitialType(split, 0)).getSize();
    for (std::uint64_t i = 0; i < size; ++i) {
      Value index = arith::ConstantIntOp::create(rewriter, loc, i, 64);
      auto call = func::CallOp::create(
          rewriter, loc, cudaq::cc::PointerType::get(qubitTy),
          cudaq::opt::QIRArrayGetElementPtr1d, ArrayRef<Value>{cab, index});
      qubits.push_back(
          cudaq::cc::LoadOp::create(rewriter, loc, call.getResult(0)));
    }
    rewriter.replaceOp(split, ValueRange{qubits});
    return success();
  }
};

struct GetMemberOpRewrite
    : public OpConversionPattern<cudaq::quake::GetMemberOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::GetMemberOp member, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto toTy = getTypeConverter()->convertType(member.getType());
    std::int32_t position = adaptor.getIndex();
    rewriter.replaceOpWithNewOp<cudaq::cc::ExtractValueOp>(
        member, toTy, adaptor.getStruq(), position);
    return success();
  }
};

struct VeqSizeOpRewrite : public OpConversionPattern<cudaq::quake::VeqSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::VeqSizeOp veqsize, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::CallOp>(
        veqsize, TypeRange{veqsize.getType()}, cudaq::opt::QIRArrayGetSize,
        adaptor.getOperands());
    return success();
  }
};

struct MakeStruqOpRewrite
    : public OpConversionPattern<cudaq::quake::MakeStruqOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::MakeStruqOp mkstruq, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = mkstruq.getLoc();
    auto *ctx = rewriter.getContext();
    auto toTy = getTypeConverter()->convertType(mkstruq.getType());
    Value result = cudaq::cc::UndefOp::create(rewriter, loc, toTy);
    std::int64_t count = 0;
    for (auto op : adaptor.getOperands()) {
      auto off = DenseI64ArrayAttr::get(ctx, ArrayRef<std::int64_t>{count});
      result = cudaq::cc::InsertValueOp::create(rewriter, loc, toTy, result, op,
                                                off);
      count++;
    }
    rewriter.replaceOp(mkstruq, result);
    return success();
  }
};

template <typename M>
struct QmemRAIIOpRewrite : public OpConversionPattern<cudaq::codegen::RAIIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::codegen::RAIIOp raii, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = raii.getLoc();
    auto arrayTy = M::getArrayType(rewriter.getContext());

    // Get the CC Pointer for the state
    auto ccState = adaptor.getInitState();

    // Inspect the element type of the complex data, need to
    // know if its f32 or f64
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

    // Cascade to set functionName.
    StringRef functionName;
    Type ptrTy;
    if (isa<cudaq::quake::StateType>(eleTy)) {
      functionName = cudaq::opt::QIRArrayQubitAllocateArrayWithCudaqStatePtr;
      ptrTy = cudaq::cc::PointerType::get(
          cudaq::quake::StateType::get(rewriter.getContext()));
    } else if (eleTy == rewriter.getF64Type()) {
      if (fromComplex) {
        functionName = cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex64;
        ptrTy = cudaq::cc::PointerType::get(
            ComplexType::get(rewriter.getF64Type()));
      } else {
        functionName = cudaq::opt::QIRArrayQubitAllocateArrayWithStateFP64;
        ptrTy = cudaq::cc::PointerType::get(rewriter.getF64Type());
      }
    } else if (eleTy == rewriter.getF32Type()) {
      if (fromComplex) {
        functionName = cudaq::opt::QIRArrayQubitAllocateArrayWithStateComplex32;
        ptrTy = cudaq::cc::PointerType::get(
            ComplexType::get(rewriter.getF32Type()));
      } else {
        functionName = cudaq::opt::QIRArrayQubitAllocateArrayWithStateFP32;
        ptrTy = cudaq::cc::PointerType::get(rewriter.getF32Type());
      }
    }

    if (functionName.empty())
      return raii.emitOpError("initialize state has an invalid element type.");
    assert(ptrTy && "argument pointer type must be set");

    // Get the size of the qubit register
    Type allocTy = adaptor.getAllocType();
    auto i64Ty = rewriter.getI64Type();

    Value sizeOperand;
    if (!adaptor.getAllocSize()) {
      auto type = dyn_cast<cudaq::quake::VeqType>(allocTy);
      auto constantSize = type ? type.getSize() : 1;
      sizeOperand =
          arith::ConstantIntOp::create(rewriter, loc, constantSize, 64);
    } else {
      sizeOperand = adaptor.getAllocSize();
      auto sizeTy = cast<IntegerType>(sizeOperand.getType());
      if (sizeTy.getWidth() < 64)
        sizeOperand = cudaq::cc::CastOp::create(
            rewriter, loc, i64Ty, sizeOperand, cudaq::cc::CastOpMode::Unsigned);
      else if (sizeTy.getWidth() > 64)
        sizeOperand =
            cudaq::cc::CastOp::create(rewriter, loc, i64Ty, sizeOperand);
    }

    // Call the allocation function
    Value casted = cudaq::cc::CastOp::create(rewriter, loc, ptrTy, ccState);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        raii, arrayTy, functionName, ArrayRef<Value>{sizeOperand, casted});
    return success();
  }
};

struct RelaxSizeOpErase
    : public OpConversionPattern<cudaq::quake::RelaxSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::RelaxSizeOp relax, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(relax, relax.getInputVec());
    return success();
  }
};

template <typename M>
struct SubveqOpRewrite : public OpConversionPattern<cudaq::quake::SubVeqOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::SubVeqOp subveq, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = subveq.getLoc();

    auto lowArg = [&]() -> Value {
      if (!adaptor.getLower())
        return arith::ConstantIntOp::create(rewriter, loc,
                                            adaptor.getRawLower(), 64);
      return adaptor.getLower();
    }();
    auto highArg = [&]() -> Value {
      if (!adaptor.getUpper())
        return arith::ConstantIntOp::create(rewriter, loc,
                                            adaptor.getRawUpper(), 64);
      return adaptor.getUpper();
    }();
    auto i64Ty = rewriter.getI64Type();
    auto extend = [&](Value &v) -> Value {
      if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
        if (intTy.getWidth() < 64)
          return cudaq::cc::CastOp::create(rewriter, loc, i64Ty, v,
                                           cudaq::cc::CastOpMode::Unsigned);
        if (intTy.getWidth() > 64)
          return cudaq::cc::CastOp::create(rewriter, loc, i64Ty, v);
      }
      return v;
    };
    lowArg = extend(lowArg);
    highArg = extend(highArg);
    Value inArr = adaptor.getVeq();
    auto i32Ty = rewriter.getI32Type();
    Value one32 = arith::ConstantIntOp::create(rewriter, loc, i32Ty, 1);
    Value one64 = arith::ConstantIntOp::create(rewriter, loc, i64Ty, 1);
    auto arrayTy = M::getArrayType(rewriter.getContext());
    rewriter.replaceOpWithNewOp<func::CallOp>(
        subveq, arrayTy, cudaq::opt::QIRArraySlice,
        ArrayRef<Value>{inArr, one32, lowArg, one64, highArg});
    return success();
  }
};

struct WrapOpErase : public OpConversionPattern<cudaq::quake::WrapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::WrapOp wrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The wire value and the ref value ought to be the same.
    LLVM_DEBUG({
      if (adaptor.getWireValue() != adaptor.getRefValue())
        llvm::dbgs() << "wire " << adaptor.getWireValue()
                     << " deviates from ref " << adaptor.getRefValue() << '\n';
    });
    rewriter.eraseOp(wrap);
    return success();
  }
};

struct UnwrapOpErase : public OpConversionPattern<cudaq::quake::UnwrapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::UnwrapOp unwrap, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(unwrap, adaptor.getRefValue());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Custom handing of irregular quantum gates.
//===----------------------------------------------------------------------===//

template <typename M>
struct CustomUnitaryOpPattern
    : public QubitHelperConversionPattern<M,
                                          cudaq::quake::CustomUnitarySymbolOp> {
  using Base =
      QubitHelperConversionPattern<M, cudaq::quake::CustomUnitarySymbolOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(cudaq::quake::CustomUnitarySymbolOp unitary,
                  Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!unitary.getParameters().empty())
      return unitary.emitOpError(
          "Parameterized custom operations not yet supported.");

    auto loc = unitary.getLoc();
    auto arrayTy = M::getArrayType(rewriter.getContext());

    if (adaptor.getTargets().empty())
      return unitary.emitOpError("Custom operations must have targets.");

    // Concat all the targets into an array.
    Type firstTy = getInitialType(unitary, adaptor.getParameters().size() +
                                               adaptor.getControls().size());
    auto targetArray = Base::wrapQubitAsArray(
        loc, rewriter, adaptor.getTargets().front(), firstTy);
    SmallVector<Type> origTys;
    for (auto [i, _] : llvm::enumerate(adaptor.getTargets().drop_front()))
      origTys.push_back(
          getInitialType(unitary, adaptor.getParameters().size() +
                                      adaptor.getControls().size() + i + 1));
    for (auto [next, origTy] :
         llvm::zip(adaptor.getTargets().drop_front(), origTys)) {
      auto wrapNext = Base::wrapQubitAsArray(loc, rewriter, next, origTy);
      auto result = func::CallOp::create(
          rewriter, loc, arrayTy, cudaq::opt::QIRArrayConcatArray,
          ArrayRef<Value>{targetArray, wrapNext});
      targetArray = result.getResult(0);
    }

    // Concat all the controls (if any) into an array.
    Value controlArray;
    if (adaptor.getControls().empty()) {
      // Use a nullptr for when 0 control qubits are present.
      Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
      controlArray = cudaq::cc::CastOp::create(rewriter, loc, arrayTy, zero);
    } else {
      Type firstTy = getInitialType(unitary, adaptor.getParameters().size());
      controlArray = Base::wrapQubitAsArray(
          loc, rewriter, adaptor.getControls().front(), firstTy);
      SmallVector<Type> origTys;
      for (auto [i, _] : llvm::enumerate(adaptor.getControls().drop_front()))
        origTys.push_back(
            getInitialType(unitary, adaptor.getParameters().size() + i + 1));
      for (auto [next, origTy] :
           llvm::zip(adaptor.getControls().drop_front(), origTys)) {
        auto wrapNext = Base::wrapQubitAsArray(loc, rewriter, next, origTy);
        auto result = func::CallOp::create(
            rewriter, loc, arrayTy, cudaq::opt::QIRArrayConcatArray,
            ArrayRef<Value>{controlArray, wrapNext});
        controlArray = result.getResult(0);
      }
    }

    // Fetch the unitary matrix generator for this custom operation
    auto generatorSym = unitary.getGenerator();
    StringRef generatorName = generatorSym.getRootReference();
    const auto customOpName = extractCustomNamePart(generatorName);

    // Create a global string for the unitary name.
    auto nameOp = createGlobalCString(unitary, loc, rewriter, customOpName);

    auto complex64Ty = ComplexType::get(rewriter.getF64Type());
    auto complex64PtrTy = cudaq::cc::PointerType::get(complex64Ty);
    auto globalObj = cast<cudaq::cc::GlobalOp>(
        unitary->getParentOfType<ModuleOp>().lookupSymbol(generatorName));
    auto addrOp = cudaq::cc::AddressOfOp::create(
        rewriter, loc, globalObj.getType(), generatorName);
    auto unitaryData =
        cudaq::cc::CastOp::create(rewriter, loc, complex64PtrTy, addrOp);

    StringRef functionName =
        unitary.isAdj() ? cudaq::opt::QIRCustomAdjOp : cudaq::opt::QIRCustomOp;

    rewriter.replaceOpWithNewOp<func::CallOp>(
        unitary, TypeRange{}, functionName,
        ArrayRef<Value>{unitaryData, controlArray, targetArray, nameOp});

    return success();
  }

  // IMPORTANT: this must match the logic to generate global data globalName =
  // f'{nvqppPrefix}{opName}_generator_{numTargets}.rodata'
  std::string extractCustomNamePart(StringRef generatorName) const {
    auto globalName = generatorName.str();
    if (globalName.starts_with(cudaq::runtime::cudaqGenPrefixName)) {
      globalName = globalName.substr(cudaq::runtime::cudaqGenPrefixLength);
      const size_t pos = globalName.find("_generator");
      if (pos != std::string::npos)
        return globalName.substr(0, pos);
    }
    return {};
  }
};

template <typename M>
struct ExpPauliOpPattern
    : public QubitHelperConversionPattern<M, cudaq::quake::ExpPauliOp> {
  using Base = QubitHelperConversionPattern<M, cudaq::quake::ExpPauliOp>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(cudaq::quake::ExpPauliOp pauli, Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = pauli.getLoc();
    // Make sure that apply-control-negations pass was run.
    if (adaptor.getNegatedQubitControls())
      return pauli->emitOpError("negated control qubits not allowed.");
    SmallVector<Value> controls;
    const auto firstControlIndex = adaptor.getParameters().size();
    if (adaptor.getControls().empty()) {
      // do nothing
    } else if (adaptor.getControls().size() > 1 ||
               !isa<cudaq::quake::VeqType>(
                   getInitialType(pauli, firstControlIndex))) {
      // Concat all controls into a single Array.
      Type arrayTy = M::getArrayType(rewriter.getContext());
      auto wrapIfQubit = [&](Value adaptorVal, Type origTy) {
        if (isa<cudaq::quake::VeqType>(origTy))
          return adaptorVal;
        return Base::wrapQubitAsArray(loc, rewriter, adaptorVal, origTy);
      };
      Value firstOperand = adaptor.getControls().front();
      Type firstTy = getInitialType(pauli, firstControlIndex);
      Value resultArray = wrapIfQubit(firstOperand, firstTy);
      SmallVector<Type> origCtrlTys;
      for (auto [i, _] : llvm::enumerate(adaptor.getControls().drop_front()))
        origCtrlTys.push_back(getInitialType(pauli, firstControlIndex + i + 1));
      for (auto [next, origCtrlTy] :
           llvm::zip(adaptor.getControls().drop_front(), origCtrlTys)) {
        Value wrapNext = wrapIfQubit(next, origCtrlTy);
        auto appended = func::CallOp::create(
            rewriter, loc, arrayTy, cudaq::opt::QIRArrayConcatArray,
            ArrayRef<Value>{resultArray, wrapNext});
        resultArray = appended.getResult(0);
      }
      controls.push_back(resultArray);
    } else {
      controls.push_back(adaptor.getControls().front());
    }
    SmallVector<Value> targets;
    const auto firstTargetIndex =
        firstControlIndex + adaptor.getControls().size();
    Type firstTy = getInitialType(pauli, firstTargetIndex);
    if (adaptor.getTargets().size() > 1 ||
        !isa<cudaq::quake::VeqType>(firstTy)) {
      // Concat all targets into a single Array.
      Type arrayTy = M::getArrayType(rewriter.getContext());
      Value firstOperand = adaptor.getTargets().front();
      Value resultArray =
          Base::wrapQubitAsArray(loc, rewriter, firstOperand, firstTy);
      SmallVector<Type> origTargTys;
      for (auto [i, _] : llvm::enumerate(adaptor.getTargets().drop_front()))
        origTargTys.push_back(getInitialType(pauli, firstTargetIndex + i + 1));
      for (auto [next, origTy] :
           llvm::zip(adaptor.getTargets().drop_front(), origTargTys)) {
        Value wrapNext = Base::wrapQubitAsArray(loc, rewriter, next, origTy);
        auto appended = func::CallOp::create(
            rewriter, loc, arrayTy, cudaq::opt::QIRArrayConcatArray,
            ArrayRef<Value>{resultArray, wrapNext});
        resultArray = appended.getResult(0);
      }
      targets.push_back(resultArray);
    } else {
      targets.push_back(adaptor.getTargets().front());
    }

    SmallVector<Value> operands;
    auto qirFunctionName = M::quakeToFuncName(pauli);
    if (pauli.isAdj()) {
      for (auto v : adaptor.getParameters())
        operands.push_back(arith::NegFOp::create(rewriter, loc, v));
    } else {
      operands.append(adaptor.getParameters().begin(),
                      adaptor.getParameters().end());
    }
    operands.append(controls.begin(), controls.end());
    operands.append(targets.begin(), targets.end());

    auto pauliWord = [&]() -> Value {
      if (auto pauliLiteral = pauli.getPauliLiteralAttr()) {
        auto glob =
            createGlobalCString(pauli, loc, rewriter, pauliLiteral.getValue());
        auto ccCast = glob.getDefiningOp<cudaq::cc::CastOp>();
        auto addrOf = ccCast.getValue();
        auto eleTy =
            cast<cudaq::cc::PointerType>(addrOf.getType()).getElementType();
        auto llvmArrTy = cast<LLVM::LLVMArrayType>(eleTy);
        Type arrEleTy = llvmArrTy.getElementType();
        auto arrSize = llvmArrTy.getNumElements();
        auto toTy = cudaq::cc::PointerType::get(cudaq::cc::ArrayType::get(
            rewriter.getContext(), arrEleTy, arrSize));
        return cudaq::cc::CastOp::create(rewriter, loc, toTy, glob);
      }
      return adaptor.getPauli();
    }();

    operands.push_back(pauliWord);

    // First need to check the type of the Pauli word. We expect a pauli_word
    // directly (a.k.a. a span)`{i8*,i64}` or a string literal `ptr<array<i8 x
    // n>>`. If it is a string literal, we need to map it to a pauli word.
    auto i8PtrTy = cudaq::cc::PointerType::get(rewriter.getI8Type());
    Type wordTy;
    if (!pauli.getPauliLiteral())
      wordTy =
          getInitialType(pauli, firstTargetIndex + adaptor.getTargets().size());
    if (wordTy && isa<cudaq::cc::SpanLikeType>(wordTy)) {
      // The attribute tells us we have a pauli word expressed as `{i8*, i64}`.
      // Allocate a stack slot for it and store what we have to that pointer,
      // pass the pointer to NVQIR.
      auto newPauliWord = pauliWord;
      auto newPauliWordTy = newPauliWord.getType();
      Value alloca =
          cudaq::opt::factory::createTemporary(loc, rewriter, newPauliWordTy);
      auto castedVar = cudaq::cc::CastOp::create(
          rewriter, loc, cudaq::cc::PointerType::get(newPauliWordTy), alloca);
      cudaq::cc::StoreOp::create(rewriter, loc, newPauliWord, castedVar);
      auto castedPauli =
          cudaq::cc::CastOp::create(rewriter, loc, i8PtrTy, alloca);
      operands.back() = castedPauli;
      rewriter.replaceOpWithNewOp<func::CallOp>(pauli, TypeRange{},
                                                qirFunctionName, operands);
      return success();
    }
    // Make sure we have the right types to extract the length of the string
    // literal.

    auto ptrTy = [&]() -> cudaq::cc::PointerType {
      if (wordTy)
        return dyn_cast<cudaq::cc::PointerType>(wordTy);
      return dyn_cast<cudaq::cc::PointerType>(pauliWord.getType());
    }();
    auto arrayTy = dyn_cast<cudaq::cc::ArrayType>(ptrTy.getElementType());
    if (!arrayTy)
      return pauli.emitOpError(
          "exp_pauli string literal must have ptr<array<i8 x N> type.");
    if (!arrayTy.getSize())
      return pauli.emitOpError("string literal may not be empty.");

    // We must create the {i8*, i64} struct from the string literal
    SmallVector<Type> structTys{i8PtrTy, rewriter.getI64Type()};
    auto structTy =
        cudaq::cc::StructType::get(rewriter.getContext(), structTys);

    // Allocate the char span struct
    Value alloca =
        cudaq::opt::factory::createTemporary(loc, rewriter, structTy);

    // Convert the number of elements to a constant op.
    auto size =
        arith::ConstantIntOp::create(rewriter, loc, arrayTy.getSize() - 1, 64);

    // Set the string literal data
    auto castedPauli =
        cudaq::cc::CastOp::create(rewriter, loc, i8PtrTy, pauliWord);
    auto strPtr = cudaq::cc::ComputePtrOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(i8PtrTy), alloca,
        ArrayRef<cudaq::cc::ComputePtrArg>{0, 0});
    cudaq::cc::StoreOp::create(rewriter, loc, castedPauli, strPtr);

    // Set the integer length
    auto intPtr = cudaq::cc::ComputePtrOp::create(
        rewriter, loc, cudaq::cc::PointerType::get(rewriter.getI64Type()),
        alloca, ArrayRef<cudaq::cc::ComputePtrArg>{0, 1});
    cudaq::cc::StoreOp::create(rewriter, loc, size, intPtr);

    // Cast to raw opaque pointer
    auto castedStore =
        cudaq::cc::CastOp::create(rewriter, loc, i8PtrTy, alloca);
    operands.back() = castedStore;
    rewriter.replaceOpWithNewOp<func::CallOp>(pauli, TypeRange{},
                                              qirFunctionName, operands);
    return success();
  }
};

template <typename M>
struct MeasurementOpPattern : public OpConversionPattern<cudaq::quake::MzOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::MzOp mz, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = mz.getLoc();
    auto regNameAttr = dyn_cast<StringAttr>(mz.getRegisterNameAttr());
    if (!regNameAttr)
      return mz.emitOpError("mz operation must have a name.");
    if (regNameAttr.getValue().empty())
      return mz.emitOpError("mz name may not be an empty string.");
    SmallVector<Value> args{adaptor.getTargets().begin(),
                            adaptor.getTargets().end()};
    auto functionName = M::getQIRMeasure();

    // Handle-form measurements produce a `!cc.measure_handle` SSA value
    // whose converted type is `i64`. The QIR measurement function still
    // returns `Result*`, so we bridge the call's `Result*` result to the
    // converted `i64` payload via `cc.cast`.
    const bool measOutIsHandle =
        isa<cudaq::cc::MeasureHandleType>(mz.getMeasOut().getType());

    // Are we using the measurement that returns a result?
    if constexpr (M::mzReturnsResultType) {
      // Yes, the measurement results the result, so we can use a
      // straightforward codegen pattern. Use either the mz or the
      // mz_to_register call (with the name as an extra argument) and forward
      // the result of the call as the result.

      if (mz->getAttr(cudaq::opt::MzAssignedNameAttrName)) {
        functionName = cudaq::opt::QIRMeasureToRegister;
        auto cstringGlobal =
            createGlobalCString(mz, loc, rewriter, regNameAttr.getValue());
        args.push_back(cstringGlobal);
      }
      auto resultTy = M::getResultType(rewriter.getContext());
      auto call =
          func::CallOp::create(rewriter, loc, resultTy, functionName, args);
      auto assundry = filterArgs(mz, adaptor.getTargets());
      SmallVector<Value> replaceVals;
      if (measOutIsHandle) {
        auto i64Ty = rewriter.getI64Type();
        replaceVals.push_back(
            cudaq::cc::CastOp::create(rewriter, loc, i64Ty, call.getResult(0)));
      } else {
        replaceVals.append(call.getResults().begin(), call.getResults().end());
      }
      replaceVals.append(assundry.begin(), assundry.end());
      rewriter.replaceOp(mz, replaceVals);
      call->setAttr(cudaq::opt::QIRRegisterNameAttr, regNameAttr);
    } else {
      // No, the measurement doesn't return any result so use a much more
      // convoluted pattern.
      // 1. Cast an integer to the result and append it to the mz call. This
      // will be the token to identify the result. The value will have been
      // attached to the MzOp in preprocessing.
      // 2. Call the mz function.
      // 3. Call the result_record_output to bind the name, which is not folded
      // into the mz call. There is always a name in this case.

      auto resultAttr = mz->getAttr(cudaq::opt::ResultIndexAttrName);
      std::int64_t annInt = cast<IntegerAttr>(resultAttr).getInt();
      Value intVal = arith::ConstantIntOp::create(rewriter, loc, annInt, 64);
      auto resultTy = M::getResultType(rewriter.getContext());
      Value res = cudaq::cc::CastOp::create(rewriter, loc, resultTy, intVal);
      args.push_back(res);
      auto call =
          func::CallOp::create(rewriter, loc, TypeRange{}, functionName, args);
      call->setAttr(cudaq::opt::QIRRegisterNameAttr, regNameAttr);
      // For handle-form callers, materialize the back-cast `Result* -> i64`
      // here so it dominates downstream uses. The `!discriminateToClassical`
      // branch below moves the insertion point to the block terminator for
      // the record-output call, after which a cast would not dominate.

      auto i64Ty = rewriter.getI64Type();
      Value handleRes =
          measOutIsHandle ? cudaq::cc::CastOp::create(rewriter, loc, i64Ty, res)
                          : res;
      auto cstringGlobal =
          createGlobalCString(mz, loc, rewriter, regNameAttr.getValue());
      if constexpr (!M::discriminateToClassical) {
        // These QIR profile variants force all record output calls to appear
        // at the end. In these variants, control-flow isn't allowed in the
        // final LLVM. Therefore, a single basic block is assumed but unchecked
        // here as the verifier will raise an error.
        rewriter.setInsertionPoint(rewriter.getBlock()->getTerminator());
      }
      auto func = mz->getParentOfType<func::FuncOp>();
      if (!func->hasAttr(cudaq::runtime::enableCudaqRun)) {
        auto recOut = func::CallOp::create(rewriter, loc, TypeRange{},
                                           cudaq::opt::QIRRecordOutput,
                                           ArrayRef<Value>{res, cstringGlobal});
        recOut->setAttr(cudaq::opt::ResultIndexAttrName, resultAttr);
        recOut->setAttr(cudaq::opt::QIRRegisterNameAttr, regNameAttr);
      }
      SmallVector<Value> results = {handleRes};
      auto assundry = filterArgs(mz, adaptor.getTargets());
      results.append(assundry.begin(), assundry.end());
      rewriter.replaceOp(mz, results);
    }
    return success();
  }
};

template <typename M>
struct ResetOpPattern : public OpConversionPattern<cudaq::quake::ResetOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::ResetOp reset, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the reset QIR function name
    auto qirFunctionName = M::getQIRReset();

    // Replace the quake op with the new call op.
    if (reset.getResults().empty()) {
      rewriter.replaceOpWithNewOp<func::CallOp>(
          reset, TypeRange{}, qirFunctionName, adaptor.getOperands());
    } else {
      auto loc = reset.getLoc();
      auto results = filterArgs(reset, adaptor.getOperands());
      func::CallOp::create(rewriter, loc, TypeRange{}, qirFunctionName,
                           adaptor.getOperands());
      rewriter.replaceOp(reset, results);
    }
    return success();
  }
};

struct ApplyOpTrap : public OpConversionPattern<cudaq::quake::ApplyOp> {
  using OpConversionPattern::OpConversionPattern;

  // If we see a `quake.apply` operation at this point, something has gone wrong
  // and we were unable to autogenerate the function that we should be calling.
  // So we replace the apply with a trap and the results with poison values.
  LogicalResult
  matchAndRewrite(cudaq::quake::ApplyOp apply, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = apply.getLoc();
    Value zero = arith::ConstantIntOp::create(rewriter, loc, 0, 64);
    func::CallOp::create(rewriter, loc, TypeRange{}, cudaq::opt::QISTrap,
                         ValueRange{zero});
    SmallVector<Value> values;
    for (auto r : apply.getResults()) {
      Value v = cudaq::cc::PoisonOp::create(rewriter, loc, r.getType());
      values.push_back(v);
    }
    rewriter.replaceOp(apply, values);
    return success();
  }
};

struct CallByRefOpRewrite
    : public OpConversionPattern<cudaq::quake::CallByRefOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::quake::CallByRefOp call, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace this with a func.call, but forward the quantum arguments to the
    // uses.
    auto loc = call.getLoc();
    auto fn = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        call, adaptor.getCalleeAttr());

    SmallVector<Value> quantumArgs;
    for (auto [valarg, qirarg] : llvm::zip(call.getArgs(), adaptor.getArgs()))
      if (cudaq::quake::isQuantumValueType(valarg.getType()))
        quantumArgs.push_back(qirarg);

    auto refCall =
        func::CallOp::create(rewriter, loc, fn.getFunctionType().getResults(),
                             adaptor.getCallee(), adaptor.getArgs());

    // Concat the formal results and the quantum arguments to rewrite the uses.
    SmallVector<Value> results{refCall.getResults().begin(),
                               refCall.getResults().end()};
    results.append(quantumArgs.begin(), quantumArgs.end());
    rewriter.replaceOp(call, results);
    return success();
  }
};

struct AnnotateKernelsWithMeasurementStringsPattern
    : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp func,
                                PatternRewriter &rewriter) const override {
    constexpr const char PassthroughAttr[] = "passthrough";
    if (!func->hasAttr(cudaq::kernelAttrName))
      return failure();
    if (!func->hasAttr(PassthroughAttr))
      return failure();
    auto passthru = cast<ArrayAttr>(func->getAttr(PassthroughAttr));
    for (auto a : passthru) {
      if (auto strArrAttr = dyn_cast<ArrayAttr>(a)) {
        auto strAttr = dyn_cast<StringAttr>(strArrAttr[0]);
        if (!strAttr)
          continue;
        if (strAttr.getValue() == cudaq::opt::QIROutputNamesAttrName)
          return failure();
      }
    }

    // Lambda to help recover an integer value (the QIR qubit or result as an
    // integer).
    auto recoverIntValue = [&](Value v) -> std::optional<std::size_t> {
      auto cast = v.getDefiningOp<cudaq::cc::CastOp>();
      if (!cast)
        return {};
      return cudaq::opt::factory::maybeValueOfIntConstant(cast.getValue());
    };

    // If we're here, then `func` is a kernel, it has a passthrough attribute,
    // and the passthrough attribute does *not* have an output names entry.
    //
    // OUTPUT-NAME-MAP: At this point, we will try to heroically generate the
    // output names attribute for the QIR consumer. The content of the
    // attribute is a map from results back to pairs of qubits and names. The
    // map is encoded in a JSON string. The map is appended to the passthrough
    // attribute array.

    std::map<std::size_t, std::size_t> measMap;
    std::map<std::size_t, std::pair<std::size_t, std::string>> nameMap;
    func.walk([&](func::CallOp call) {
      auto calleeName = call.getCallee();
      if (calleeName == cudaq::opt::QIRMeasureBody) {
        auto qubit = recoverIntValue(call.getOperand(0));
        auto meas = recoverIntValue(call.getOperand(1));
        if (qubit && meas)
          measMap[*meas] = *qubit;
      } else if (calleeName == cudaq::opt::QIRRecordOutput) {
        auto resAttr = call->getAttr(cudaq::opt::ResultIndexAttrName);
        std::size_t res = cast<IntegerAttr>(resAttr).getInt();
        auto regNameAttr = call->getAttr(cudaq::opt::QIRRegisterNameAttr);
        std::string regName = cast<StringAttr>(regNameAttr).getValue().str();
        if (measMap.count(res)) {
          std::size_t qubit = measMap[res];
          nameMap[res] = std::pair{qubit, regName};
        }
      }
    });

    // If there were no measurements, then nothing to see here.
    if (nameMap.empty())
      return failure();

    // Append the name map. Use a `const T&` to introduce another layer of
    // brackets here to maintain backwards compatibility.
    const auto &outputNameMapRef = nameMap;
    nlohmann::json outputNames{outputNameMapRef};
    std::string outputNamesStr = outputNames.dump();
    SmallVector<Attribute> funcAttrs(passthru.begin(), passthru.end());
    funcAttrs.push_back(
        rewriter.getStrArrayAttr({cudaq::opt::QIROutputNamesAttrName,
                                  rewriter.getStringAttr(outputNamesStr)}));
    func->setAttr(PassthroughAttr, rewriter.getArrayAttr(funcAttrs));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Generic handling of regular quantum gates.
//===----------------------------------------------------------------------===//

template <typename M, typename OP>
struct QuantumGatePattern : public OpConversionPattern<OP> {
  using Base = OpConversionPattern<OP>;
  using Base::Base;

  LogicalResult
  matchAndRewrite(OP op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto forwardOrEraseOp = [&]() {
      if (op.getResults().empty()) {
        rewriter.eraseOp(op);
      } else {
        auto results = filterArgs(op, adaptor.getOperands());
        rewriter.replaceOp(op, results);
      }
      return success();
    };
    auto qirFunctionName = M::quakeToFuncName(op);

    // Make sure that apply-control-negations pass was run.
    if (adaptor.getNegatedQubitControls())
      return op.emitOpError("negated control qubits not allowed.");

    // Prepare any floating-point parameters.
    auto loc = op.getLoc();
    SmallVector<Value> opParams = adaptor.getParameters();
    if (!opParams.empty()) {
      // If this is adjoint, each parameter is negated.
      if (op.getIsAdj()) {
        for (std::size_t i = 0; i < opParams.size(); ++i)
          opParams[i] = arith::NegFOp::create(rewriter, loc, opParams[i]);
        if constexpr (std::is_same_v<OP, cudaq::quake::U2Op>) {
          std::swap(opParams[0], opParams[1]);
          auto fltTy = cast<FloatType>(opParams[0].getType());
          Value pi = arith::ConstantFloatOp::create(rewriter, loc, fltTy,
                                                    llvm::APFloat{M_PI});
          opParams[0] = arith::SubFOp::create(rewriter, loc, opParams[0], pi);
          opParams[1] = arith::AddFOp::create(rewriter, loc, opParams[1], pi);
        } else if constexpr (std::is_same_v<OP, cudaq::quake::U3Op>) {
          // swap the 2nd and 3rd parameter for correctness
          std::swap(opParams[1], opParams[2]);
        }
      }

      // Each parameter must be converted to double-precision.
      auto f64Ty = rewriter.getF64Type();
      for (std::size_t i = 0; i < opParams.size(); ++i) {
        if (opParams[i].getType().getIntOrFloatBitWidth() != 64)
          opParams[i] =
              cudaq::cc::CastOp::create(rewriter, loc, f64Ty, opParams[i]);
      }
    }

    // If no control qubits or if there is 1 control and it is already a veq,
    // just add a call and forward the target qubits as needed.
    auto numControls = adaptor.getControls().size();
    if (op.getControls().empty() ||
        conformsToIntendedCall(numControls, getInitialType(op, opParams.size()),
                               op, qirFunctionName)) {
      SmallVector<Value> args{opParams.begin(), opParams.end()};
      args.append(adaptor.getControls().begin(), adaptor.getControls().end());
      args.append(adaptor.getTargets().begin(), adaptor.getTargets().end());
      qirFunctionName =
          specializeFunctionName(op, qirFunctionName, numControls);
      func::CallOp::create(rewriter, loc, TypeRange{}, qirFunctionName, args);
      return forwardOrEraseOp();
    }

    // Otherwise, we'll use the generalized invoke helper function. This
    // function takes 4 size_t values, which delimit the different argument
    // types, a pointer to the function to be invoked, and varargs of all the
    // arguments being used. This function's signature is tuned so as to
    // reduce or eliminate the creation of auxiliary temporaries needs to make
    // the call to the helper.
    std::size_t numArrayCtrls = 0;
    SmallVector<Value> opArrCtrls;
    std::size_t numQubitCtrls = 0;
    SmallVector<Value> opQubitCtrls;
    Type i64Ty = rewriter.getI64Type();
    auto ptrNoneTy = M::getLLVMPointerType(rewriter.getContext());

    // Process the controls, sorting them by type. Using the original
    // type recorded by QuakeToQIRAPIPrep, since opaque pointers
    // make Array* and Qubit* indistinguishable on the live operand.
    for (auto [i, val] : llvm::enumerate(adaptor.getControls())) {
      Type origCtrlTy = getInitialType(op, opParams.size() + i);
      if (isaVeqArgument(origCtrlTy)) {
        numArrayCtrls++;
        auto sizeCall = func::CallOp::create(
            rewriter, loc, i64Ty, cudaq::opt::QIRArrayGetSize, ValueRange{val});
        // Arrays are encoded as pairs of arguments: length and Array*
        opArrCtrls.push_back(sizeCall.getResult(0));
        opArrCtrls.push_back(
            cudaq::cc::CastOp::create(rewriter, loc, ptrNoneTy, val));
      } else {
        numQubitCtrls++;
        // Qubits are simply the Qubit**
        opQubitCtrls.emplace_back(
            cudaq::cc::CastOp::create(rewriter, loc, ptrNoneTy, val));
      }
    }

    // Lookup and process the gate operation we're invoking.
    auto module = op->template getParentOfType<ModuleOp>();
    auto symOp = module.lookupSymbol(qirFunctionName);
    if (!symOp)
      return op.emitError("cannot find QIR function");
    auto funOp = dyn_cast<func::FuncOp>(symOp);
    if (!funOp)
      return op.emitError("cannot find " + qirFunctionName);
    FunctionType qirFunctionTy = funOp.getFunctionType();
    auto funCon =
        func::ConstantOp::create(rewriter, loc, qirFunctionTy, qirFunctionName);
    auto funPtr =
        cudaq::cc::FuncToPtrOp::create(rewriter, loc, ptrNoneTy, funCon);

    // Process the target qubits.
    auto numTargets = adaptor.getTargets().size();
    if (numTargets == 0)
      return op.emitOpError("quake op must have at least 1 target.");
    SmallVector<Value> opTargs;
    for (auto t : adaptor.getTargets())
      opTargs.push_back(cudaq::cc::CastOp::create(rewriter, loc, ptrNoneTy, t));

    // Build the declared arguments for the helper call (5 total).
    SmallVector<Value> args;
    args.emplace_back(
        arith::ConstantIntOp::create(rewriter, loc, opParams.size(), 64));
    args.emplace_back(
        arith::ConstantIntOp::create(rewriter, loc, numArrayCtrls, 64));
    args.emplace_back(
        arith::ConstantIntOp::create(rewriter, loc, numQubitCtrls, 64));
    args.emplace_back(
        arith::ConstantIntOp::create(rewriter, loc, numTargets, 64));
    args.emplace_back(funPtr);

    // Finally, append the varargs to the end of the argument list.
    args.append(opParams.begin(), opParams.end());
    args.append(opArrCtrls.begin(), opArrCtrls.end());
    args.append(opQubitCtrls.begin(), opQubitCtrls.end());
    args.append(opTargs.begin(), opTargs.end());

    // Call the generalized version of the gate invocation.
    cudaq::cc::VarargCallOp::create(rewriter, loc, TypeRange{},
                                    cudaq::opt::NVQIRGeneralizedInvokeAny,
                                    args);
    return forwardOrEraseOp();
  }

  static bool isaVeqArgument(Type ty) {
    // TODO: Need a way to identify arrays when using the opaque pointer
    // variant. (In Python, the arguments may already be converted.)
    auto alreadyConverted = [](Type ty) {
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty))
        if (auto strTy = dyn_cast<LLVM::LLVMStructType>(ptrTy.getElementType()))
          return strTy.isIdentified() && strTy.getName() == "Array";
      return false;
    };
    return isa<cudaq::quake::VeqType>(ty) || alreadyConverted(ty);
  }

  static bool conformsToIntendedCall(std::size_t numControls, Type ctrlTy,
                                     OP op, StringRef qirFunctionName) {
    if (numControls != 1)
      return false;
    auto trivialName = specializeFunctionName(op, qirFunctionName, numControls);
    const bool nameChanged = trivialName != qirFunctionName;
    if (nameChanged && !isa<cudaq::quake::VeqType>(ctrlTy))
      return true;
    return !nameChanged && isa<cudaq::quake::VeqType>(ctrlTy);
  }

  static StringRef specializeFunctionName(OP op, StringRef funcName,
                                          std::size_t numCtrls) {
    // Last resort to change the names of particular functions from the
    // general scheme to specialized names under the right conditions.
    if constexpr (std::is_same_v<OP, cudaq::quake::XOp> && M::convertToCNot) {
      if (numCtrls == 1)
        return cudaq::opt::QIRCnot;
    }
    if constexpr (std::is_same_v<OP, cudaq::quake::ZOp> && M::convertToCZ) {
      if (numCtrls == 1)
        return cudaq::opt::QIRCZ;
    }
    return funcName;
  }
};

//===----------------------------------------------------------------------===//
// Handling of functions, calls, and classic memory ops on callables.
//===----------------------------------------------------------------------===//

struct AllocaOpPattern : public OpConversionPattern<cudaq::cc::AllocaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::AllocaOp alloc, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto eleTy = alloc.getElementType();
    auto newEleTy = getTypeConverter()->convertType(eleTy);
    if (eleTy == newEleTy)
      return failure();
    Value ss = alloc.getSeqSize();
    if (ss)
      rewriter.replaceOpWithNewOp<cudaq::cc::AllocaOp>(alloc, newEleTy, ss);
    else
      rewriter.replaceOpWithNewOp<cudaq::cc::AllocaOp>(alloc, newEleTy);
    return success();
  }
};

struct ReturnOpPattern : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Convert the quake types in `func::FuncOp` signatures.
struct FuncSignaturePattern : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp func, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcTy = func.getFunctionType();
    auto newFuncTy =
        cast<FunctionType>(getTypeConverter()->convertType(funcTy));
    if (funcTy != newFuncTy) {
      // Convert the entry block to the new argument types.
      if (funcTy.getNumInputs() && !func.getBody().empty()) {
        // Replace the block argument types.
        for (auto [blockArg, argTy] : llvm::zip(
                 func.getBody().front().getArguments(), newFuncTy.getInputs()))
          blockArg.setType(argTy);
      }
    }
    // Convert any other blocks, as needed.
    for (auto &block : func.getBody().getBlocks()) {
      if (&block == &func.getBody().front())
        continue;
      SmallVector<Type> newTypes;
      for (auto blockArg : block.getArguments())
        newTypes.push_back(getTypeConverter()->convertType(blockArg.getType()));
      for (auto [blockArg, newTy] : llvm::zip(block.getArguments(), newTypes))
        blockArg.setType(newTy);
    }
    // Replace the signature.
    rewriter.modifyOpInPlace(func, [&]() {
      func.setFunctionType(newFuncTy);
      func->setAttr(FuncIsQIRAPI, rewriter.getUnitAttr());
    });
    return success();
  }
};

struct CreateLambdaPattern
    : public OpConversionPattern<cudaq::cc::CreateLambdaOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CreateLambdaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sigTy = cast<cudaq::cc::CallableType>(op.getSignature().getType());
    auto newSigTy =
        cast<cudaq::cc::CallableType>(getTypeConverter()->convertType(sigTy));
    if (sigTy == newSigTy)
      return failure();
    if (sigTy.getSignature().getNumInputs() && !op.getInitRegion().empty()) {
      // Replace the block argument types.
      for (auto [blockArg, argTy] :
           llvm::zip(op.getInitRegion().front().getArguments(),
                     newSigTy.getSignature().getInputs()))
        blockArg.setType(argTy);
    }
    // Replace the signature.
    rewriter.modifyOpInPlace(op,
                             [&]() { op.getSignature().setType(newSigTy); });
    return success();
  }
};

struct CallableFuncPattern
    : public OpConversionPattern<cudaq::cc::CallableFuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CallableFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcTy = op.getFunction().getType();
    auto newFuncTy =
        cast<FunctionType>(getTypeConverter()->convertType(funcTy));
    rewriter.replaceOpWithNewOp<cudaq::cc::CallableFuncOp>(op, newFuncTy,
                                                           op.getCallable());
    return success();
  }
};

template <typename OP>
struct OpInterfacePattern : public OpConversionPattern<OP> {
  using Base = OpConversionPattern<OP>;
  using Base::Base;
  using Base::getTypeConverter;

  LogicalResult
  matchAndRewrite(OP op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newResultTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<OP>(op, newResultTy, adaptor.getOperands(),
                                    op->getAttrs());
    return success();
  }
};

using FuncConstantPattern = OpInterfacePattern<func::ConstantOp>;
using FuncToPtrPattern = OpInterfacePattern<cudaq::cc::FuncToPtrOp>;
using LoadOpPattern = OpInterfacePattern<cudaq::cc::LoadOp>;
using UndefOpPattern = OpInterfacePattern<cudaq::cc::UndefOp>;
using PoisonOpPattern = OpInterfacePattern<cudaq::cc::PoisonOp>;
using CastOpPattern = OpInterfacePattern<cudaq::cc::CastOp>;
using SelectOpPattern = OpInterfacePattern<arith::SelectOp>;
// Pointer-arithmetic and `stdvec` accessors carry pointer/`stdvec` types whose
// element types may need conversion (notably `!cc.measure_handle` -> `i64`).
// Without explicit patterns the type converter inserts unrealized casts on
// their results that the partial conversion cannot resolve.
using ComputePtrOpPattern = OpInterfacePattern<cudaq::cc::ComputePtrOp>;
using StdvecDataOpPattern = OpInterfacePattern<cudaq::cc::StdvecDataOp>;
using StdvecInitOpPattern = OpInterfacePattern<cudaq::cc::StdvecInitOp>;
using StdvecSizeOpPattern = OpInterfacePattern<cudaq::cc::StdvecSizeOp>;

struct InstantiateCallablePattern
    : public OpConversionPattern<cudaq::cc::InstantiateCallableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::InstantiateCallableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sigTy = cast<cudaq::cc::CallableType>(op.getSignature().getType());
    auto newSigTy =
        cast<cudaq::cc::CallableType>(getTypeConverter()->convertType(sigTy));
    rewriter.replaceOpWithNewOp<cudaq::cc::InstantiateCallableOp>(
        op, newSigTy, op.getCallee(), adaptor.getClosureData(),
        op.getNoCaptureAttr());
    return success();
  }
};

struct StoreOpPattern : public OpConversionPattern<cudaq::cc::StoreOp> {
  using Base = OpConversionPattern;
  using Base::Base;
  using Base::getTypeConverter;

  LogicalResult
  matchAndRewrite(cudaq::cc::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cudaq::cc::StoreOp>(
        op, TypeRange{}, adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

template <typename CALLOP>
struct CallOpInterfacePattern : public OpConversionPattern<CALLOP> {
  using Base = OpConversionPattern<CALLOP>;
  using Base::Base;
  using Base::getTypeConverter;

  LogicalResult
  matchAndRewrite(CALLOP op, typename Base::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTys;
    for (auto ty : op.getResultTypes())
      newResultTys.emplace_back(getTypeConverter()->convertType(ty));
    rewriter.replaceOpWithNewOp<CALLOP>(op, newResultTys, adaptor.getOperands(),
                                        op->getAttrs());
    return success();
  }
};

using CallOpPattern = CallOpInterfacePattern<func::CallOp>;
using CallIndirectOpPattern = CallOpInterfacePattern<func::CallIndirectOp>;
using CallVarargOpPattern = CallOpInterfacePattern<cudaq::cc::VarargCallOp>;
using CallNoInlineOpPattern = CallOpInterfacePattern<cudaq::cc::NoInlineCallOp>;
using CallCallableOpPattern = CallOpInterfacePattern<cudaq::cc::CallCallableOp>;
using CallIndirectCallableOpPattern =
    CallOpInterfacePattern<cudaq::cc::CallIndirectCallableOp>;

struct BranchOpPattern : public OpConversionPattern<cf::BranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::BranchOp>(op, adaptor.getDestOperands(),
                                              op.getDest());
    return success();
  }
};

struct CondBranchOpPattern : public OpConversionPattern<cf::CondBranchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), adaptor.getTrueDestOperands(),
        adaptor.getFalseDestOperands(), DenseI32ArrayAttr(), op.getTrueDest(),
        op.getFalseDest());
    return success();
  }
};

struct CallableClosurePattern
    : public OpConversionPattern<cudaq::cc::CallableClosureOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cudaq::cc::CallableClosureOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newTys;
    for (auto ty : op.getResultTypes())
      newTys.push_back(getTypeConverter()->convertType(ty));
    rewriter.replaceOpWithNewOp<cudaq::cc::CallableClosureOp>(
        op, newTys, adaptor.getCallable());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Patterns that are common to all QIR conversions.
//===----------------------------------------------------------------------===//

static void commonClassicalHandlingPatterns(RewritePatternSet &patterns,
                                            TypeConverter &typeConverter,
                                            MLIRContext *ctx) {
  patterns.insert<
      AllocaOpPattern, BranchOpPattern, CallableClosurePattern,
      CallableFuncPattern, CallCallableOpPattern, CallIndirectCallableOpPattern,
      CallIndirectOpPattern, CallOpPattern, CallNoInlineOpPattern,
      CallVarargOpPattern, CastOpPattern, CondBranchOpPattern,
      ComputePtrOpPattern, CreateLambdaPattern, FuncConstantPattern,
      FuncSignaturePattern, FuncToPtrPattern, InstantiateCallablePattern,
      LoadOpPattern, PoisonOpPattern, SelectOpPattern, StdvecDataOpPattern,
      StdvecInitOpPattern, StdvecSizeOpPattern, StoreOpPattern, UndefOpPattern>(
      typeConverter, ctx);
}

static void commonQuakeHandlingPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter,
                                        MLIRContext *ctx) {
  patterns.insert<ApplyOpTrap, CallByRefOpRewrite, GetMemberOpRewrite,
                  MakeStruqOpRewrite, ReturnOpPattern, RelaxSizeOpErase,
                  UnwrapOpErase, VeqSizeOpRewrite, WrapOpErase>(typeConverter,
                                                                ctx);
}

//===----------------------------------------------------------------------===//
// Modifier classes
//===----------------------------------------------------------------------===//

template <bool opaquePtr>
Type GetLLVMPointerType(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(ctx);
}

/// The modifier class for the "full QIR" API.
template <bool opaquePtr>
struct FullQIR {
  using Self = FullQIR;

  template <typename QuakeOp>
  static std::string quakeToFuncName(QuakeOp op) {
    auto [prefix, _] = generateGateFunctionName(op);
    return prefix;
  }

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    auto *ctx = patterns.getContext();
    patterns.insert<
        /* Rewrites for qubit management and aggregation. */
        AllocaOpToCallsRewrite<Self>, BundleCableOpRewrite<Self>,
        ConcatOpRewrite<Self>, DeallocOpRewrite, DiscriminateOpRewrite,
        ExtractRefOpRewrite<Self>, NullCableOpToCallsRewrite<Self>,
        NullWireOpToCallsRewrite<Self>, QmemRAIIOpRewrite<Self>, SinkOpRewrite,
        SplitCableOpRewrite<Self>, SubveqOpRewrite<Self>,

        /* Irregular quantum operators. */
        CustomUnitaryOpPattern<Self>, ExpPauliOpPattern<Self>,
        MeasurementOpPattern<Self>, ResetOpPattern<Self>,
        ApplyNoiseOpRewrite<Self>,

        /* Regular quantum operators. */
        QuantumGatePattern<Self, cudaq::quake::HOp>,
        QuantumGatePattern<Self, cudaq::quake::PhasedRxOp>,
        QuantumGatePattern<Self, cudaq::quake::R1Op>,
        QuantumGatePattern<Self, cudaq::quake::RxOp>,
        QuantumGatePattern<Self, cudaq::quake::RyOp>,
        QuantumGatePattern<Self, cudaq::quake::RzOp>,
        QuantumGatePattern<Self, cudaq::quake::SOp>,
        QuantumGatePattern<Self, cudaq::quake::SwapOp>,
        QuantumGatePattern<Self, cudaq::quake::TOp>,
        QuantumGatePattern<Self, cudaq::quake::U2Op>,
        QuantumGatePattern<Self, cudaq::quake::U3Op>,
        QuantumGatePattern<Self, cudaq::quake::XOp>,
        QuantumGatePattern<Self, cudaq::quake::YOp>,
        QuantumGatePattern<Self, cudaq::quake::ZOp>>(typeConverter, ctx);
    commonQuakeHandlingPatterns(patterns, typeConverter, ctx);
    commonClassicalHandlingPatterns(patterns, typeConverter, ctx);
  }

  static StringRef getQIRMeasure() { return cudaq::opt::QIRMeasure; }
  static StringRef getQIRReset() { return cudaq::opt::QIRReset; }

  static constexpr bool mzReturnsResultType = true;
  static constexpr bool convertToCNot = false;
  static constexpr bool convertToCZ = false;

  static Type getQubitType(MLIRContext *ctx) {
    return cudaq::cg::getQubitType(ctx, opaquePtr);
  }
  static Type getArrayType(MLIRContext *ctx) {
    return cudaq::cg::getArrayType(ctx, opaquePtr);
  }
  static Type getResultType(MLIRContext *ctx) {
    return cudaq::cg::getResultType(ctx, opaquePtr);
  }
  static Type getCharPointerType(MLIRContext *ctx) {
    return cudaq::cg::getCharPointerType(ctx, opaquePtr);
  }
  static Type getLLVMPointerType(MLIRContext *ctx) {
    return GetLLVMPointerType<opaquePtr>(ctx);
  }
};

/// The base modifier class for the "profile QIR" APIs.
template <bool opaquePtr>
struct AnyProfileQIR {
  using Self = AnyProfileQIR;

  template <typename QuakeOp>
  static std::string quakeToFuncName(QuakeOp op) {
    auto [prefix, isBarePrefix] = generateGateFunctionName(op);
    return isBarePrefix ? prefix + "__body" : prefix;
  }

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    auto *ctx = patterns.getContext();
    patterns.insert<
        /* Rewrites for qubit management and aggregation. */
        AllocaOpToIntRewrite<Self>, BundleCableOpRewrite<Self>,
        ConcatOpRewrite<Self>, DeallocOpErase, ExtractRefOpRewrite<Self>,
        NullCableOpToIntRewrite<Self>, NullWireOpToIntRewrite<Self>,
        QmemRAIIOpRewrite<Self>, SinkOpErase, SplitCableOpRewrite<Self>,
        SubveqOpRewrite<Self>,

        /* Irregular quantum operators. */
        CustomUnitaryOpPattern<Self>, ExpPauliOpPattern<Self>,
        ResetOpPattern<Self>, ApplyNoiseOpRewrite<Self>,

        /* Regular quantum operators. */
        QuantumGatePattern<Self, cudaq::quake::HOp>,
        QuantumGatePattern<Self, cudaq::quake::PhasedRxOp>,
        QuantumGatePattern<Self, cudaq::quake::R1Op>,
        QuantumGatePattern<Self, cudaq::quake::RxOp>,
        QuantumGatePattern<Self, cudaq::quake::RyOp>,
        QuantumGatePattern<Self, cudaq::quake::RzOp>,
        QuantumGatePattern<Self, cudaq::quake::SOp>,
        QuantumGatePattern<Self, cudaq::quake::SwapOp>,
        QuantumGatePattern<Self, cudaq::quake::TOp>,
        QuantumGatePattern<Self, cudaq::quake::U2Op>,
        QuantumGatePattern<Self, cudaq::quake::U3Op>,
        QuantumGatePattern<Self, cudaq::quake::XOp>,
        QuantumGatePattern<Self, cudaq::quake::YOp>,
        QuantumGatePattern<Self, cudaq::quake::ZOp>>(typeConverter, ctx);
    commonQuakeHandlingPatterns(patterns, typeConverter, ctx);
    commonClassicalHandlingPatterns(patterns, typeConverter, ctx);
  }

  static StringRef getQIRMeasure() { return cudaq::opt::QIRMeasureBody; }
  static StringRef getQIRReset() { return cudaq::opt::QIRResetBody; }

  static constexpr bool mzReturnsResultType = false;
  static constexpr bool convertToCNot = true;
  static constexpr bool convertToCZ = true;

  static Type getQubitType(MLIRContext *ctx) {
    return cudaq::cg::getQubitType(ctx, opaquePtr);
  }
  static Type getArrayType(MLIRContext *ctx) {
    return cudaq::cg::getArrayType(ctx, opaquePtr);
  }
  static Type getResultType(MLIRContext *ctx) {
    return cudaq::cg::getResultType(ctx, opaquePtr);
  }
  static Type getCharPointerType(MLIRContext *ctx) {
    return cudaq::cg::getCharPointerType(ctx, opaquePtr);
  }
  static Type getLLVMPointerType(MLIRContext *ctx) {
    return GetLLVMPointerType<opaquePtr>(ctx);
  }
};

/// The QIR base profile modifier class.
template <bool opaquePtr, QirVersion version>
struct BaseProfileQIR : public AnyProfileQIR<opaquePtr> {
  using Self = BaseProfileQIR;
  using Base = AnyProfileQIR<opaquePtr>;

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    Base::populateRewritePatterns(patterns, typeConverter);
    patterns
        .insert<DiscriminateOpToCallRewrite<Self>, MeasurementOpPattern<Self>>(
            typeConverter, patterns.getContext());
  }

  static constexpr bool discriminateToClassical = false;
  static constexpr QirVersion qirVersion = version;
};

/// The QIR adaptive profile modifier class.
template <bool opaquePtr, QirVersion version>
struct AdaptiveProfileQIR : public AnyProfileQIR<opaquePtr> {
  using Self = AdaptiveProfileQIR;
  using Base = AnyProfileQIR<opaquePtr>;

  static void populateRewritePatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
    Base::populateRewritePatterns(patterns, typeConverter);
    patterns
        .insert<DiscriminateOpToCallRewrite<Self>, MeasurementOpPattern<Self>>(
            typeConverter, patterns.getContext());
  }

  static constexpr bool discriminateToClassical = true;
  static constexpr QirVersion qirVersion = version;
};

//===----------------------------------------------------------------------===//
// Quake conversion to the QIR API driver pass.
//
// This is done in 3 phased: preparation, conversion, and finalization.
//===----------------------------------------------------------------------===//

/// Conversion of quake IR to QIR calls for the intended API.
struct QuakeToQIRAPIPass
    : public cudaq::opt::impl::QuakeToQIRAPIBase<QuakeToQIRAPIPass> {
  using QuakeToQIRAPIBase::QuakeToQIRAPIBase;

  template <typename A>
  void processOperation(QIRAPITypeConverter &typeConverter) {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before QIR API conversion:\n" << *op << '\n');
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    A::populateRewritePatterns(patterns, typeConverter);
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, cudaq::cc::CCDialect,
                           cf::ControlFlowDialect, func::FuncDialect,
                           LLVM::LLVMDialect>();
    target.addIllegalDialect<cudaq::quake::QuakeDialect,
                             cudaq::codegen::CodeGenDialect>();
    target.addLegalOp<cudaq::codegen::MaterializeConstantArrayOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp fn) {
      return !needsTypeConversion(fn.getFunctionType()) &&
             (!fn->hasAttr(cudaq::kernelAttrName) || fn->hasAttr(FuncIsQIRAPI));
    });
    target.addDynamicallyLegalOp<func::ConstantOp>([&](func::ConstantOp op) {
      return !needsTypeConversion(op.getResult().getType());
    });
    target.addDynamicallyLegalOp<cudaq::cc::UndefOp, cudaq::cc::PoisonOp>(
        [&](Operation *op) {
          return !needsTypeConversion(op->getResult(0).getType());
        });
    target.addDynamicallyLegalOp<cudaq::cc::CallableFuncOp>(
        [&](cudaq::cc::CallableFuncOp op) {
          return !needsTypeConversion(op.getFunction().getType());
        });
    target.addDynamicallyLegalOp<cudaq::cc::CreateLambdaOp>(
        [&](cudaq::cc::CreateLambdaOp op) {
          return !needsTypeConversion(op.getSignature().getType());
        });
    target.addDynamicallyLegalOp<cudaq::cc::InstantiateCallableOp>(
        [&](cudaq::cc::InstantiateCallableOp op) {
          for (auto d : op.getClosureData())
            if (needsTypeConversion(d.getType()))
              return false;
          return !needsTypeConversion(op.getSignature().getType());
        });
    target.addDynamicallyLegalOp<cudaq::cc::CallableClosureOp>(
        [&](cudaq::cc::CallableClosureOp op) {
          for (auto ty : op.getResultTypes())
            if (needsTypeConversion(ty))
              return false;
          return !needsTypeConversion(op.getCallable().getType());
        });
    target.addDynamicallyLegalOp<cudaq::cc::AllocaOp>(
        [&](cudaq::cc::AllocaOp op) {
          return !needsTypeConversion(op.getElementType());
        });
    target.addDynamicallyLegalOp<arith::SelectOp>([&](arith::SelectOp op) {
      return !needsTypeConversion(op.getResult().getType());
    });
    target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
        [&](Operation *op) {
          for (auto opnd : op->getOperands())
            if (needsTypeConversion(opnd.getType()))
              return false;
          return true;
        });
    target.addDynamicallyLegalOp<
        func::CallOp, func::CallIndirectOp, func::ReturnOp,
        cudaq::cc::NoInlineCallOp, cudaq::cc::VarargCallOp,
        cudaq::cc::CallCallableOp, cudaq::cc::CallIndirectCallableOp,
        cudaq::cc::CastOp, cudaq::cc::ComputePtrOp, cudaq::cc::FuncToPtrOp,
        cudaq::cc::StoreOp, cudaq::cc::LoadOp, cudaq::cc::StdvecDataOp,
        cudaq::cc::StdvecInitOp, cudaq::cc::StdvecSizeOp>([&](Operation *op) {
      for (auto opnd : op->getOperands())
        if (needsTypeConversion(opnd.getType()))
          return false;
      for (auto res : op->getResults())
        if (needsTypeConversion(res.getType()))
          return false;
      return true;
    });
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After QIR API conversion:\n" << *op << '\n');
  }

  // Returns true iff `ty` (or some type nested inside it) requires conversion
  // by `QIRAPITypeConverter`. The recursion descends through CC container
  // types that the converter rewrites (`cc.ptr`, `cc.callable`,
  // `cc.indirect_callable`, function types, `cc.array`, `cc.stdvec`) and the
  // leaf check covers Quake types and `!cc.measure_handle` (the IR alias of
  // `cudaq::measure_handle`, lowered to `i64`).
  static bool needsTypeConversion(Type ty) {
    if (auto pty = dyn_cast<cudaq::cc::PointerType>(ty))
      return needsTypeConversion(pty.getElementType());
    if (auto cty = dyn_cast<cudaq::cc::CallableType>(ty))
      return needsTypeConversion(cty.getSignature());
    if (auto cty = dyn_cast<cudaq::cc::IndirectCallableType>(ty))
      return needsTypeConversion(cty.getSignature());
    if (auto aty = dyn_cast<cudaq::cc::ArrayType>(ty))
      return needsTypeConversion(aty.getElementType());
    if (auto sty = dyn_cast<cudaq::cc::StdvecType>(ty))
      return needsTypeConversion(sty.getElementType());
    if (auto fty = dyn_cast<FunctionType>(ty)) {
      for (auto t : fty.getInputs())
        if (needsTypeConversion(t))
          return true;
      for (auto t : fty.getResults())
        if (needsTypeConversion(t))
          return true;
      return false;
    }
    return cudaq::quake::isQuakeType(ty) ||
           isa<cudaq::cc::MeasureHandleType>(ty);
  }

  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "Begin converting to QIR\n");
    QIRAPITypeConverter typeConverter(opaquePtr);
    SmallVector<StringRef> apiField;
    splitTransportTriple(apiField, api);
    if (apiField[0] == "full") {
      if (opaquePtr)
        processOperation<FullQIR</*opaquePtr=*/true>>(typeConverter);
      else
        processOperation<FullQIR</*opaquePtr=*/false>>(typeConverter);
    } else if (apiField[0] == "base-profile") {
      if (apiField.size() > 1 && apiField[1] == "1.0") {
        if (opaquePtr)
          processOperation<
              BaseProfileQIR</*opaquePtr=*/true, QirVersion::version_1_0>>(
              typeConverter);
        else
          processOperation<
              BaseProfileQIR</*opaquePtr=*/false, QirVersion::version_1_0>>(
              typeConverter);
      } else {
        if (opaquePtr)
          processOperation<
              BaseProfileQIR</*opaquePtr=*/true, QirVersion::version_0_1>>(
              typeConverter);
        else
          processOperation<
              BaseProfileQIR</*opaquePtr=*/false, QirVersion::version_0_1>>(
              typeConverter);
      }
    } else if (apiField[0] == "adaptive-profile") {
      if (apiField.size() > 1 && apiField[1] == "1.0") {
        if (opaquePtr)
          processOperation<
              AdaptiveProfileQIR</*opaquePtr=*/true, QirVersion::version_1_0>>(
              typeConverter);
        else
          processOperation<
              AdaptiveProfileQIR</*opaquePtr=*/false, QirVersion::version_1_0>>(
              typeConverter);
      } else {
        if (opaquePtr)
          processOperation<
              AdaptiveProfileQIR</*opaquePtr=*/true, QirVersion::version_0_1>>(
              typeConverter);
        else
          processOperation<
              AdaptiveProfileQIR</*opaquePtr=*/false, QirVersion::version_0_1>>(
              typeConverter);
      }
    } else {
      getOperation()->emitOpError("The currently supported APIs are: 'full', "
                                  "'base-profile', 'adaptive-profile'.");
      signalPassFailure();
    }
  }
};

struct QuakeToQIRAPIPrepPass
    : public cudaq::opt::impl::QuakeToQIRAPIPrepBase<QuakeToQIRAPIPrepPass> {
  using QuakeToQIRAPIPrepBase::QuakeToQIRAPIPrepBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<StringRef> apiFields;
    splitTransportTriple(apiFields, api);

    if (apiFields.empty()) {
      emitError(module.getLoc(), "api may not be empty");
      signalPassFailure();
      return;
    }
    // Extract the QIR version.
    StringRef qirVersion = "0.1";
    if (apiFields.size() > 1)
      qirVersion = apiFields[1];

    {
      auto *ctx = &getContext();
      RewritePatternSet patterns(ctx);
      QIRAPITypeConverter typeConverter(opaquePtr);
      cudaq::opt::populateQuakeToCCPrepPatterns(patterns);
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    auto irBuilder = cudaq::IRBuilder::atBlockEnd(module.getBody());

    // Get the type aliases to use to dynamically configure the prototypes.
    StringRef typeAliases;
    if (opaquePtr) {
      LLVM_DEBUG(llvm::dbgs() << "Using opaque pointers\n");
      typeAliases = irBuilder.getIntrinsicText("qir_opaque_pointer");
    } else {
      LLVM_DEBUG(llvm::dbgs() << "Using pointers to opaque structs\n");
      typeAliases = irBuilder.getIntrinsicText("qir_opaque_struct");
    }

    bool usingFullQIR = apiFields[0] == "full";
    if (usingFullQIR) {
      if (failed(irBuilder.loadIntrinsicWithAliases(module, "qir_full",
                                                    typeAliases))) {
        module.emitError("could not load full QIR declarations.");
        signalPassFailure();
        return;
      }

    } else {
      if (failed(irBuilder.loadIntrinsicWithAliases(
              module, "qir_common_profile", typeAliases))) {
        module.emitError("could not load QIR profile declarations.");
        signalPassFailure();
        return;
      }
    }

    if (usingFullQIR) {
      OpBuilder builder(module);
      module.walk([&](func::FuncOp func) {
        int counter = 0;
        func.walk([&](cudaq::quake::MzOp mz) {
          guaranteeMzIsLabeled(mz, counter, builder);
        });
      });
    } else {
      // If the API is one of the profile variants, we must perform allocation
      // and measurement analysis and stick attributes on the Ops as needed.
      OpBuilder builder(module);
      module.walk([&](func::FuncOp func) {
        std::size_t totalQubits = 0;
        std::size_t totalResults = 0;
        int counter = 0;

        // A map to keep track of wire set usage.
        DenseMap<StringRef, DenseSet<std::size_t>> borrowSets;

        // Recursive walk in func.
        func.walk([&](Operation *op) {
          // Annotate all qubit allocations with the starting qubit index
          // value. This ought to handle both reference and value semantics. If
          // the value semantics is using wire sets, no (redundant) annotation
          // is needed.
          if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op)) {
            auto allocTy = alloc.getType();
            if (isa<cudaq::quake::RefType>(allocTy)) {
              alloc->setAttr(cudaq::opt::StartingOffsetAttrName,
                             builder.getI64IntegerAttr(totalQubits++));
              return;
            }
            if (!isa<cudaq::quake::VeqType>(allocTy)) {
              alloc.emitOpError("must be veq type.");
              return;
            }
            auto veqTy = cast<cudaq::quake::VeqType>(allocTy);
            if (!veqTy.hasSpecifiedSize()) {
              alloc.emitOpError("must have a constant size.");
              return;
            }
            alloc->setAttr(cudaq::opt::StartingOffsetAttrName,
                           builder.getI64IntegerAttr(totalQubits));
            totalQubits += veqTy.getSize();
            return;
          }
          if (auto nw = dyn_cast<cudaq::quake::NullWireOp>(op)) {
            nw->setAttr(cudaq::opt::StartingOffsetAttrName,
                        builder.getI64IntegerAttr(totalQubits++));
            return;
          }
          if (auto nc = dyn_cast<cudaq::quake::NullCableOp>(op)) {
            cudaq::quake::CableType cableTy = nc.getType();
            nc->setAttr(cudaq::opt::StartingOffsetAttrName,
                        builder.getI64IntegerAttr(totalQubits));
            totalQubits += cableTy.getSize();
            return;
          }
          if (auto bw = dyn_cast<cudaq::quake::BorrowWireOp>(op)) {
            [[maybe_unused]] StringRef name = bw.getSetName();
            [[maybe_unused]] std::int32_t wire = bw.getIdentity();
            bw.emitOpError("not implemented.");
            return;
          }

          // For each mz, we want to assign it a result index.
          if (auto mz = dyn_cast<cudaq::quake::MzOp>(op)) {
            // Verify there is exactly one qubit being measured.
            if (mz.getTargets().empty() ||
                std::distance(mz.getTargets().begin(), mz.getTargets().end()) !=
                    1) {
              mz.emitOpError("must measure exactly one qubit.");
              return;
            }
            mz->setAttr(cudaq::opt::ResultIndexAttrName,
                        builder.getI64IntegerAttr(totalResults++));
            guaranteeMzIsLabeled(mz, counter, builder);
          }
        });

        // If the API is one of the profile variants, the QIR consumer expects
        // some bonus information by way of attributes. Add most of them here.
        // (See also OUTPUT-NAME-MAP.)
        SmallVector<Attribute> funcAttrs;
        if (func->hasAttr(cudaq::kernelAttrName)) {
          if (func->getAttr(cudaq::entryPointAttrName))
            funcAttrs.push_back(
                builder.getStringAttr(cudaq::opt::QIREntryPointAttrName));
          if (apiFields[0] == "base-profile") {
            funcAttrs.push_back(builder.getStrArrayAttr(
                {cudaq::opt::QIRProfilesAttrName, "base_profile"}));
            funcAttrs.push_back(builder.getStrArrayAttr(
                {cudaq::opt::QIROutputLabelingSchemaAttrName, "schema_id"}));
          } else if (apiFields[0] == "adaptive-profile") {
            funcAttrs.push_back(builder.getStrArrayAttr(
                {cudaq::opt::QIRProfilesAttrName, "adaptive_profile"}));
            funcAttrs.push_back(builder.getStrArrayAttr(
                {cudaq::opt::QIROutputLabelingSchemaAttrName, "schema_id"}));
          }
          if (totalQubits)
            funcAttrs.push_back(builder.getStrArrayAttr(
                {getRequiredQubitsAttrName(qirVersion),
                 builder.getStringAttr(std::to_string(totalQubits))}));
          if (totalResults)
            funcAttrs.push_back(builder.getStrArrayAttr(
                {getRequiredResultsAttrName(qirVersion),
                 builder.getStringAttr(std::to_string(totalResults))}));
        }
        if (!funcAttrs.empty())
          func->setAttr("passthrough", builder.getArrayAttr(funcAttrs));
      });
    }

    auto *ctx = module.getContext();
    module.walk([&](Operation *op) {
      if (std::all_of(
              op->getResultTypes().begin(), op->getResultTypes().end(),
              [&](Type ty) { return !cudaq::quake::isQuantumType(ty); }) &&
          std::all_of(
              op->getOperandTypes().begin(), op->getOperandTypes().end(),
              [&](Type ty) { return !cudaq::quake::isQuantumType(ty); }))
        return;
      SmallVector<Attribute> typeAttrs;
      typeAttrs.reserve(op->getOperands().size());
      for (Type ty : op->getOperandTypes())
        typeAttrs.push_back(TypeAttr::get(ty));
      auto operandTypes = ArrayAttr::get(ctx, typeAttrs);
      op->setAttr(InitialArgTypesAttrName, operandTypes);
    });
  }

  static StringRef getRequiredQubitsAttrName(StringRef version) {
    if (version == "1.0")
      return cudaq::opt::qir1_0::RequiredQubitsAttrName;
    return cudaq::opt::qir0_1::RequiredQubitsAttrName;
  }

  static StringRef getRequiredResultsAttrName(StringRef version) {
    if (version == "1.0")
      return cudaq::opt::qir1_0::RequiredResultsAttrName;
    return cudaq::opt::qir0_1::RequiredResultsAttrName;
  }

  void guaranteeMzIsLabeled(cudaq::quake::MzOp mz, int &counter,
                            OpBuilder &builder) {
    if (mz.getRegisterNameAttr()) {
      mz->setAttr(cudaq::opt::MzAssignedNameAttrName, builder.getUnitAttr());
      return;
    }
    // Manufacture a bogus name on demand here.
    std::string manuName = std::to_string(counter++);
    constexpr std::size_t padSize = 5;
    manuName =
        std::string(padSize - std::min(padSize, manuName.length()), '0') +
        manuName;
    mz.setRegisterName("r" + manuName);
  }
};

struct QuakeToQIRAPIFinalPass
    : public cudaq::opt::impl::QuakeToQIRAPIFinalBase<QuakeToQIRAPIFinalPass> {
  using QuakeToQIRAPIFinalBase::QuakeToQIRAPIFinalBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp module = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<MaterializeConstantArrayOpRewrite,
                    AnnotateKernelsWithMeasurementStringsPattern>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void cudaq::opt::addConvertToQIRAPIPipeline(OpPassManager &pm, StringRef api,
                                            bool opaquePtr) {
  QuakeToQIRAPIPrepOptions prepApiOpt{.api = api.str(), .opaquePtr = opaquePtr};
  pm.addPass(cudaq::opt::createQuakeToQIRAPIPrep(prepApiOpt));
  pm.addPass(cudaq::opt::createLowerToCG());
  QuakeToQIRAPIOptions apiOpt{.api = api.str(), .opaquePtr = opaquePtr};
  pm.addPass(cudaq::opt::createQuakeToQIRAPI(apiOpt));
  pm.addPass(cudaq::opt::createQirInsertArrayRecord());
  pm.addPass(createCanonicalizerPass());
  QuakeToQIRAPIFinalOptions finalApiOpt{.api = api.str()};
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  pm.addPass(cudaq::opt::createQuakeToQIRAPIFinal(finalApiOpt));
  pm.addPass(createCanonicalizerPass());
}

namespace {
struct QIRAPIPipelineOptions
    : public PassPipelineOptions<QIRAPIPipelineOptions> {
  PassOptions::Option<std::string> api{
      *this, "api",
      llvm::cl::desc("select the profile to convert to [full, base-profile, "
                     "adaptive-profile]"),
      llvm::cl::init("full")};
  PassOptions::Option<bool> opaquePtr{*this, "opaque-pointer",
                                      llvm::cl::desc("use opaque pointers"),
                                      llvm::cl::init(false)};
};
} // namespace

void cudaq::opt::registerToQIRAPIPipeline() {
  PassPipelineRegistration<QIRAPIPipelineOptions>(
      "convert-to-qir-api", "Convert quake to one of the QIR APIs.",
      [](OpPassManager &pm, const QIRAPIPipelineOptions &opt) {
        addConvertToQIRAPIPipeline(pm, opt.api, opt.opaquePtr);
      });
}
