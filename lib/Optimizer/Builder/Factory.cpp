/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace cudaq::opt {

cc::StateType factory::getCudaqStateType(MLIRContext *ctx) {
  return cc::StateType::get(ctx);
}

bool factory::isX86_64(ModuleOp module) {
  auto ta = module->getAttr(targetTripleAttrName);
  llvm::Triple tr(cast<StringAttr>(ta).str());
  return tr.getArch() == llvm::Triple::x86_64;
}

bool factory::isAArch64(ModuleOp module) {
  auto ta = module->getAttr(targetTripleAttrName);
  llvm::Triple tr(cast<StringAttr>(ta).str());
  return tr.getArch() == llvm::Triple::aarch64;
}

template <bool isOutput>
static Type genBufferType(Type ty) {
  auto *ctx = ty.getContext();
  if (isa<cudaq::cc::CallableType>(ty))
    return cudaq::cc::PointerType::get(ctx);
  if (auto vecTy = dyn_cast<cudaq::cc::SpanLikeType>(ty)) {
    auto i64Ty = IntegerType::get(ctx, 64);
    if (isOutput) {
      SmallVector<Type> mems = {
          cudaq::cc::PointerType::get(vecTy.getElementType()), i64Ty};
      return cudaq::cc::StructType::get(ctx, mems);
    }
    return i64Ty;
  }
  if (auto strTy = dyn_cast<cudaq::cc::StructType>(ty)) {
    if (strTy.isEmpty())
      return IntegerType::get(ctx, 64);
    SmallVector<Type> mems;
    for (auto memTy : strTy.getMembers())
      mems.push_back(genBufferType<isOutput>(memTy));
    return cudaq::cc::StructType::get(ctx, mems);
  }
  if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(ty)) {
    assert(!cudaq::cc::isDynamicType(ty) && "must be a type of static extent");
    return ty;
  }
  return ty;
}

Type factory::genArgumentBufferType(Type ty) {
  return genBufferType</*isOutput=*/false>(ty);
}

cudaq::cc::StructType factory::buildInvokeStructType(FunctionType funcTy) {
  auto *ctx = funcTy.getContext();
  SmallVector<Type> eleTys;
  for (auto inTy : funcTy.getInputs())
    eleTys.push_back(genBufferType</*isOutput=*/false>(inTy));
  for (auto outTy : funcTy.getResults())
    eleTys.push_back(genBufferType</*isOutput=*/true>(outTy));
  return cudaq::cc::StructType::get(ctx, eleTys /*packed=false*/);
}

Value factory::packIsArrayAndLengthArray(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         ModuleOp parentModule,
                                         Value numOperands,
                                         ValueRange operands) {
  // Create an integer array where the kth element is N if the kth
  // control operand is a veq<N>, and 0 otherwise.
  auto i64Type = rewriter.getI64Type();
  auto context = rewriter.getContext();
  Value isArrayAndLengthArr = rewriter.create<LLVM::AllocaOp>(
      loc, LLVM::LLVMPointerType::get(i64Type), numOperands);
  auto intPtrTy = LLVM::LLVMPointerType::get(i64Type);
  Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
  auto getSizeSymbolRef = opt::factory::createLLVMFunctionSymbol(
      opt::QIRArrayGetSize, i64Type, {opt::getArrayType(context)},
      parentModule);
  for (auto iter : llvm::enumerate(operands)) {
    auto operand = iter.value();
    auto i = iter.index();
    Value idx = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
    Value ptr = rewriter.create<LLVM::GEPOp>(loc, intPtrTy, isArrayAndLengthArr,
                                             ValueRange{idx});
    Value element;
    if (operand.getType() == opt::getQubitType(context))
      element = zero;
    else
      // get array size with the runtime function
      element = rewriter
                    .create<LLVM::CallOp>(loc, rewriter.getI64Type(),
                                          getSizeSymbolRef, ValueRange{operand})
                    .getResult();

    rewriter.create<LLVM::StoreOp>(loc, element, ptr);
  }
  return isArrayAndLengthArr;
}

FlatSymbolRefAttr factory::createLLVMFunctionSymbol(StringRef name,
                                                    Type retType,
                                                    ArrayRef<Type> inArgTypes,
                                                    ModuleOp module,
                                                    bool isVar) {
  OpBuilder rewriter(module);
  auto *context = module.getContext();
  FlatSymbolRefAttr symbolRef;
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name)) {
    symbolRef = FlatSymbolRefAttr::get(context, name);
  } else {
    // Create the LLVM FunctionType
    auto fType = LLVM::LLVMFunctionType::get(retType, inArgTypes, isVar);

    // Insert the function since it hasn't been seen yet
    auto insPt = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module->getLoc(), name, fType);
    symbolRef = SymbolRefAttr::get(context, name);
    rewriter.restoreInsertionPoint(insPt);
  }
  return symbolRef;
}

func::FuncOp factory::createFunction(StringRef name, ArrayRef<Type> retTypes,
                                     ArrayRef<Type> inArgTypes,
                                     ModuleOp module) {
  OpBuilder rewriter(module);
  auto *context = module.getContext();
  if (auto func = module.lookupSymbol<func::FuncOp>(name))
    return func;

  // Create the LLVM FunctionType
  auto fType = FunctionType::get(context, inArgTypes, retTypes);

  // Insert the function since it hasn't been seen yet
  auto insPt = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(module.getBody());
  auto func = rewriter.create<func::FuncOp>(module->getLoc(), name, fType);
  rewriter.restoreInsertionPoint(insPt);
  return func;
}

std::optional<std::uint64_t> factory::maybeValueOfIntConstant(Value v) {
  APInt cst;
  if (matchPattern(v, m_ConstantInt(&cst)))
    return {cst.getZExtValue()};
  return std::nullopt;
}

std::optional<double> factory::maybeValueOfFloatConstant(Value v) {
  APFloat cst(0.0);
  if (matchPattern(v, m_ConstantFloat(&cst)))
    return {cst.convertToDouble()};
  return std::nullopt;
}

void factory::createGlobalCtorCall(ModuleOp mod, FlatSymbolRefAttr ctor) {
  auto *ctx = mod.getContext();
  auto loc = mod.getLoc();
  auto ctorAttr = ArrayAttr::get(ctx, {ctor});
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(mod.getBody());
  auto i32Ty = builder.getI32Type();
  constexpr int prio = 17;
  auto prioAttr = ArrayAttr::get(ctx, {IntegerAttr::get(i32Ty, prio)});
  builder.create<LLVM::GlobalCtorsOp>(loc, ctorAttr, prioAttr);
}

cc::LoopOp factory::createInvariantLoop(
    OpBuilder &builder, Location loc, Value totalIterations,
    llvm::function_ref<void(OpBuilder &, Location, Region &, Block &)>
        bodyBuilder) {
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 64);
  Type i64Ty = builder.getI64Type();
  SmallVector<Value> inputs = {zero};
  SmallVector<Type> resultTys = {i64Ty};
  auto loop = builder.create<cc::LoopOp>(
      loc, resultTys, inputs, /*postCondition=*/false,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{i64Ty});
        auto &block = *builder.getBlock();
        Value cmpi = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, block.getArgument(0),
            totalIterations);
        builder.create<cc::ConditionOp>(loc, cmpi, block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{i64Ty});
        auto &block = *builder.getBlock();
        bodyBuilder(builder, loc, region, block);
        builder.create<cc::ContinueOp>(loc, block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region, TypeRange{i64Ty});
        auto &block = *builder.getBlock();
        auto incr =
            builder.create<arith::AddIOp>(loc, block.getArgument(0), one);
        builder.create<cc::ContinueOp>(loc, ValueRange{incr});
      });
  loop->setAttr("invariant", builder.getUnitAttr());
  return loop;
}

// This builder will transform the monotonic loop into an invariant loop during
// construction. This is meant to save some time in loop analysis and
// normalization, which would perform a similar transformation.
cc::LoopOp factory::createMonotonicLoop(
    OpBuilder &builder, Location loc, Value start, Value stop, Value step,
    llvm::function_ref<void(OpBuilder &, Location, Region &, Block &)>
        bodyBuilder) {
  IRBuilder irBuilder(builder.getContext());
  auto mod = builder.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  [[maybe_unused]] auto loadedIntrinsic =
      irBuilder.loadIntrinsic(mod, getCudaqSizeFromTriple);
  assert(succeeded(loadedIntrinsic) && "loading intrinsic should never fail");
  auto i64Ty = builder.getI64Type();
  Value begin =
      builder.create<cc::CastOp>(loc, i64Ty, start, cc::CastOpMode::Signed);
  Value stepBy =
      builder.create<cc::CastOp>(loc, i64Ty, step, cc::CastOpMode::Signed);
  Value end =
      builder.create<cc::CastOp>(loc, i64Ty, stop, cc::CastOpMode::Signed);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  SmallVector<Value> inputs = {zero, begin};
  SmallVector<Type> resultTys = {i64Ty, i64Ty};
  auto totalIters = builder.create<func::CallOp>(
      loc, i64Ty, getCudaqSizeFromTriple, ValueRange{begin, end, stepBy});
  auto loop = builder.create<cc::LoopOp>(
      loc, resultTys, inputs, /*postCondition=*/false,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region,
                                     TypeRange{i64Ty, i64Ty});
        auto &block = *builder.getBlock();
        Value cmpi = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, block.getArgument(0),
            totalIters.getResult(0));
        builder.create<cc::ConditionOp>(loc, cmpi, block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region,
                                     TypeRange{i64Ty, i64Ty});
        auto &block = *builder.getBlock();
        bodyBuilder(builder, loc, region, block);
        builder.create<cc::ContinueOp>(loc, block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cc::RegionBuilderGuard guard(builder, loc, region,
                                     TypeRange{i64Ty, i64Ty});
        auto &block = *builder.getBlock();
        auto one = builder.create<arith::ConstantIntOp>(loc, 1, 64);
        Value count =
            builder.create<arith::AddIOp>(loc, block.getArgument(0), one);
        Value incr =
            builder.create<arith::AddIOp>(loc, block.getArgument(1), stepBy);
        builder.create<cc::ContinueOp>(loc, ValueRange{count, incr});
      });
  loop->setAttr("invariant", builder.getUnitAttr());
  return loop;
}

// FIXME: some ABIs may return a small struct in registers rather than via an
// sret pointer.
//
// On x86_64,
//   pair of:  argument         return value    packed from msb to lsb
//    i32   :   i64              i64             (second, first)
//    i64   :   i64, i64         { i64, i64 }
//    f32   :   <2 x float>      <2 x float>
//    f64   :   double, double   { double, double }
//
// On aarch64,
//   pair of:  argument         return value    packed from msb to lsb
//    i32   :   i64              i64             (second, first)
//    i64   :   [2 x i64]        [2 x i64]
//    f32   :   [2 x float]      { float, float }
//    f64   :   [2 x double]     { double, double }
bool factory::hasHiddenSRet(FunctionType funcTy) {
  // If a function has more than 1 result, the results are promoted to a
  // structured return argument. Otherwise, if there is 1 result and it is an
  // aggregate type, then it is promoted to a structured return argument.
  auto numResults = funcTy.getNumResults();
  return numResults > 1 ||
         (numResults == 1 && funcTy.getResult(0)
                                 .isa<cc::SpanLikeType, cc::StructType,
                                      cc::ArrayType, cc::CallableType>());
}

cc::StructType factory::stlStringType(MLIRContext *ctx) {
  auto i8Ty = IntegerType::get(ctx, 8);
  auto ptrI8Ty = cc::PointerType::get(i8Ty);
  auto i64Ty = IntegerType::get(ctx, 64);
  auto padTy = cc::ArrayType::get(ctx, i8Ty, 16);
  return cc::StructType::get(ctx, ArrayRef<Type>{ptrI8Ty, i64Ty, padTy});
}

// FIXME: We should get the underlying structure of a std::vector from the
// AST. For expediency, we just construct the expected type directly here.
cc::StructType factory::stlVectorType(Type eleTy) {
  MLIRContext *ctx = eleTy.getContext();
  auto ptrTy = cc::PointerType::get(eleTy);
  return cc::StructType::get(ctx, ArrayRef<Type>{ptrTy, ptrTy, ptrTy});
}

// FIXME: Give these front-end names so we can disambiguate more types.
cc::StructType factory::getDynamicBufferType(MLIRContext *ctx) {
  auto ptrTy = cc::PointerType::get(IntegerType::get(ctx, 8));
  return cc::StructType::get(ctx,
                             ArrayRef<Type>{ptrTy, IntegerType::get(ctx, 64)});
}

cc::PointerType factory::getIndexedObjectType(mlir::Type eleTy) {
  return cc::PointerType::get(cc::ArrayType::get(eleTy));
}

Type factory::getSRetElementType(FunctionType funcTy) {
  assert(funcTy.getNumResults() && "function type must have results");
  auto *ctx = funcTy.getContext();
  if (funcTy.getNumResults() > 1)
    return cc::StructType::get(ctx, funcTy.getResults());
  if (isa<cc::SpanLikeType>(funcTy.getResult(0)))
    return getDynamicBufferType(ctx);
  return funcTy.getResult(0);
}

static Type convertToHostSideType(Type ty) {
  if (auto memrefTy = dyn_cast<cc::StdvecType>(ty))
    return convertToHostSideType(
        factory::stlVectorType(memrefTy.getElementType()));
  if (auto memrefTy = dyn_cast<cc::CharspanType>(ty))
    return convertToHostSideType(factory::stlStringType(memrefTy.getContext()));
  auto *ctx = ty.getContext();
  if (auto structTy = dyn_cast<cc::StructType>(ty)) {
    SmallVector<Type> newMembers;
    for (auto mem : structTy.getMembers())
      newMembers.push_back(convertToHostSideType(mem));
    if (structTy.getName())
      return cc::StructType::get(ctx, structTy.getName(), newMembers,
                                 structTy.getBitSize(), structTy.getAlignment(),
                                 structTy.getPacked());
    return cc::StructType::get(ctx, newMembers, structTy.getBitSize(),
                               structTy.getAlignment(), structTy.getPacked());
  }
  if (auto memrefTy = dyn_cast<quake::VeqType>(ty)) {
    // Use pointer as these must be pass-by-reference.
    return cc::PointerType::get(factory::stlVectorType(
        IntegerType::get(ctx, /*FIXME sizeof a pointer?*/ 64)));
  }
  return ty;
}

// This code intends to simulate the X86_64ABIInfo::classify() function. That
// function tries to simulate GCC argument passing conventions. classify() also
// has a number of FIXME comments, where it diverges from the referenced ABI.
// Empirical evidence show that on x86_64, integers and floats are packed in
// integers of size 32 or 64 together, unless the float member fits by itself.
static bool shouldExpand(SmallVectorImpl<Type> &packedTys,
                         cc::StructType structTy) {
  if (structTy.isEmpty())
    return false;
  auto *ctx = structTy.getContext();
  unsigned bits = 0;

  // First split the members into a "lo" set and a "hi" set.
  SmallVector<Type> set1;
  SmallVector<Type> set2;
  for (auto ty : structTy.getMembers()) {
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      bits += intTy.getWidth();
      if (bits <= 64)
        set1.push_back(ty);
      else
        set2.push_back(ty);
    } else if (auto fltTy = dyn_cast<FloatType>(ty)) {
      bits += fltTy.getWidth();
      if (bits <= 64)
        set1.push_back(ty);
      else
        set2.push_back(ty);
    } else {
      return false;
    }
  }

  // Process the sets. If the set has anything integral, use integer. If the set
  // has one float or double, use it. Otherwise the set has 2 floats, and we use
  // <2 x f32>.
  auto useInt = [&](auto theSet) {
    for (auto ty : theSet)
      if (isa<IntegerType>(ty))
        return true;
    return false;
  };
  auto processMembers = [&](auto theSet, unsigned packIdx) {
    if (useInt(theSet)) {
      packedTys[packIdx] = IntegerType::get(ctx, bits > 32 ? 64 : 32);
    } else if (theSet.size() == 1) {
      packedTys[packIdx] = theSet[0];
    } else {
      packedTys[packIdx] =
          VectorType::get(ArrayRef<std::int64_t>{2}, theSet[0]);
    }
  };
  assert(!set1.empty() && "struct must have members");
  packedTys.resize(set2.empty() ? 1 : 2);
  processMembers(set1, 0);
  if (!set2.empty())
    processMembers(set2, 1);
  return true;
}

bool factory::structUsesTwoArguments(mlir::Type ty) {
  // Unchecked! This is only valid if target is X86-64.
  auto structTy = dyn_cast<cc::StructType>(ty);
  if (!structTy || structTy.getBitSize() == 0 || structTy.getBitSize() > 128)
    return false;
  SmallVector<Type> unused;
  return shouldExpand(unused, structTy);
}

static bool onlyArithmeticMembers(cc::StructType structTy) {
  for (auto t : structTy.getMembers()) {
    // FIXME: check complex type
    if (isa<IntegerType, FloatType>(t))
      continue;
    return false;
  }
  return true;
}

// When the kernel comes from a class, there is always a default `this` argument
// to the kernel entry function. The CUDA Quantum spec doesn't allow the kernel
// object to contain data members (yet), so we can ignore the `this` pointer.
FunctionType factory::toHostSideFuncType(FunctionType funcTy, bool addThisPtr,
                                         ModuleOp module) {
  auto *ctx = funcTy.getContext();
  SmallVector<Type> inputTys;
  bool hasSRet = false;
  if (factory::hasHiddenSRet(funcTy)) {
    // When the kernel is returning a std::vector<T> result, the result is
    // returned via a sret argument in the first position. When this argument
    // is added, the this pointer becomes the second argument. Both are opaque
    // pointers at this point.
    auto eleTy = convertToHostSideType(getSRetElementType(funcTy));
    inputTys.push_back(cc::PointerType::get(eleTy));
    hasSRet = true;
  }
  // If this kernel is a plain old function or a static member function, we
  // don't want to add a hidden `this` argument.
  auto i64Ty = IntegerType::get(ctx, 64);
  auto ptrTy = cc::PointerType::get(IntegerType::get(ctx, 8));
  if (addThisPtr)
    inputTys.push_back(ptrTy);

  // Add all the explicit (not hidden) arguments after the hidden ones.
  for (auto kernelTy : funcTy.getInputs()) {
    auto hostTy = convertToHostSideType(kernelTy);
    if (auto strTy = dyn_cast<cc::StructType>(hostTy)) {
      // On x86_64 and aarch64, a struct that is smaller than 128 bits may be
      // passed in registers as separate arguments. See classifyArgumentType()
      // in CodeGen/TargetInfo.cpp.
      if (strTy.getBitSize() != 0 && strTy.getBitSize() <= 128) {
        if (isX86_64(module)) {
          SmallVector<Type, 2> packedTys;
          if (shouldExpand(packedTys, strTy)) {
            for (auto ty : packedTys)
              inputTys.push_back(ty);
            continue;
          }
        } else {
          assert(isAArch64(module) && "aarch64 expected");
          if (onlyArithmeticMembers(strTy)) {
            // Empirical evidence shows that on aarch64, arguments are packed
            // into a single i64 or a [2 x i64] typed value based on the size of
            // the struct. This is regardless of whether the value(s) are
            // floating-point or not.
            if (strTy.getBitSize() > 64)
              inputTys.push_back(cc::ArrayType::get(ctx, i64Ty, 2));
            else
              inputTys.push_back(i64Ty);
            continue;
          }
        }
      }
      // Pass a struct as a byval pointer.
      hostTy = cc::PointerType::get(hostTy);
    }
    inputTys.push_back(hostTy);
  }

  // Handle the result type. We only add a result type when there is a result
  // and it hasn't been converted to a hidden sret argument.
  if (funcTy.getNumResults() == 0 || hasSRet)
    return FunctionType::get(ctx, inputTys, {});
  assert(funcTy.getNumResults() == 1);
  return FunctionType::get(ctx, inputTys, funcTy.getResults());
}

bool factory::isStdVecArg(Type type) {
  auto ptrTy = dyn_cast<cc::PointerType>(type);
  if (!ptrTy)
    return false;

  auto elementTy = ptrTy.getElementType();
  auto structTy = dyn_cast<cc::StructType>(elementTy);
  if (!structTy)
    return false;

  auto memberTys = structTy.getMembers();
  if (memberTys.size() != 3)
    return false;

  for (std::size_t i = 0; i < 3; i++)
    if (!dyn_cast<cc::PointerType>(memberTys[i]))
      return false;

  // This is a stdvec type to us.
  return true;
}

} // namespace cudaq::opt
