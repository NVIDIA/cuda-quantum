/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

using namespace mlir;

namespace cudaq::opt {

/// Return an i64 array where the kth element is N if the kth
/// operand is veq<N> and 0 otherwise (e.g. is a ref).
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
  auto getSizeSymbolRef = cudaq::opt::factory::createLLVMFunctionSymbol(
      cudaq::opt::QIRArrayGetSize, i64Type, {cudaq::opt::getArrayType(context)},
      parentModule);
  for (auto iter : llvm::enumerate(operands)) {
    auto operand = iter.value();
    auto i = iter.index();
    Value idx = rewriter.create<arith::ConstantIntOp>(loc, i, 64);
    Value ptr = rewriter.create<LLVM::GEPOp>(loc, intPtrTy, isArrayAndLengthArr,
                                             ValueRange{idx});
    Value element;
    if (operand.getType() == cudaq::opt::getQubitType(context))
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

cudaq::cc::LoopOp factory::createInvariantLoop(
    OpBuilder &builder, Location loc, Value totalIterations,
    llvm::function_ref<void(OpBuilder &, Location, Region &, Block &)>
        bodyBuilder) {
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Type indexTy = builder.getIndexType();
  SmallVector<Value> inputs = {zero};
  SmallVector<Type> resultTys = {indexTy};
  auto loop = builder.create<cudaq::cc::LoopOp>(
      loc, resultTys, inputs, /*postCondition=*/false,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region,
                                            TypeRange{zero.getType()});
        auto &block = *builder.getBlock();
        Value cmpi = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, block.getArgument(0),
            totalIterations);
        builder.create<cudaq::cc::ConditionOp>(loc, cmpi, block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region,
                                            TypeRange{zero.getType()});
        auto &block = *builder.getBlock();
        bodyBuilder(builder, loc, region, block);
        builder.create<cudaq::cc::ContinueOp>(loc, block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region,
                                            TypeRange{zero.getType()});
        auto &block = *builder.getBlock();
        auto incr =
            builder.create<arith::AddIOp>(loc, block.getArgument(0), one);
        builder.create<cudaq::cc::ContinueOp>(loc, ValueRange{incr});
      });
  loop->setAttr("invariant", builder.getUnitAttr());
  return loop;
}

bool factory::hasHiddenSRet(FunctionType funcTy) {
  // If a function has more than 1 result, the results are promoted to a
  // structured return argument. Otherwise, if there is 1 result and it is an
  // aggregate type, then it is promoted to a structured return argument.
  auto numResults = funcTy.getNumResults();
  return numResults > 1 ||
         (numResults == 1 &&
          funcTy.getResult(0)
              .isa<cudaq::cc::StdvecType, cudaq::cc::StructType,
                   cudaq::cc::ArrayType, cudaq::cc::CallableType>());
}

// FIXME: We should get the underlying structure of a std::vector from the
// AST. For expediency, we just construct the expected type directly here.
cudaq::cc::StructType factory::stlVectorType(Type eleTy) {
  MLIRContext *ctx = eleTy.getContext();
  auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
  SmallVector<Type> eleTys = {elePtrTy, elePtrTy, elePtrTy};
  return cudaq::cc::StructType::get(ctx, eleTys);
}

cudaq::cc::StructType factory::getDynamicBufferType(MLIRContext *ctx) {
  auto ptrTy = cudaq::cc::PointerType::get(IntegerType::get(ctx, 8));
  return cudaq::cc::StructType::get(
      ctx, ArrayRef<Type>{ptrTy, IntegerType::get(ctx, 64)});
}

Type factory::getSRetElementType(FunctionType funcTy) {
  assert(funcTy.getNumResults() && "function type must have results");
  auto *ctx = funcTy.getContext();
  if (funcTy.getNumResults() > 1)
    return cudaq::cc::StructType::get(ctx, funcTy.getResults());
  if (isa<cudaq::cc::StdvecType>(funcTy.getResult(0)))
    return getDynamicBufferType(ctx);
  return funcTy.getResult(0);
}

FunctionType factory::toCpuSideFuncType(FunctionType funcTy, bool addThisPtr) {
  auto *ctx = funcTy.getContext();
  // When the kernel comes from a class, there is always a default "this"
  // argument to the kernel entry function. The CUDA Quantum language spec
  // doesn't allow the kernel object to contain data members (yet), so we can
  // ignore the `this` pointer.
  SmallVector<Type> inputTys;
  bool hasSRet = false;
  if (factory::hasHiddenSRet(funcTy)) {
    // When the kernel is returning a std::vector<T> result, the result is
    // returned via a sret argument in the first position. When this argument
    // is added, the this pointer becomes the second argument. Both are opaque
    // pointers at this point.
    auto eleTy = getSRetElementType(funcTy);
    inputTys.push_back(cudaq::cc::PointerType::get(eleTy));
    hasSRet = true;
  }
  // If this kernel is a plain old function or a static member function, we
  // don't want to add a hidden `this` argument.
  auto ptrTy = cudaq::cc::PointerType::get(IntegerType::get(ctx, 8));
  if (addThisPtr)
    inputTys.push_back(ptrTy);

  // Add all the explicit (not hidden) arguments after the hidden ones.
  for (auto inTy : funcTy.getInputs()) {
    if (auto memrefTy = dyn_cast<cudaq::cc::StdvecType>(inTy))
      inputTys.push_back(cudaq::cc::PointerType::get(
          stlVectorType(memrefTy.getElementType())));
    else if (auto structTy = dyn_cast<cudaq::cc::StructType>(inTy))
      // cc.struct args are callable (at this point), need them as pointers
      // for the new entry point
      inputTys.push_back(cudaq::cc::PointerType::get(structTy));
    else if (auto memrefTy = dyn_cast<quake::VeqType>(inTy))
      inputTys.push_back(cudaq::cc::PointerType::get(stlVectorType(
          IntegerType::get(ctx, /*FIXME sizeof a pointer?*/ 64))));
    else
      inputTys.push_back(inTy);
  }

  // Handle the result type. We only add a result type when there is a result
  // and it hasn't been converted to a hidden sret argument.
  if (funcTy.getNumResults() == 0 || hasSRet)
    return FunctionType::get(ctx, inputTys, {});
  assert(funcTy.getNumResults() == 1);
  return FunctionType::get(ctx, inputTys, funcTy.getResults());
}

bool factory::isStdVecArg(Type type) {
  auto ptrTy = dyn_cast<cudaq::cc::PointerType>(type);
  if (!ptrTy)
    return false;

  auto elementTy = ptrTy.getElementType();
  auto structTy = dyn_cast<cudaq::cc::StructType>(elementTy);
  if (!structTy)
    return false;

  auto memberTys = structTy.getMembers();
  if (memberTys.size() != 3)
    return false;

  for (std::size_t i = 0; i < 3; i++)
    if (!dyn_cast<cudaq::cc::PointerType>(memberTys[i]))
      return false;

  // This is a stdvec type to us.
  return true;
}

} // namespace cudaq::opt
