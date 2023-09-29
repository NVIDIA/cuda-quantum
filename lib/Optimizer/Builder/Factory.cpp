/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"

using namespace mlir;

namespace cudaq::opt {

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
  return funcTy.getNumResults() == 1 &&
         funcTy.getResult(0).isa<cudaq::cc::StdvecType>();
}

// FIXME: We should get the underlying structure of a std::vector from the
// AST. For expediency, we just construct the expected type directly here.
static cudaq::cc::StructType stlVectorType(Type eleTy) {
  MLIRContext *ctx = eleTy.getContext();
  auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
  SmallVector<Type> eleTys = {elePtrTy, elePtrTy, elePtrTy};
  return cudaq::cc::StructType::get(ctx, eleTys);
}

FunctionType factory::toCpuSideFuncType(FunctionType funcTy, bool addThisPtr) {
  auto *ctx = funcTy.getContext();
  // When the kernel comes from a class, there is always a default "this"
  // argument to the kernel entry function. The CUDA Quantum language spec
  // doesn't allow the kernel object to contain data members (yet), so we can
  // ignore the `this` pointer.
  auto ptrTy = cudaq::cc::PointerType::get(IntegerType::get(ctx, 8));
  SmallVector<Type> inputTys;
  // If this kernel is a plain old function or a static member function, we
  // don't want to add a hidden `this` argument.
  if (addThisPtr)
    inputTys.push_back(ptrTy);
  bool hasSRet = false;
  if (factory::hasHiddenSRet(funcTy)) {
    // When the kernel is returning a std::vector<T> result, the result is
    // returned via a sret argument in the first position. When this argument
    // is added, the this pointer becomes the second argument. Both are opaque
    // pointers at this point.
    inputTys.push_back(ptrTy);
    hasSRet = true;
  }

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
