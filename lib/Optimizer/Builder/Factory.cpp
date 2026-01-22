/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace cudaq::opt {

// The common small struct limit for architectures cudaq is supporting.
static constexpr unsigned CommonSmallStructSize = 128;

bool factory::isX86_64(ModuleOp module) {
  std::string triple;
  if (auto ta = module->getAttr(targetTripleAttrName))
    triple = cast<StringAttr>(ta).str();
  else
    triple = llvm::sys::getDefaultTargetTriple();
  llvm::Triple tr(triple);
  return tr.getArch() == llvm::Triple::x86_64;
}

bool factory::isAArch64(ModuleOp module) {
  std::string triple;
  if (auto ta = module->getAttr(targetTripleAttrName))
    triple = cast<StringAttr>(ta).str();
  else
    triple = llvm::sys::getDefaultTargetTriple();
  llvm::Triple tr(triple);
  return tr.getArch() == llvm::Triple::aarch64;
}

template <bool isOutput>
Type genBufferType(Type ty) {
  auto *ctx = ty.getContext();
  if (isa<cudaq::cc::CallableType>(ty))
    return cudaq::cc::PointerType::get(ctx);
  if (isa<cudaq::cc::IndirectCallableType>(ty))
    return IntegerType::get(ctx, 64);
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

cudaq::cc::StructType
factory::buildInvokeStructType(FunctionType funcTy,
                               std::size_t startingArgIdx) {
  auto *ctx = funcTy.getContext();
  SmallVector<Type> eleTys;
  for (auto inTy : llvm::enumerate(funcTy.getInputs()))
    if (inTy.index() >= startingArgIdx)
      eleTys.push_back(genBufferType</*isOutput=*/false>(inTy.value()));
  for (auto outTy : funcTy.getResults())
    eleTys.push_back(genBufferType</*isOutput=*/true>(outTy));
  return cudaq::cc::StructType::get(ctx, eleTys, /*packed=*/false);
}

Value factory::packIsArrayAndLengthArray(Location loc,
                                         ConversionPatternRewriter &rewriter,
                                         ModuleOp parentModule,
                                         std::size_t numOperands,
                                         ValueRange operands) {
  // Create an integer array where the kth element is N if the kth control
  // operand is a veq<N>, and 0 otherwise.
  auto i64Type = rewriter.getI64Type();
  auto context = rewriter.getContext();
  Value isArrayAndLengthArr = createLLVMTemporary(
      loc, rewriter, LLVM::LLVMPointerType::get(i64Type), numOperands);
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

bool factory::isConstantOp(Value v) {
  Attribute attr;
  return matchPattern(v, m_Constant(&attr));
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

/// Create a temporary on the stack. The temporary is created such that it is
/// \em{not} control dependent (other than on function entry).
Value factory::createLLVMTemporary(Location loc, OpBuilder &builder, Type type,
                                   std::size_t size) {
  Operation *op = builder.getBlock()->getParentOp();
  auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
  if (!func)
    func = op->getParentOfType<LLVM::LLVMFuncOp>();
  assert(func && "must be in a function");
  auto *entryBlock = &func.getRegion().front();
  assert(entryBlock && "function must have an entry block");
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entryBlock);
  Value len = genLlvmI64Constant(loc, builder, size);
  return builder.create<LLVM::AllocaOp>(loc, type, ArrayRef<Value>{len});
}

Value factory::createTemporary(Location loc, OpBuilder &builder, Type type,
                               std::size_t size) {
  Operation *op = builder.getBlock()->getParentOp();
  auto func = dyn_cast<func::FuncOp>(op);
  if (!func)
    func = op->getParentOfType<func::FuncOp>();
  assert(func && "must be in a function");
  auto *entryBlock = &func.getRegion().front();
  assert(entryBlock && "function must have an entry block");
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entryBlock);
  Value len = builder.create<arith::ConstantIntOp>(loc, size, 64);
  return builder.create<cudaq::cc::AllocaOp>(loc, type, len);
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

cc::ArrayType factory::genHostStringType(ModuleOp mod) {
  auto *ctx = mod.getContext();
  auto i8Ty = IntegerType::get(ctx, 8);
  auto sizeAttr = mod->getAttr(cudaq::runtime::sizeofStringAttrName);
  if (sizeAttr) {
    auto size = cast<IntegerAttr>(sizeAttr).getInt();
    return cc::ArrayType::get(ctx, i8Ty, size);
  }
  return cc::ArrayType::get(ctx, i8Ty, sizeof(std::string));
}

// FIXME: We should get the underlying structure of a std::vector from the
// AST. For expediency, we just construct the expected type directly here.
cc::StructType factory::stlVectorType(Type eleTy) {
  MLIRContext *ctx = eleTy.getContext();
  auto ptrTy = cc::PointerType::get(eleTy);
  return cc::StructType::get(ctx, ArrayRef<Type>{ptrTy, ptrTy, ptrTy});
}

// Note that this is the raw host type, where std::vector<bool> is distinct.
// When converting to the device side, the distinction is deliberately removed
// making std::vector<bool> the same format as std::vector<char>.
static cc::StructType stlHostVectorType(Type eleTy) {
  MLIRContext *ctx = eleTy.getContext();
  if (eleTy != IntegerType::get(ctx, 1)) {
    // std::vector<T> where T != bool.
    return factory::stlVectorType(eleTy);
  }
  // std::vector<bool> is a different type than std::vector<T>.
  auto ptrTy = cc::PointerType::get(eleTy);
  auto i8Ty = IntegerType::get(ctx, 8);
  auto padout = cc::ArrayType::get(ctx, i8Ty, 32);
  return cc::StructType::get(ctx, ArrayRef<Type>{ptrTy, padout});
}

bool factory::isStlVectorBoolHostType(Type ty) {
  auto strTy = dyn_cast<cc::StructType>(ty);
  if (!strTy)
    return false;
  if (strTy.getMembers().size() != 2)
    return false;
  auto ptrTy = dyn_cast<cc::PointerType>(strTy.getMember(0));
  if (!ptrTy)
    return false;
  if (ptrTy.getElementType() != IntegerType::get(ty.getContext(), 1))
    return false;
  auto arrTy = dyn_cast<cc::ArrayType>(strTy.getMember(1));
  if (!arrTy)
    return false;
  if (arrTy.getElementType() != IntegerType::get(ty.getContext(), 8))
    return false;
  if (arrTy.isUnknownSize() || (arrTy.getSize() != 32))
    return false;
  return true;
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
  if (auto spanTy = dyn_cast<cc::SpanLikeType>(funcTy.getResult(0)))
    return stlHostVectorType(spanTy.getElementType());
  return funcTy.getResult(0);
}

Type factory::convertToHostSideType(Type ty, ModuleOp mod) {
  if (auto memrefTy = dyn_cast<cc::StdvecType>(ty))
    return stlHostVectorType(
        convertToHostSideType(memrefTy.getElementType(), mod));
  if (isa<cc::IndirectCallableType>(ty))
    return cc::PointerType::get(IntegerType::get(ty.getContext(), 8));
  if (auto csTy = dyn_cast<cc::CharspanType>(ty))
    return genHostStringType(mod);
  auto *ctx = ty.getContext();
  if (auto structTy = dyn_cast<cc::StructType>(ty)) {
    SmallVector<Type> newMembers;
    for (auto mem : structTy.getMembers())
      newMembers.push_back(convertToHostSideType(mem, mod));
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
// integers of size 8, 16, 24, 32 or 64 together, unless the float member fits
// by itself.
static bool shouldExpand(SmallVectorImpl<Type> &packedTys,
                         cc::StructType structTy, unsigned scaling = 8) {
  if (structTy.isEmpty())
    return false;
  auto *ctx = structTy.getContext();
  unsigned bits = 0;
  const auto scaleBy = scaling - 1;
  auto scaleBits = [&](unsigned size) {
    if (size < 32)
      size = (size + scaleBy) & ~scaleBy;
    if (size > 32 && size <= 64)
      size = 64;
    return size;
  };

  // First split the members into a "lo" set and a "hi" set.
  SmallVector<Type> set1;
  SmallVector<Type> set2;
  for (auto ty : structTy.getMembers()) {
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      auto addBits = scaleBits(intTy.getWidth());
      if (bits + addBits <= 64) {
        bits += addBits;
        set1.push_back(ty);
      } else {
        bits = std::max(bits, 64u) + addBits;
        set2.push_back(ty);
      }
    } else if (auto fltTy = dyn_cast<FloatType>(ty)) {
      auto addBits = fltTy.getWidth();
      if (bits + addBits <= 64) {
        bits += addBits;
        set1.push_back(ty);
      } else {
        bits = std::max(bits, 64u) + addBits;
        set2.push_back(ty);
      }
    } else {
      return false;
    }
    if (bits > CommonSmallStructSize)
      return false;
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
  auto intSetSize = [&](auto theSet) {
    unsigned size = 0;
    for (auto ty : theSet)
      size += scaleBits(ty.getIntOrFloatBitWidth());
    return size;
  };
  auto processMembers = [&](auto theSet, unsigned packIdx) {
    if (useInt(theSet)) {
      auto size = intSetSize(theSet);
      if (size <= 32)
        packedTys[packIdx] = IntegerType::get(ctx, size);
      else
        packedTys[packIdx] = IntegerType::get(ctx, 64);
    } else if (theSet.size() == 1) {
      packedTys[packIdx] = theSet[0];
    } else {
      assert(theSet[0] == FloatType::getF32(ctx) && "must be float");
      packedTys[packIdx] =
          VectorType::get(ArrayRef<std::int64_t>{2}, theSet[0]);
    }
  };
  assert(!set1.empty() && "struct must have members");
  packedTys.resize(set2.empty() ? 1 : 2);
  processMembers(set1, 0);
  if (set2.empty())
    return false;
  processMembers(set2, 1);
  return true;
}

bool factory::hasSRet(func::FuncOp funcOp) {
  if (funcOp.getNumArguments() > 0)
    if (auto dict = funcOp.getArgAttrDict(0))
      return dict.contains(LLVM::LLVMDialect::getStructRetAttrName());
  return false;
}

// On x86_64,
//   pair of:  argument         return value    packed from msb to lsb
//    i32   :   i64              i64             (second, first)
//    i64   :   i64, i64         { i64, i64 }
//    f32   :   <2 x float>      <2 x float>
//    f64   :   double, double   { double, double }
//    ptr   :   ptr, ptr         { ptr, ptr }
//
// On aarch64,
//   pair of:  argument         return value    packed from msb to lsb
//    i32   :   i64              i64             (second, first)
//    i64   :   [2 x i64]        [2 x i64]
//    f32   :   [2 x float]      { float, float }
//    f64   :   [2 x double]     { double, double }
//    ptr   :   [2 x i64]        [2 x i64]
bool factory::hasHiddenSRet(FunctionType funcTy) {
  // If a function has more than 1 result, the results are promoted to a
  // structured return argument. Otherwise, if there is 1 result and it is an
  // aggregate type, then it is promoted to a structured return argument.
  auto numResults = funcTy.getNumResults();
  if (numResults == 0)
    return false;
  if (numResults > 1)
    return true;
  auto resTy = funcTy.getResult(0);
  if (isa<cc::SpanLikeType, cc::ArrayType, cc::CallableType>(resTy))
    return true;
  if (auto strTy = dyn_cast<cc::StructType>(resTy)) {
    if (strTy.getMembers().empty())
      return false;
    SmallVector<Type> packedTys;
    bool inRegisters = shouldExpand(packedTys, strTy) || !packedTys.empty();
    return !inRegisters;
  }
  return false;
}

bool factory::structUsesTwoArguments(mlir::Type ty) {
  // Unchecked! This is only valid if target is X86-64.
  auto structTy = dyn_cast<cc::StructType>(ty);
  if (!structTy || structTy.getBitSize() == 0 ||
      structTy.getBitSize() > CommonSmallStructSize)
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

// Unchecked precondition: structTy must be entirely arithmetic.
static unsigned getLargestWidth(cc::StructType structTy) {
  unsigned largest = 8;
  for (auto ty : structTy.getMembers()) {
    auto width = ty.getIntOrFloatBitWidth();
    if (width > largest)
      largest = width;
  }
  return largest;
}

// When the kernel comes from a class, there is always a default `this` argument
// to the kernel entry function. The CUDA-Q spec doesn't allow the kernel
// object to contain data members (yet), so we can ignore the `this` pointer.
FunctionType factory::toHostSideFuncType(FunctionType funcTy, bool addThisPtr,
                                         ModuleOp module) {
  auto *ctx = funcTy.getContext();
  SmallVector<Type> inputTys;
  bool hasSRet = false;
  Type resultTy;
  auto i64Ty = IntegerType::get(ctx, 64);
  if (funcTy.getNumResults() == 1)
    if (auto strTy = dyn_cast<cc::StructType>(funcTy.getResult(0)))
      if (strTy.getBitSize() != 0 &&
          strTy.getBitSize() <= CommonSmallStructSize) {
        if (isX86_64(module)) {
          // X86_64: Byte addressable scaling (packed registers). Default is a
          // struct.
          SmallVector<Type, 2> packedTys;
          if (shouldExpand(packedTys, strTy) || !packedTys.empty()) {
            if (packedTys.size() == 1)
              resultTy = packedTys[0];
            else
              resultTy = cc::StructType::get(ctx, packedTys);
          }
        } else if (isAArch64(module) && onlyArithmeticMembers(strTy)) {
          // AARCH64: Padded registers. Default is a two-element array.
          unsigned largest = getLargestWidth(strTy);
          SmallVector<Type, 2> packedTys;
          if (shouldExpand(packedTys, strTy, largest) || !packedTys.empty()) {
            if (packedTys.size() == 1)
              resultTy = packedTys[0];
            else
              resultTy = cc::ArrayType::get(ctx, packedTys[0], 2);
          }
        }
      }
  if (!resultTy && funcTy.getNumResults()) {
    if (factory::hasHiddenSRet(funcTy)) {
      // When the kernel is returning a std::vector<T> result, the result is
      // returned via a sret argument in the first position. When this argument
      // is added, the this pointer becomes the second argument. Both are opaque
      // pointers at this point.
      auto eleTy = convertToHostSideType(getSRetElementType(funcTy), module);
      inputTys.push_back(cc::PointerType::get(eleTy));
      hasSRet = true;
    } else {
      assert(funcTy.getNumResults() == 1);
      resultTy = funcTy.getResult(0);
    }
  }
  // If this kernel is a plain old function or a static member function, we
  // don't want to add a hidden `this` argument.
  auto ptrTy = cc::PointerType::get(IntegerType::get(ctx, 8));
  if (addThisPtr)
    inputTys.push_back(ptrTy);

  // Add all the explicit (not hidden) arguments after the hidden ones.
  for (auto kernelTy : funcTy.getInputs()) {
    auto hostTy = convertToHostSideType(kernelTy, module);
    if (auto strTy = dyn_cast<cc::StructType>(hostTy)) {
      // On x86_64 and aarch64, a struct that is smaller than 128 bits may be
      // passed in registers as separate arguments. See classifyArgumentType()
      // in CodeGen/TargetInfo.cpp.
      if (strTy.getBitSize() != 0 &&
          strTy.getBitSize() <= CommonSmallStructSize) {
        if (isX86_64(module)) {
          SmallVector<Type, 2> packedTys;
          if (shouldExpand(packedTys, strTy)) {
            for (auto ty : packedTys)
              inputTys.push_back(ty);
            continue;
          } else if (!packedTys.empty()) {
            for (auto ty : packedTys)
              inputTys.push_back(ty);
            continue;
          }
        } else {
          assert(isAArch64(module) && "aarch64 expected");
          if (onlyArithmeticMembers(strTy)) {
            // Empirical evidence shows that on aarch64, arguments are packed
            // into a single i64 or a [2 x i64] typed value based on the size
            // of the struct. The exception is when there are 2 elements and
            // they are both float or both double.
            if ((strTy.getMembers().size() == 2) &&
                (strTy.getMember(0) == strTy.getMember(1)) &&
                ((strTy.getMember(0) == Float32Type::get(ctx)) ||
                 (strTy.getMember(0) == Float64Type::get(ctx))))
              inputTys.push_back(
                  cc::ArrayType::get(ctx, strTy.getMember(0), 2));
            else if (strTy.getBitSize() > 64)
              inputTys.push_back(cc::ArrayType::get(ctx, i64Ty, 2));
            else
              inputTys.push_back(i64Ty);
            continue;
          }
        }
      }
      // Pass a struct as a byval pointer.
      hostTy = cc::PointerType::get(hostTy);
    } else if (isa<cc::ArrayType>(hostTy)) {
      // Pass a raw data block as a pointer. (It's a struct passed as a blob.)
      hostTy = cc::PointerType::get(hostTy);
    }
    inputTys.push_back(hostTy);
  }

  // Handle the result type. We only add a result type when there is a result
  // and it hasn't been converted to a hidden sret argument.
  if (funcTy.getNumResults() == 0 || hasSRet)
    return FunctionType::get(ctx, inputTys, {});
  assert(funcTy.getNumResults() == 1 && resultTy);
  return FunctionType::get(ctx, inputTys, resultTy);
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

Value factory::createCast(OpBuilder &builder, Location loc, Type toType,
                          Value fromValue, bool signExtend, bool zeroExtend) {
  if (signExtend && zeroExtend) {
    emitError(loc, "cannot both sign and zero extend in a cast");
    return fromValue;
  }
  if (fromValue.getType() == toType)
    return fromValue;
  auto unit = UnitAttr::get(builder.getContext());
  UnitAttr none;
  return builder.create<cudaq::cc::CastOp>(loc, toType, fromValue,
                                           signExtend ? unit : none,
                                           zeroExtend ? unit : none);
}

std::vector<std::complex<double>>
factory::readGlobalConstantArray(cudaq::cc::GlobalOp &global) {
  std::vector<std::complex<double>> result{};

  auto attr = global.getValue();
  auto elementsAttr = cast<mlir::ElementsAttr>(attr.value());
  auto eleTy = elementsAttr.getElementType();
  auto values = elementsAttr.getValues<mlir::Attribute>();

  for (auto it = values.begin(); it != values.end(); ++it) {
    auto valAttr = *it;

    auto v = [&]() -> std::complex<double> {
      if (isa<FloatType>(eleTy))
        return {cast<FloatAttr>(valAttr).getValue().convertToDouble(),
                static_cast<double>(0.0)};
      if (isa<IntegerType>(eleTy))
        return {static_cast<double>(cast<IntegerAttr>(valAttr).getInt()),
                static_cast<double>(0.0)};
      assert(isa<ComplexType>(eleTy));
      auto arrayAttr = cast<mlir::ArrayAttr>(valAttr);
      auto real = cast<FloatAttr>(arrayAttr[0]).getValue().convertToDouble();
      auto imag = cast<FloatAttr>(arrayAttr[1]).getValue().convertToDouble();
      return {real, imag};
    }();

    result.push_back(v);
  }
  return result;
}

std::pair<mlir::func::FuncOp, bool>
factory::getOrAddFunc(mlir::Location loc, mlir::StringRef funcName,
                      mlir::FunctionType funcTy, mlir::ModuleOp module) {
  auto func = module.lookupSymbol<func::FuncOp>(funcName);
  if (func) {
    if (!func.empty()) {
      // Already lowered function func, skip it.
      return {func, /*defined=*/true};
    }
    // Function was declared but not defined.
    return {func, /*defined=*/false};
  }
  // Function not found, so add it to the module.
  OpBuilder build(module.getBodyRegion());
  OpBuilder::InsertionGuard guard(build);
  build.setInsertionPointToEnd(module.getBody());
  SmallVector<NamedAttribute> attrs;
  func = build.create<func::FuncOp>(loc, funcName, funcTy, attrs);
  func.setPrivate();
  return {func, /*defined=*/false};
}

void factory::mergeModules(ModuleOp into, ModuleOp from) {
  for (Operation &op : from) {
    auto sym = dyn_cast<SymbolOpInterface>(op);
    if (!sym)
      continue; // Only merge named symbols, avoids duplicating anonymous ops.

    // If `into` already has a symbol with this name, skip it.
    if (SymbolTable::lookupSymbolIn(into, sym.getName()))
      continue;

    into.push_back(op.clone());
  }
}
} // namespace cudaq::opt
