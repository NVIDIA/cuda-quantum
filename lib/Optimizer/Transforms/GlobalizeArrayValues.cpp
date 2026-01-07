/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_GLOBALIZEARRAYVALUES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "globalize-array-values"

using namespace mlir;

template <typename A, typename B>
SmallVector<A> conversion(ArrayAttr seq, Type) {
  SmallVector<A> result;
  for (auto v : seq) {
    B c = cast<B>(v);
    result.emplace_back(c.getValue());
  }
  return result;
}
template <>
SmallVector<APInt> conversion<APInt, IntegerAttr>(ArrayAttr seq, Type ty) {
  SmallVector<APInt> result;
  for (auto v : seq) {
    auto c = cast<IntegerAttr>(v);
    APInt ap = c.getValue();
    if (c.getType() != ty)
      result.emplace_back(ty.getIntOrFloatBitWidth(), ap.getLimitedValue());
    else
      result.emplace_back(ap);
  }
  return result;
}
template <>
SmallVector<std::complex<APFloat>>
conversion<std::complex<APFloat>, ArrayAttr>(ArrayAttr seq, Type) {
  SmallVector<std::complex<APFloat>> result;
  for (auto v : seq) {
    auto p = cast<ArrayAttr>(v);
    result.emplace_back(cast<FloatAttr>(p[0]).getValue(),
                        cast<FloatAttr>(p[1]).getValue());
  }
  return result;
}

static LogicalResult
convertArrayAttrToGlobalConstant(MLIRContext *ctx, Location loc,
                                 ArrayAttr arrAttr, ModuleOp module,
                                 StringRef globalName, Type eleTy) {
  cudaq::IRBuilder irBuilder(ctx);
  auto tensorTy = RankedTensorType::get(arrAttr.size(), eleTy);
  if (isa<ComplexType>(eleTy)) {
    auto blockValues =
        conversion<std::complex<APFloat>, ArrayAttr>(arrAttr, eleTy);
    auto dense = DenseElementsAttr::get(tensorTy, blockValues);
    irBuilder.genVectorOfConstants(loc, module, globalName, dense, eleTy);
  } else if (isa<FloatType>(eleTy)) {
    auto blockValues = conversion<APFloat, FloatAttr>(arrAttr, eleTy);
    auto dense = DenseElementsAttr::get(tensorTy, blockValues);
    irBuilder.genVectorOfConstants(loc, module, globalName, dense, eleTy);
  } else if (isa<IntegerType>(eleTy)) {
    auto blockValues = conversion<APInt, IntegerAttr>(arrAttr, eleTy);
    auto dense = DenseElementsAttr::get(tensorTy, blockValues);
    irBuilder.genVectorOfConstants(loc, module, globalName, dense, eleTy);
  } else {
    return failure();
  }
  return success();
}

/// Determine if this type, \p ty, a multidimensional array.
static bool multidimensionalArray(Type ty) {
  if (auto t0 = dyn_cast<cudaq::cc::ArrayType>(ty))
    return isa<cudaq::cc::ArrayType>(t0.getElementType());
  return false;
}

static bool useIsReifySpans(cudaq::cc::ConstantArrayOp conarr) {
  return (std::distance(conarr->user_begin(), conarr->user_end()) == 1) &&
         isa<cudaq::cc::ReifySpanOp>(*conarr->user_begin());
}

static bool useDataToInitState(cudaq::cc::ReifySpanOp reify) {
  for (auto *user : reify->getUsers())
    if (auto data = dyn_cast<cudaq::cc::StdvecDataOp>(user))
      if (std::distance(data->user_begin(), data->user_end()) == 1)
        return isa<quake::InitializeStateOp>(*data->user_begin());
  return false;
}

namespace {

// This pattern replaces a cc.const_array with a global constant. It can
// recognize a couple of usage patterns and will generate efficient IR in those
// cases.
//
// Pattern 1: The entire constant array is stored to a stack variable(s). Here
// we can eliminate the stack allocation and use the global constant.
//
// Pattern 2: Individual elements at dynamic offsets are extracted from the
// constant array and used. This can be replaced with a compute pointer
// operation using the global constant and a load of the element at the computed
// offset.
//
// Default: If the usage is not recognized, the constant array value is replaced
// with a load of the entire global variable. In this case, LLVM's optimizations
// are counted on to help demote the (large?) sequence value to primitive memory
// address arithmetic.
struct ConstantArrayPattern
    : public OpRewritePattern<cudaq::cc::ConstantArrayOp> {
  explicit ConstantArrayPattern(MLIRContext *ctx, ModuleOp module,
                                unsigned &counter)
      : OpRewritePattern{ctx}, module{module}, counter{counter} {}

  LogicalResult matchAndRewrite(cudaq::cc::ConstantArrayOp conarr,
                                PatternRewriter &rewriter) const override {
    auto func = conarr->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();
    if (useIsReifySpans(conarr) || multidimensionalArray(conarr.getType()))
      return failure();

    SmallVector<cudaq::cc::AllocaOp> allocas;
    SmallVector<cudaq::cc::StoreOp> stores;
    SmallVector<cudaq::cc::ExtractValueOp> extracts;
    bool loadAsValue = false;
    for (auto *usr : conarr->getUsers()) {
      auto store = dyn_cast<cudaq::cc::StoreOp>(usr);
      auto extract = dyn_cast<cudaq::cc::ExtractValueOp>(usr);
      if (store) {
        auto alloca = store.getPtrvalue().getDefiningOp<cudaq::cc::AllocaOp>();
        if (alloca) {
          stores.push_back(store);
          allocas.push_back(alloca);
          continue;
        }
      } else if (extract) {
        extracts.push_back(extract);
        continue;
      }
      loadAsValue = true;
    }
    std::string globalName =
        func.getName().str() + ".rodata_" + std::to_string(counter++);
    auto *ctx = rewriter.getContext();
    auto valueAttr = conarr.getConstantValues();
    auto eleTy = cast<cudaq::cc::ArrayType>(conarr.getType()).getElementType();
    if (failed(convertArrayAttrToGlobalConstant(ctx, conarr.getLoc(), valueAttr,
                                                module, globalName, eleTy)))
      return failure();
    auto loc = conarr.getLoc();
    if (!extracts.empty()) {
      auto base = rewriter.create<cudaq::cc::AddressOfOp>(
          loc, cudaq::cc::PointerType::get(conarr.getType()), globalName);
      auto elePtrTy = cudaq::cc::PointerType::get(eleTy);
      for (auto extract : extracts) {
        SmallVector<cudaq::cc::ComputePtrArg> args;
        unsigned i = 0;
        for (auto arg : extract.getRawConstantIndices()) {
          if (arg == cudaq::cc::ExtractValueOp::getDynamicIndexValue())
            args.push_back(extract.getDynamicIndices()[i++]);
          else
            args.push_back(arg);
        }
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(extract);
        auto addrVal =
            rewriter.create<cudaq::cc::ComputePtrOp>(loc, elePtrTy, base, args);
        rewriter.replaceOpWithNewOp<cudaq::cc::LoadOp>(extract, addrVal);
      }
    }
    if (!stores.empty()) {
      for (auto alloca : allocas)
        rewriter.replaceOpWithNewOp<cudaq::cc::AddressOfOp>(
            alloca, alloca.getType(), globalName);
      for (auto store : stores)
        rewriter.eraseOp(store);
    }
    if (loadAsValue) {
      auto base = rewriter.create<cudaq::cc::AddressOfOp>(
          loc, cudaq::cc::PointerType::get(conarr.getType()), globalName);
      rewriter.replaceOpWithNewOp<cudaq::cc::LoadOp>(conarr, base);
    }
    return success();
  }

  ModuleOp module;
  unsigned &counter;
};

/// This pattern converts a (possibly) multidimensional (and possibly ragged)
/// tree of constants into a collection of one-dimensional global constant
/// arrays and generates the boilerplate to construct a tree of spans around
/// those globals. The expansion of the tree of spans may involve a significant
/// number of more primitive operations. Ideally, the constant propagation pass
/// will have already eliminated the cc.reify_span operations.
struct ReifySpanPattern : public OpRewritePattern<cudaq::cc::ReifySpanOp> {
  explicit ReifySpanPattern(MLIRContext *ctx, ModuleOp module,
                            unsigned &counter)
      : OpRewritePattern{ctx}, module{module}, counter{counter} {}

  LogicalResult matchAndRewrite(cudaq::cc::ReifySpanOp reify,
                                PatternRewriter &rewriter) const override {
    auto conArr =
        reify.getElements().getDefiningOp<cudaq::cc::ConstantArrayOp>();
    if (!conArr)
      return failure();
    if (!multidimensionalArray(conArr.getType())) {
      if (useDataToInitState(reify)) {
        auto loc = reify.getLoc();
        auto eleTy =
            cast<cudaq::cc::StdvecType>(reify.getType()).getElementType();
        auto numEle = rewriter.create<arith::ConstantIntOp>(
            loc, conArr.getConstantValues().size(), 64);
        Value buff = rewriter.create<cudaq::cc::AllocaOp>(loc, eleTy, numEle);
        rewriter.create<cudaq::cc::StoreOp>(loc, conArr, buff);
        rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(
            reify, reify.getType(), buff, numEle);
        return success();
      }
    }

    Value replacementSpan = buildSpans(
        reify.getLoc(), cast<cudaq::cc::SpanLikeType>(reify.getType()),
        rewriter, conArr.getConstantValues());
    rewriter.replaceOp(reify, replacementSpan);
    return success();
  }

  Value buildSpans(Location loc, cudaq::cc::SpanLikeType ty,
                   PatternRewriter &rewriter, ArrayAttr arrAttr) const {
    SmallVector<Value> members;
    auto eleTy = ty.getElementType();
    for (auto attr : arrAttr) {
      if (auto a = dyn_cast<ArrayAttr>(attr)) {
        // Recursive case.
        members.push_back(
            buildSpans(loc, cast<cudaq::cc::SpanLikeType>(eleTy), rewriter, a));
      } else if (auto stringAttr = dyn_cast<StringAttr>(attr)) {
        // Strings require some special handling to build a proper span.
        auto *ctx = rewriter.getContext();
        std::int64_t len = stringAttr.getValue().size() + 1;
        Type litTy = cudaq::cc::PointerType::get(
            cudaq::cc::ArrayType::get(ctx, rewriter.getI8Type(), len));
        auto strLit = rewriter.create<cudaq::cc::CreateStringLiteralOp>(
            loc, litTy, stringAttr);
        auto size = rewriter.create<arith::ConstantIntOp>(loc, len, 64);
        members.push_back(rewriter.create<cudaq::cc::StdvecInitOp>(
            loc, cudaq::cc::CharspanType::get(ctx), strLit, size));
      } else if (auto a = dyn_cast<IntegerAttr>(attr)) {
        members.push_back(rewriter.create<arith::ConstantOp>(loc, a, eleTy));
      } else if (auto a = dyn_cast<FloatAttr>(attr)) {
        members.push_back(rewriter.create<arith::ConstantOp>(loc, a, eleTy));
      } else {
        // Unexpected attribute.
        LLVM_DEBUG(llvm::dbgs() << "unexpected attribute: " << attr << '\n');
        members.push_back(rewriter.create<cudaq::cc::PoisonOp>(loc, eleTy));
      }
    }

    // FIXME: get rid of this;
    // see https://github.com/NVIDIA/cuda-quantum/issues/3593
    auto hasBoolElems = false;
    if (auto iTy = dyn_cast<IntegerType>(eleTy)) {
      if (iTy.getWidth() == 1) {
        eleTy = IntegerType::get(ty.getContext(), 8);
        hasBoolElems = true;
      }
    }

    auto size = rewriter.create<arith::ConstantIntOp>(loc, members.size(), 64);
    auto buff = rewriter.create<cudaq::cc::AllocaOp>(loc, eleTy, size);
    for (auto iter : llvm::enumerate(members)) {
      std::int32_t idx = iter.index();
      auto m = iter.value();
      if (hasBoolElems) {
        auto unit = UnitAttr::get(rewriter.getContext());
        m = rewriter.create<cudaq::cc::CastOp>(loc, eleTy, m, UnitAttr(), unit);
      }
      auto ptrEleTy = cudaq::cc::PointerType::get(eleTy);
      auto ptr = rewriter.create<cudaq::cc::ComputePtrOp>(
          loc, ptrEleTy, buff, ArrayRef<cudaq::cc::ComputePtrArg>{idx});
      rewriter.create<cudaq::cc::StoreOp>(loc, m, ptr);
    }
    Value result =
        rewriter.create<cudaq::cc::StdvecInitOp>(loc, ty, buff, size);
    return result;
  }

  ModuleOp module;
  unsigned &counter;
};

/// This is a `ModuleOp` pass since it adds the arrays as global objects to the
/// `.rodata` section.
class GlobalizeArrayValuesPass
    : public cudaq::opt::impl::GlobalizeArrayValuesBase<
          GlobalizeArrayValuesPass> {
public:
  using GlobalizeArrayValuesBase::GlobalizeArrayValuesBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp module = getOperation();

    // Make the unchecked assumption that a ConstArrayOp was added by the
    // LiftArrayAlloc pass. This assumption means that the backing store of the
    // ConstArrayOp has been checked that it is never written to.
    RewritePatternSet patterns(ctx);
    unsigned counter = 0;
    patterns.insert<ReifySpanPattern, ConstantArrayPattern>(ctx, module,
                                                            counter);
    LLVM_DEBUG(llvm::dbgs() << "Before globalizing array values:\n"
                            << module << '\n');
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "After globalizing array values:\n"
                            << module << '\n');
  }
};
} // namespace
