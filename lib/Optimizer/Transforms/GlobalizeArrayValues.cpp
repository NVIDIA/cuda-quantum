/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
    patterns.insert<ConstantArrayPattern>(ctx, module, counter);
    LLVM_DEBUG(llvm::dbgs() << "Before globalizing array values:\n"
                            << module << '\n');
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After globalizing array values:\n"
                            << module << '\n');
  }
};
} // namespace
