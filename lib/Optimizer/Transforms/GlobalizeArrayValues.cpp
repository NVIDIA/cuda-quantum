/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
SmallVector<A> conversion(ArrayAttr seq) {
  SmallVector<A> result;
  for (auto v : seq) {
    B c = cast<B>(v);
    result.emplace_back(c.getValue());
  }
  return result;
}
template <>
SmallVector<std::complex<APFloat>>
conversion<std::complex<APFloat>, ArrayAttr>(ArrayAttr seq) {
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
    auto blockValues = conversion<std::complex<APFloat>, ArrayAttr>(arrAttr);
    auto dense = DenseElementsAttr::get(tensorTy, blockValues);
    irBuilder.genVectorOfConstants(loc, module, globalName, dense, eleTy);
  } else if (isa<FloatType>(eleTy)) {
    auto blockValues = conversion<APFloat, FloatAttr>(arrAttr);
    auto dense = DenseElementsAttr::get(tensorTy, blockValues);
    irBuilder.genVectorOfConstants(loc, module, globalName, dense, eleTy);
  } else if (isa<IntegerType>(eleTy)) {
    auto blockValues = conversion<APInt, IntegerAttr>(arrAttr);
    auto dense = DenseElementsAttr::get(tensorTy, blockValues);
    irBuilder.genVectorOfConstants(loc, module, globalName, dense, eleTy);
  } else {
    return failure();
  }
  return success();
}

namespace {
struct ConstantArrayPattern
    : public OpRewritePattern<cudaq::cc::ConstantArrayOp> {
  explicit ConstantArrayPattern(MLIRContext *ctx, ModuleOp module,
                                unsigned &counter)
      : OpRewritePattern{ctx}, module{module}, counter{counter} {}

  LogicalResult matchAndRewrite(cudaq::cc::ConstantArrayOp conarr,
                                PatternRewriter &rewriter) const override {
    if (!conarr->hasOneUse())
      return failure();
    auto store = dyn_cast<cudaq::cc::StoreOp>(*conarr->getUsers().begin());
    if (!store)
      return failure();
    auto alloca = store.getPtrvalue().getDefiningOp<cudaq::cc::AllocaOp>();
    if (!alloca)
      return failure();
    auto func = conarr->getParentOfType<func::FuncOp>();
    if (!func)
      return failure();
    std::string globalName =
        func.getName().str() + ".rodata_" + std::to_string(counter++);
    auto *ctx = rewriter.getContext();
    auto valueAttr = conarr.getConstantValues();
    auto eleTy = cast<cudaq::cc::ArrayType>(conarr.getType()).getElementType();
    if (failed(convertArrayAttrToGlobalConstant(ctx, conarr.getLoc(), valueAttr,
                                                module, globalName, eleTy)))
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::cc::AddressOfOp>(
        alloca, alloca.getType(), globalName);
    rewriter.eraseOp(store);
    rewriter.eraseOp(conarr);
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
