/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#define GEN_PASS_DEF_CONSTANTPROPAGATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "constant-propagation"

using namespace mlir;

/// Common code to determine if \p load is from a reified span over a constant
/// array. See the examples below in the `LoadOp` patterns. If \p loadSpan is
/// `true`, the load is loading a `!cc.stdvec<T>` type value, otherwise load
/// must \e not load a `!cc.stdvec<T>` type value.
static LogicalResult
checkIfLoadFromReifiedSpan(cudaq::cc::ConstantArrayOp &conArr,
                           SmallVectorImpl<std::int32_t> &indices,
                           cudaq::cc::LoadOp load, bool loadSpan) {
  Value ptrVal = load.getPtrvalue();
  auto dataOp = [&]() -> cudaq::cc::StdvecDataOp {
    if (auto cast = ptrVal.getDefiningOp<cudaq::cc::CastOp>()) {
      indices.push_back(0);
      return cast.getValue().getDefiningOp<cudaq::cc::StdvecDataOp>();
    }
    if (auto comp = ptrVal.getDefiningOp<cudaq::cc::ComputePtrOp>()) {
      if (comp.getNumOperands() != 1)
        return {};
      indices.append(comp.getRawConstantIndices().begin(),
                     comp.getRawConstantIndices().end());
      return comp.getBase().getDefiningOp<cudaq::cc::StdvecDataOp>();
    }
    return {};
  }();
  if (!dataOp)
    return failure();
  auto reify = dataOp.getStdvec().getDefiningOp<cudaq::cc::ReifySpanOp>();
  if (!reify)
    return failure();
  conArr = reify.getElements().getDefiningOp<cudaq::cc::ConstantArrayOp>();
  if (!conArr)
    return failure();

  // Verify that the data type loaded is a match.
  return success(loadSpan == isa<cudaq::cc::StdvecType>(load.getType()));
}

namespace {
// If this is reifying a multidimensional array, then we want to pare down the
// dimensionality by forwarding the subarrays to the using load ops.
//
//   %0 = cc.const_array [["XY", "ZI"], ["IZ", "YX"]] :
//           !cc.array<!cc.array<!cc.array<i8 x 3> x 2> x 2>
//   %1 = cc.reify_span %0 : (!cc.array<!cc.array<!cc.array<i8 x 3> x 2> x 2>)
//           -> !cc.stdvec<!cc.stdvec<!cc.charspan>>
//   %2 = cc.stdvec_data %1 : (!cc.stdvec<!cc.stdvec<!cc.charspan>>) ->
//           !cc.ptr<!cc.array<!cc.stdvec<!cc.charspan> x ?>>
//   %3 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<!cc.stdvec<!cc.charspan>
//           x ?>>) -> !cc.ptr<!cc.stdvec<!cc.charspan>>
//   %4 = cc.load %3 : !cc.ptr<!cc.stdvec<!cc.charspan>>
//   ─────────────────────────────────────────────────────────────────────────
//   %0 = cc.const_array [["XY", "ZI"], ["IZ", "YX"]] :
//           !cc.array<!cc.array<!cc.array<i8 x 3> x 2> x 2>
//   %1 = cc.reify_span %0 : (!cc.array<!cc.array<!cc.array<i8 x 3> x 2> x 2>)
//           -> !cc.stdvec<!cc.stdvec<!cc.charspan>>
//   %2 = cc.stdvec_data %1 : (!cc.stdvec<!cc.stdvec<!cc.charspan>>) ->
//           !cc.ptr<!cc.array<!cc.stdvec<!cc.charspan> x ?>>
//   %3 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<!cc.stdvec<!cc.charspan>
//           x ?>>) -> !cc.ptr<!cc.stdvec<!cc.charspan>>
//   %a = cc.const_array ["IZ", "YX"] : !cc.array<!cc.array<i8 x 3> x 2>
//   %4 = cc.reify_span %a : (!cc.array<!cc.array<i8 x 3> x 2>)
//           -> !cc.stdvec<!cc.charspan>
//
class ForwardConstSubArray : public OpRewritePattern<cudaq::cc::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::LoadOp loadSpan,
                                PatternRewriter &rewriter) const override {
    cudaq::cc::ConstantArrayOp conArr;
    SmallVector<std::int32_t> indices;
    if (failed(checkIfLoadFromReifiedSpan(conArr, indices, loadSpan,
                                          /*loadSpan=*/true)))
      return failure();

    // The preconditions are met. At this point, we replace this load with a
    // pair of new ops: a constant array containing the subarray from
    // conArr[indices] and a new reify op.
    ArrayAttr aa = conArr.getConstantValues();
    Attribute attr;
    Type ty = conArr.getType();
    for (auto idx : indices) {
      attr = aa[idx];
      aa = cast<ArrayAttr>(attr);
      ty = cast<cudaq::cc::ArrayType>(ty).getElementType();
    }
    Type loadTy = loadSpan.getType();
    auto arrayAttr = cast<ArrayAttr>(attr);
    Value newConArr = rewriter.create<cudaq::cc::ConstantArrayOp>(
        loadSpan.getLoc(), ty, arrayAttr);
    rewriter.replaceOpWithNewOp<cudaq::cc::ReifySpanOp>(loadSpan, loadTy,
                                                        newConArr);
    return success();
  }
};

// If this is reifying a single dimension array, then we want to forward the
// constant element itself. Note that we do not count the innermost array if the
// data is a string, so in this case, there is a distinction between !cc.stdvec
// and !cc.charspan.
//
// Strings require two operations to get the types to check, as shown below.
//
//   %0 = cc.const_array ["IZ", "YX"] : !cc.array<!cc.array<i8 x 3> x 2>
//   %1 = cc.reify_span %0 : (!cc.array<!cc.array<i8 x 3> x 2>) ->
//           !cc.stdvec<!cc.charspan>
//   %2 = cc.stdvec_data %1 : (!cc.stdvec<!cc.charspan>) -> !cc.ptr<
//           !cc.array<!cc.charspan x ?>>
//   %3 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<!cc.charspan x ?>>) ->
//           !cc.ptr<!cc.charspan>
//   %4 = cc.load %3 : !cc.ptr<!cc.charspan>
//   ─────────────────────────────────────────────────────────────────────────
//   %0 = cc.const_array ["IZ", "YX"] : !cc.array<!cc.array<i8 x 3> x 2>
//   %1 = cc.reify_span %0 : (!cc.array<!cc.array<i8 x 3> x 2>) ->
//           !cc.stdvec<!cc.charspan>
//   %2 = cc.stdvec_data %1 : (!cc.stdvec<!cc.charspan>) -> !cc.ptr<
//           !cc.array<!cc.charspan x ?>>
//   %3 = cc.compute_ptr %2[0] : (!cc.ptr<!cc.array<!cc.charspan x ?>>) ->
//           !cc.ptr<!cc.charspan>
//   %a = cc.string_literal "IZ" : !cc.ptr<!cc.array<i8 x 3>>
//   %4 = cc.stdvec_init %a, %c3 : (!cc.ptr<!cc.array<i8 x 3>>, i64) ->
//           !cc.charspan
//
// For a non-string array this looks like the following rewrite.
//
//   %0 = cc.const_array [1, 2, 4, 8] : !cc.array<i64 x 4>
//   %1 = cc.reify_span %0 : (!cc.array<i64 x 2>) -> !cc.stdvec<i64>
//   %2 = cc.stdvec_data %1 : (!cc.stdvec<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
//   %3 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<i64 x ?>>) -> !cc.ptr<i64>
//   %4 = cc.load %3 : !cc.ptr<i64>
//   ─────────────────────────────────────────────────────────────────────────
//   %0 = cc.const_array [1, 2, 4, 8] : !cc.array<i64 x 4>
//   %1 = cc.reify_span %0 : (!cc.array<i64 x 2>) -> !cc.stdvec<i64>
//   %2 = cc.stdvec_data %1 : (!cc.stdvec<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
//   %3 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<i64 x ?>>) -> !cc.ptr<i64>
//   %4 = cc.constant 2 : i64
//
class ForwardSingleDimensionData : public OpRewritePattern<cudaq::cc::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::LoadOp loadSpanEle,
                                PatternRewriter &rewriter) const override {
    cudaq::cc::ConstantArrayOp conArr;
    SmallVector<std::int32_t> indices;
    if (failed(checkIfLoadFromReifiedSpan(conArr, indices, loadSpanEle,
                                          /*loadSpan=*/false)))
      return failure();

    // The preconditions are met. At this point, we replace this load depending
    // on the type. If this is loading a charspan, we replace the load with a
    // string_literal from conArr[indices] and stdvec_init op. Otherwise, we
    // replace the load with the constant from conArr[indices].
    ArrayAttr aa = conArr.getConstantValues();
    auto numIndices = indices.size();
    Attribute attr;
    Type ty = conArr.getType();
    for (auto [i, idx] : llvm::enumerate(indices)) {
      attr = aa[idx];
      if (i < numIndices - 1)
        aa = cast<ArrayAttr>(attr);
      ty = cast<cudaq::cc::ArrayType>(ty).getElementType();
    }
    Type loadTy = loadSpanEle.getType();
    auto loc = loadSpanEle.getLoc();
    if (isa<cudaq::cc::CharspanType>(loadTy)) {
      auto stringAttr = cast<StringAttr>(attr);
      auto lit = rewriter.create<cudaq::cc::CreateStringLiteralOp>(
          loc, cudaq::cc::PointerType::get(ty), stringAttr);
      auto len = rewriter.create<arith::ConstantIntOp>(
          loc, stringAttr.getValue().size() + 1, 64);
      rewriter.replaceOpWithNewOp<cudaq::cc::StdvecInitOp>(loadSpanEle, loadTy,
                                                           lit, len);
      return success();
    }
    if (auto intTy = dyn_cast<IntegerType>(loadTy)) {
      auto intAttr = cast<IntegerAttr>(attr);
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(
          loadSpanEle, intAttr.getInt(), intTy);
      return success();
    }
    if (auto floatTy = dyn_cast<FloatType>(loadTy)) {
      auto floatAttr = cast<FloatAttr>(attr);
      rewriter.replaceOpWithNewOp<arith::ConstantFloatOp>(
          loadSpanEle, floatAttr.getValue(), floatTy);
      return success();
    }
    return failure();
  }
};

class ConstantPropagationPass
    : public cudaq::opt::impl::ConstantPropagationBase<
          ConstantPropagationPass> {
public:
  using ConstantPropagationBase::ConstantPropagationBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<ForwardSingleDimensionData, ForwardConstSubArray>(ctx);

    LLVM_DEBUG(llvm::dbgs() << "Before constant prop:\n" << func << '\n');

    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "After constant prop:\n" << func << '\n');
  }
};
} // namespace
