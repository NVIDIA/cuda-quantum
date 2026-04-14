/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"

namespace quake::canonical {

inline mlir::Value createCast(mlir::PatternRewriter &rewriter,
                              mlir::Location loc, mlir::Value inVal) {
  auto i64Ty = rewriter.getI64Type();
  assert(inVal.getType() != rewriter.getIndexType() &&
         "use of index type is deprecated");
  return cudaq::cc::CastOp::create(rewriter, loc, i64Ty, inVal,
                                   cudaq::cc::CastOpMode::Unsigned);
}

class ExtractRefFromSubVeqPattern
    : public mlir::OpRewritePattern<ExtractRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Replace a pattern such as:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %2 = quake.subveq %1, %c2, %c3 : (!quake.veq<4>, i32, i32) ->
  //        !quake.veq<2>
  //   %3 = quake.extract_ref %2[0] : (!quake.veq<2>) -> !quake.ref
  // ```
  // with:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %3 = quake.extract_ref %1[2] : (!uwake.veq<4>) -> !quake.ref
  // ```
  mlir::LogicalResult
  matchAndRewrite(ExtractRefOp extract,
                  mlir::PatternRewriter &rewriter) const override {
    auto subveq = extract.getVeq().getDefiningOp<SubVeqOp>();
    if (!subveq)
      return mlir::failure();
    // Let the combining of back-to-back subveq ops happen first.
    if (isa<SubVeqOp>(subveq.getVeq().getDefiningOp()))
      return mlir::failure();

    mlir::Value offset;
    auto loc = extract.getLoc();
    auto low = [&]() -> mlir::Value {
      if (subveq.hasConstantLowerBound())
        return mlir::arith::ConstantIntOp::create(
            rewriter, loc, rewriter.getIntegerType(64),
            subveq.getConstantLowerBound());
      return subveq.getLower();
    }();
    if (extract.hasConstantIndex()) {
      mlir::Value cv = mlir::arith::ConstantIntOp::create(
          rewriter, loc, low.getType(), extract.getConstantIndex());
      offset = mlir::arith::AddIOp::create(rewriter, loc, cv, low);
    } else {
      auto cast1 = createCast(rewriter, loc, extract.getIndex());
      auto cast2 = createCast(rewriter, loc, low);
      offset = mlir::arith::AddIOp::create(rewriter, loc, cast1, cast2);
    }
    rewriter.replaceOpWithNewOp<ExtractRefOp>(extract, subveq.getVeq(), offset);
    return mlir::success();
  }
};

// Combine back-to-back quake.subveq operations.
//
// %10 = quake.subveq %4, 1, 6 : (!quake.veq<?>) -> !quake.veq<7>
// %11 = quake.subveq %10, 0, 2 : (!quake.veq<7>) -> !quake.veq<3>
// ───────────────────────────────────────────────────────────────
// %11 = quake.subveq %4, 1, 3 : (!quake.veq<?>) -> !quake.veq<3>
class CombineSubVeqsPattern : public mlir::OpRewritePattern<SubVeqOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SubVeqOp subveq,
                  mlir::PatternRewriter &rewriter) const override {
    auto prior = subveq.getVeq().getDefiningOp<SubVeqOp>();
    if (!prior)
      return mlir::failure();

    auto loc = subveq.getLoc();

    // Lambda to create a Value for the lower bound of `s`.
    auto lofunc = [&](SubVeqOp s) -> mlir::Value {
      if (s.hasConstantLowerBound())
        return mlir::arith::ConstantIntOp::create(rewriter, loc,
                                                  rewriter.getIntegerType(64),
                                                  s.getConstantLowerBound());
      return s.getLower();
    };
    auto priorlo = lofunc(prior);
    auto svlo = lofunc(subveq);

    // Lambda for creating the upper bound Value.
    auto svup = [&]() -> mlir::Value {
      if (subveq.hasConstantUpperBound())
        return mlir::arith::ConstantIntOp::create(
            rewriter, loc, rewriter.getIntegerType(64),
            subveq.getConstantUpperBound());
      return subveq.getUpper();
    }();
    auto cast1 = createCast(rewriter, loc, priorlo);
    auto cast2 = createCast(rewriter, loc, svlo);
    auto cast3 = createCast(rewriter, loc, svup);
    mlir::Value sum1 = mlir::arith::AddIOp::create(rewriter, loc, cast1, cast2);
    mlir::Value sum2 = mlir::arith::AddIOp::create(rewriter, loc, cast1, cast3);
    auto veqTy = subveq.getType();
    rewriter.replaceOpWithNewOp<SubVeqOp>(subveq, veqTy, prior.getVeq(), sum1,
                                          sum2);
    return mlir::success();
  }
};

} // namespace quake::canonical
