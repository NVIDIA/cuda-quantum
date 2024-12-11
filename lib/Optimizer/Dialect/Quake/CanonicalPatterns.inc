/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// These canonicalization patterns are used by the canonicalize pass and not
// shared for other uses. Generally speaking, these patterns should be trivial
// peephole optimizations that reduce the size and complexity of the input IR.

// This file must be included after a `using namespace mlir;` as it uses bare
// identifiers from that namespace.

namespace {

// %4 = quake.veq_size %3 : (!quake.veq<10>) -> 164
// ────────────────────────────────────────────────
// %4 = constant 10 : i64
struct ForwardConstantVeqSizePattern
    : public OpRewritePattern<quake::VeqSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::VeqSizeOp veqSize,
                                PatternRewriter &rewriter) const override {
    auto veqTy = dyn_cast<quake::VeqType>(veqSize.getVeq().getType());
    if (!veqTy)
      return failure();
    if (!veqTy.hasSpecifiedSize())
      return failure();
    auto resTy = veqSize.getType();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(veqSize, veqTy.getSize(),
                                                      resTy);
    return success();
  }
};

// %2 = constant 10 : i32
// %3 = quake.alloca !quake.veq<?>[%2 : i32]
// ─────────────────────────────────────────
// %3 = quake.alloca !quake.veq<10>
struct FuseConstantToAllocaPattern : public OpRewritePattern<quake::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    auto size = alloc.getSize();
    if (!size)
      return failure();
    auto intCon = cudaq::opt::factory::getIntIfConstant(size);
    if (!intCon)
      return failure();
    auto veqTy = dyn_cast<quake::VeqType>(alloc.getType());
    if (!veqTy)
      return failure();
    if (veqTy.hasSpecifiedSize())
      return failure();
    auto loc = alloc.getLoc();
    auto resTy = alloc.getType();
    auto newAlloc = rewriter.create<quake::AllocaOp>(
        loc, static_cast<std::size_t>(*intCon));
    rewriter.replaceOpWithNewOp<quake::RelaxSizeOp>(alloc, resTy, newAlloc);
    return success();
  }
};

// %2 = constant 10 : i32
// %3 = quake.extract_ref %1[%2] : (!quake.veq<?>, i32) -> !quake.ref
// ──────────────────────────────────────────────────────────────────
// %3 = quake.extract_ref %1[10] : (!quake.veq<?>) -> !quake.ref
struct FuseConstantToExtractRefPattern
    : public OpRewritePattern<quake::ExtractRefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    auto index = extract.getIndex();
    if (!index)
      return failure();
    auto intCon = cudaq::opt::factory::getIntIfConstant(index);
    if (!intCon)
      return failure();
    rewriter.replaceOpWithNewOp<quake::ExtractRefOp>(
        extract, extract.getVeq(), static_cast<std::size_t>(*intCon));
    return success();
  }
};

} // namespace
