/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASEVECTORCOPYCTOR
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-vector-copy-ctor"

using namespace mlir;

namespace {
struct PatternAnalysis {
  func::CallOp copyFrom;
  func::CallOp copyTo;
  func::CallOp freeMem;
};

// Transformation is:
//
//  %36 = func.call @malloc(%35) : (i64) -> !cc.ptr<i8>
//  func.call @llvm.memcpy.p0.p0.i64(%36, %34, %35, %false) :
//      (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
//  %37 = cc.alloca i8[%35 : i64]
//  %38 = cc.cast %37 : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
//  func.call @llvm.memcpy.p0.p0.i64(%38, %36, %35, %false) :
//      (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
//  func.call @free(%36) : (!cc.ptr<i8>) -> ()
//  ───────────────────────────────────────────────────────────────
//  [{ %37 \ %34 }]   // i.e., replace uses of %37 with %34
class EraseVectorCopyCtorPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (call.getCallee() != "malloc")
      return failure();
    PatternAnalysis analysis;
    if (!expectedUsers(analysis, call->getUsers(), call.getResult(0)))
      return failure();
    auto casted =
        analysis.copyTo.getOperand(0).getDefiningOp<cudaq::cc::CastOp>();
    if (!casted)
      return failure();
    auto newStackSlot = casted.getValue().getDefiningOp<cudaq::cc::AllocaOp>();
    if (!newStackSlot)
      return failure();
    auto source =
        analysis.copyFrom.getOperand(1).getDefiningOp<cudaq::cc::CastOp>();
    if (!source)
      return failure();
    auto globalConst =
        source.getValue().getDefiningOp<cudaq::cc::AddressOfOp>();
    if (globalConst) {
      auto ip = rewriter.saveInsertionPoint();
      rewriter.setInsertionPointAfter(analysis.copyFrom);
      auto loaded = cudaq::cc::LoadOp::create(
          rewriter, analysis.copyFrom.getLoc(), globalConst);
      rewriter.setInsertionPointAfter(analysis.copyTo);
      cudaq::cc::StoreOp::create(rewriter, analysis.copyTo.getLoc(), loaded,
                                 newStackSlot);
      rewriter.restoreInsertionPoint(ip);
    } else {
      rewriter.replaceOpWithNewOp<cudaq::cc::CastOp>(
          newStackSlot, newStackSlot.getType(),
          analysis.copyFrom.getOperand(1));
    }
    rewriter.eraseOp(analysis.copyFrom);
    rewriter.eraseOp(analysis.copyTo);
    rewriter.eraseOp(analysis.freeMem);
    rewriter.replaceOpWithNewOp<cudaq::cc::UndefOp>(
        call, call.getResult(0).getType());
    return success();
  }

  static bool expectedUsers(PatternAnalysis &analysis,
                            ResultRange::user_range users, Value val) {
    for (auto u : users) {
      auto call = dyn_cast<func::CallOp>(u);
      if (!call)
        return false;
      if (call.getCallee().starts_with("llvm.memcpy")) {
        if (call.getOperand(0) == val && !analysis.copyFrom) {
          analysis.copyFrom = call;
          continue;
        }
        if (call.getOperand(1) == val && !analysis.copyTo) {
          analysis.copyTo = call;
          continue;
        }
      }
      if (call.getCallee() == "free") {
        if (!analysis.freeMem) {
          analysis.freeMem = call;
          continue;
        }
      }
      return false;
    }
    return analysis.copyFrom && analysis.copyTo && analysis.freeMem;
  }
};

// When inlining a function that returns a span, we can have the span buffer and
// creation code captured in a scope block. In that case, we have to look at the
// user(s) of the scope's return value. If that checks, we have to hoist the
// span buffer to the parent scope.
//
// Transformation is:
//
//  %30 = cc.scope -> (!cc.stdvec<i1>) {
//    ...
//    %34 = cc.alloca i8[%15 : i64]
//    ...
//    %36 = func.call @malloc(%15) : (i64) -> !cc.ptr<i8>
//    func.call @llvm.memcpy.p0.p0.i64(%36, %34, %35, %false) :
//      (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> () [dead]
//    %37 = cc.stdvec_init %36, %15 : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
//    cc.continue %37
//  }
//  %37 = cc.stdvec_data %30 : (!stdvec<i1>) -> !cc.ptr<i8>
//  %38 = cc.stdvec_size %30 : (!stdvec<i1>) -> i64
//  %39 = cc.alloca i8[%38 : i64]
//  %40 = cc.cast %39 : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
//  func.call @llvm.memcpy.p0.p0.i64(%40, %37, %38, %false) :
//      (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
//  func.call @free(%37) : (!cc.ptr<i8>) -> ()
//  ───────────────────────────────────────────────────────────────
//  %29 = cc.alloca i8[%15 : i64]
//  %30 = cc.scope ...
//    ...
//  func.call @free(%37) : (!cc.ptr<i8>) -> () [dead]
//  [{ %34 \ %29;  %39 \ %29 }]   // i.e., replace %34 and %39 with %29
class EraseScopedVectorCopyCtorPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (call.getCallee() != "malloc")
      return failure();
    auto scope = dyn_cast_if_present<cudaq::cc::ScopeOp>(call->getParentOp());
    if (!(scope && scope.getResults().size() == 1 &&
          isa<cudaq::cc::StdvecType>(scope->getResult(0).getType()) &&
          std::distance(call->getUsers().begin(), call->getUsers().end()) == 2))
      return failure();

    // Check call (malloc) users.
    func::CallOp copyFrom;
    cudaq::cc::StdvecInitOp initOp;
    for (auto *u : call->getUsers()) {
      auto c = dyn_cast<func::CallOp>(u);
      if (c && c.getCallee().starts_with("llvm.memcpy") &&
          c.getOperand(0) == call.getResult(0) && !copyFrom) {
        copyFrom = c;
        continue;
      }
      auto io = dyn_cast<cudaq::cc::StdvecInitOp>(u);
      if (io && !initOp) {
        initOp = io;
        continue;
      }
      return failure();
    }

    // Check initOp users.
    if (!initOp->hasOneUse())
      return failure();
    auto contOp = dyn_cast<cudaq::cc::ContinueOp>(*initOp->getUsers().begin());
    if (!contOp)
      return failure();

    // Check scope users.
    if (std::distance(scope->getUsers().begin(), scope->getUsers().end()) != 2)
      return failure();
    cudaq::cc::StdvecDataOp data;
    cudaq::cc::StdvecSizeOp size;
    for (auto *u : scope->getUsers()) {
      auto d = dyn_cast<cudaq::cc::StdvecDataOp>(u);
      if (d && !data) {
        data = d;
        continue;
      }
      auto s = dyn_cast<cudaq::cc::StdvecSizeOp>(u);
      if (s && !size) {
        size = s;
        continue;
      }
      return failure();
    }

    // Check vec data users.
    if (std::distance(data->getUsers().begin(), data->getUsers().end()) != 2)
      return failure();
    func::CallOp copyTo;
    func::CallOp freeHeap;
    for (auto *u : data->getUsers()) {
      auto c = dyn_cast<func::CallOp>(u);
      if (c && c.getCallee().starts_with("llvm.memcpy") &&
          c.getOperand(1) == data.getResult() && !copyTo) {
        copyTo = c;
        continue;
      }
      if (c && c.getCallee() == "free" && c.getOperand(0) == data.getResult() &&
          !freeHeap) {
        freeHeap = c;
        continue;
      }
      return failure();
    }

    // Find the to-buffer.
    auto *buff = copyTo.getOperand(0).getDefiningOp();
    while (auto cast = dyn_cast<cudaq::cc::CastOp>(buff))
      buff = cast.getValue().getDefiningOp();
    auto toBuff = dyn_cast<cudaq::cc::AllocaOp>(buff);
    if (!toBuff)
      return failure();

    // Hoist the allocation.
    auto *alloc = copyFrom.getOperand(1).getDefiningOp();
    while (auto cast = dyn_cast<cudaq::cc::CastOp>(alloc))
      alloc = cast.getValue().getDefiningOp();
    auto al = dyn_cast<cudaq::cc::AllocaOp>(alloc);
    if (!al || al->getNumOperands())
      return failure();
    auto newAlloc = [&]() -> cudaq::cc::AllocaOp {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(scope);
      return rewriter.replaceOpWithNewOp<cudaq::cc::AllocaOp>(
          al, al.getElementType());
    }();

    // Replace the uses.
    rewriter.replaceOp(toBuff, newAlloc);

    // Remove dead calls.
    rewriter.eraseOp(copyFrom);
    rewriter.eraseOp(copyTo);
    rewriter.eraseOp(freeHeap);
    rewriter.replaceOpWithNewOp<cudaq::cc::UndefOp>(
        call, call->getResult(0).getType());
    return success();
  }
};

class EraseVectorCopyCtorPass
    : public cudaq::opt::impl::EraseVectorCopyCtorBase<
          EraseVectorCopyCtorPass> {
public:
  using EraseVectorCopyCtorBase::EraseVectorCopyCtorBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns
        .insert<EraseVectorCopyCtorPattern, EraseScopedVectorCopyCtorPattern>(
            ctx);
    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
