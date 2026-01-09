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
//  func.call @llvm.memcpy.p0i8.p0i8.i64(%36, %34, %35, %false) :
//      (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
//  %37 = cc.alloca i8[%35 : i64]
//  %38 = cc.cast %37 : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
//  func.call @llvm.memcpy.p0i8.p0i8.i64(%38, %36, %35, %false) :
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
      auto loaded = rewriter.create<cudaq::cc::LoadOp>(
          analysis.copyFrom.getLoc(), globalConst);
      rewriter.setInsertionPointAfter(analysis.copyTo);
      rewriter.create<cudaq::cc::StoreOp>(analysis.copyTo.getLoc(), loaded,
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
    rewriter.eraseOp(call);
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
    patterns.insert<EraseVectorCopyCtorPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
