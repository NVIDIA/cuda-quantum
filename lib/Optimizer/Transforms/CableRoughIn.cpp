/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CABLEROUGHIN
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cable-rough-in"

using namespace mlir;

// Convert the quake.concat and func.call pattern into a quake.bundle_cable and
// quake.call_by_ref pattern.
//
//   %2 = quake.concat %0, %1 : (!quake.ref, !quake.ref) -> !quake.veq<2>
//   %3 = quake.relax_size %2 : (!quake.veq<2>) -> !quake.veq<?>
//   call @callee(%3, %cst) : (!quake.veq<?>, f32) -> ()
//   %4 = quake.extract_ref %2[0] : (!quake.veq<2>) -> !quake.ref
//   %5 = quake.extract_ref %2[1] : (!quake.veq<2>) -> !quake.ref
//   ――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――――
//   %a = quake.unwrap %0 : (!quake.ref) -> !quake.wire
//   %b = quake.unwrap %1 : (!quake.ref) -> !quake.wire
//   %c = quake.bundle_cable %a, %b : (!quake.wire, !quake.wire) ->
//                                     !quake.cable<2>
//   %d = quake.call_by_ref @callee(%c, %cst) : (!quake.cable<2>, f32) ->
//                                               !quake.cable<2>
//   %e, %f = quake.split_cable %d : (!quake.cable<2>) ->
//                                    (!quake.wire, !quake.wire)
//   quake.wrap %e to %0 : !quake.wire, !quake.ref  // [%0/%4]
//   quake.wrap %f to %1 : !quake.wire, !quake.ref  // [%1/%5]
//

namespace {

class CallPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    bool performRewrite = [&]() {
      for (auto arg : call.getOperands())
        if (quake::isQuantumReferenceType(arg.getType()))
          return true;
      return false;
    }();
    if (!performRewrite) {
      LLVM_DEBUG(llvm::dbgs() << call << " is not a candidate.\n");
      return failure();
    }

    auto loc = call.getLoc();
    auto *ctx = rewriter.getContext();
    auto refTy = quake::RefType::get(ctx);
    auto wireTy = quake::WireType::get(ctx);

    // Walk arguments and map them to value types and keep track of the new wire
    // types in left-to-right order.
    SmallVector<Value> newArgs;
    const std::size_t origCoarity = call.getResultTypes().size();
    SmallVector<Type> resultTys{call.getResultTypes().begin(),
                                call.getResultTypes().end()};
    for (auto arg : call.getOperands()) {
      Type argTy = arg.getType();
      if (argTy == refTy) {
        newArgs.push_back(rewriter.create<quake::UnwrapOp>(loc, wireTy, arg));
        resultTys.push_back(wireTy);
        continue;
      }
      if (isa<quake::VeqType>(argTy)) {
        // Cases we handle are concat or concat + relax_size.
        if (auto relax = arg.getDefiningOp<quake::RelaxSizeOp>())
          arg = relax.getInputVec();
        auto concat = arg.getDefiningOp<quake::ConcatOp>();
        if (!concat) {
          LLVM_DEBUG(llvm::dbgs() << arg << " is not a concat.\n");
          return failure();
        }
        for (auto carg : concat.getTargets())
          if (carg.getType() != refTy) {
            LLVM_DEBUG(llvm::dbgs() << concat << " must have ref arguments.\n");
            return failure();
          }
        const std::size_t cableSize = concat.getTargets().size();
        auto cableTy = quake::CableType::get(ctx, cableSize);
        SmallVector<Value> unwraps;
        for (auto carg : concat.getTargets())
          unwraps.push_back(
              rewriter.create<quake::UnwrapOp>(loc, wireTy, carg));
        newArgs.push_back(
            rewriter.create<quake::BundleCableOp>(loc, cableTy, unwraps));
        resultTys.push_back(cableTy);
        continue;
      }
      if (isa<quake::StruqType>(argTy)) {
        auto mkStruq = arg.getDefiningOp<quake::MakeStruqOp>();
        if (!mkStruq) {
          LLVM_DEBUG(llvm::dbgs() << arg << " is not a make_struq.\n");
          return failure();
        }
        std::size_t cableSize = 0;
        SmallVector<Value> unwraps;
        for (auto strArg : mkStruq.getVeqs()) {
          auto strArgTy = strArg.getType();
          if (isa<quake::RefType>(strArgTy)) {
            unwraps.push_back(
                rewriter.create<quake::UnwrapOp>(loc, wireTy, strArg));
            cableSize++;
            continue;
          }
          if (auto veqTy = dyn_cast<quake::VeqType>(strArgTy)) {
            if (auto relax = strArg.getDefiningOp<quake::RelaxSizeOp>())
              strArg = relax.getInputVec();
            auto concat = strArg.getDefiningOp<quake::ConcatOp>();
            if (!concat) {
              LLVM_DEBUG(llvm::dbgs() << arg << " is not a concat.\n");
              return failure();
            }
            for (auto carg : concat.getTargets())
              if (carg.getType() != refTy) {
                LLVM_DEBUG(llvm::dbgs()
                           << concat << " must have ref arguments.\n");
                return failure();
              }
            cableSize += concat.getTargets().size();
            for (auto carg : concat.getTargets())
              unwraps.push_back(
                  rewriter.create<quake::UnwrapOp>(loc, wireTy, carg));
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << strArg << " is not supported.\n");
          return failure();
        }
        auto cableTy = quake::CableType::get(ctx, cableSize);
        newArgs.push_back(
            rewriter.create<quake::BundleCableOp>(loc, cableTy, unwraps));
        resultTys.push_back(cableTy);
        continue;
      }
      // Pass non-quantum arguments as-is.
      newArgs.push_back(arg);
    }

    // Create a quake.call_by_ref operation.
    auto callByRef = rewriter.create<quake::CallByRefOp>(
        loc, resultTys, call.getCalleeAttr(), newArgs);

    // Wrap the wires and cables.
    std::size_t i = origCoarity;
    SmallVector<Value> results{callByRef.getResults().begin(),
                               callByRef.getResults().end()};
    for (auto arg : call.getOperands()) {
      Type argTy = arg.getType();
      if (argTy == refTy) {
        rewriter.create<quake::WrapOp>(loc, results[i++], arg);
        continue;
      }
      if (isa<quake::VeqType>(argTy)) {
        if (auto relax = arg.getDefiningOp<quake::RelaxSizeOp>())
          arg = relax.getInputVec();
        auto concat = arg.getDefiningOp<quake::ConcatOp>();
        const std::size_t cableSize =
            cast<quake::CableType>(resultTys[i]).getSize();
        SmallVector<Type> wireTys(cableSize);
        std::fill(wireTys.begin(), wireTys.end(), wireTy);
        auto split =
            rewriter.create<quake::SplitCableOp>(loc, wireTys, results[i++]);
        SmallVector<Value> concatTargs{concat.getTargets().begin(),
                                       concat.getTargets().end()};
        for (auto [j, wire] : llvm::enumerate(split.getResults()))
          rewriter.create<quake::WrapOp>(loc, wire, concatTargs[j]);
      }
      if (isa<quake::StruqType>(argTy)) {
        auto mkStruq = arg.getDefiningOp<quake::MakeStruqOp>();
        const std::size_t cableSize =
            cast<quake::CableType>(resultTys[i]).getSize();
        SmallVector<Type> wireTys(cableSize);
        std::fill(wireTys.begin(), wireTys.end(), wireTy);
        auto split =
            rewriter.create<quake::SplitCableOp>(loc, wireTys, results[i++]);
        std::size_t j = 0;
        SmallVector<Value> splitResults{split.getResults().begin(),
                                        split.getResults().end()};
        for (auto strArg : mkStruq.getVeqs()) {
          auto strArgTy = strArg.getType();
          if (isa<quake::RefType>(strArgTy)) {
            rewriter.create<quake::WrapOp>(loc, splitResults[j++], strArg);
            continue;
          }
          if (isa<quake::VeqType>(strArgTy)) {
            if (auto relax = strArg.getDefiningOp<quake::RelaxSizeOp>())
              strArg = relax.getInputVec();
            auto concat = strArg.getDefiningOp<quake::ConcatOp>();
            SmallVector<Value> concatTargs{concat.getTargets().begin(),
                                           concat.getTargets().end()};
            for (std::size_t k = 0, K = concatTargs.size(); k < K; ++k)
              rewriter.create<quake::WrapOp>(loc, splitResults[j++],
                                             concatTargs[k]);
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << strArg << " is not supported.\n");
          return failure();
        }
      }
    }

    rewriter.replaceOp(
        call, callByRef.getResults().drop_back(resultTys.size() - origCoarity));
    return success();
  }
};

class CableRoughInPass
    : public cudaq::opt::impl::CableRoughInBase<CableRoughInPass> {
public:
  using CableRoughInBase::CableRoughInBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    patterns.insert<CallPattern>(ctx);
    quake::ExtractRefOp::getCanonicalizationPatterns(patterns, ctx);
    quake::GetMemberOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
