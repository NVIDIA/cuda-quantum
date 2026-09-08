/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
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

// Collect the individual qubit references behind a `veq` argument. The cable's
// arity has to be known here, so the argument must resolve to a statically
// sized collection of references.
static LogicalResult collectVeqRefs(PatternRewriter &rewriter, Location loc,
                                    Value arg, SmallVectorImpl<Value> &refs) {
  auto refTy = cudaq::quake::RefType::get(rewriter.getContext());
  if (auto relax = arg.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    arg = relax.getInputVec();

  // A concat already names the references, so use them directly and thread the
  // wires back to the values the caller is holding.
  if (auto concat = arg.getDefiningOp<cudaq::quake::ConcatOp>()) {
    for (auto carg : concat.getTargets()) {
      if (carg.getType() != refTy) {
        LLVM_DEBUG(llvm::dbgs() << concat << " must have ref arguments.\n");
        return failure();
      }
      refs.push_back(carg);
    }
    return success();
  }

  // Otherwise any statically sized veq will do, such as a subveq with constant
  // bounds. Materialize a reference per qubit.
  auto veqTy = cast<cudaq::quake::VeqType>(arg.getType());
  if (!veqTy.hasSpecifiedSize()) {
    LLVM_DEBUG(llvm::dbgs() << arg << " does not have a static size.\n");
    return failure();
  }
  for (std::size_t i = 0, n = veqTy.getSize(); i < n; ++i)
    refs.push_back(cudaq::quake::ExtractRefOp::create(rewriter, loc, arg, i));
  return success();
}

// Whether a `veq` nested in a struq can be lowered. The struq case below
// bundles a member's references straight from a concat, so nothing else will
// do.
static LogicalResult checkStruqMember(Value arg) {
  auto refTy = cudaq::quake::RefType::get(arg.getType().getContext());
  if (arg.getType() == refTy)
    return success();
  if (!isa<cudaq::quake::VeqType>(arg.getType()))
    return failure();
  if (auto relax = arg.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    arg = relax.getInputVec();
  auto concat = arg.getDefiningOp<cudaq::quake::ConcatOp>();
  if (!concat)
    return failure();
  for (auto carg : concat.getTargets())
    if (carg.getType() != refTy)
      return failure();
  return success();
}

// Whether a quantum argument can be lowered to a cable. Checked without
// touching the IR, so that a call with an argument this pattern cannot handle
// is left exactly as it was. Creating operations first and failing afterwards
// would strand an unwrap with no matching wrap.
static LogicalResult checkQuantumArg(Value arg) {
  Type argTy = arg.getType();
  auto refTy = cudaq::quake::RefType::get(argTy.getContext());
  if (argTy == refTy)
    return success();
  if (isa<cudaq::quake::VeqType>(argTy)) {
    if (auto relax = arg.getDefiningOp<cudaq::quake::RelaxSizeOp>())
      arg = relax.getInputVec();
    if (auto concat = arg.getDefiningOp<cudaq::quake::ConcatOp>()) {
      for (auto carg : concat.getTargets())
        if (carg.getType() != refTy)
          return failure();
      return success();
    }
    auto veqTy = dyn_cast<cudaq::quake::VeqType>(arg.getType());
    return success(veqTy && veqTy.hasSpecifiedSize());
  }
  if (isa<cudaq::quake::StruqType>(argTy)) {
    auto mkStruq = arg.getDefiningOp<cudaq::quake::MakeStruqOp>();
    if (!mkStruq)
      return failure();
    for (auto member : mkStruq.getVeqs())
      if (failed(checkStruqMember(member)))
        return failure();
    return success();
  }
  return failure();
}

class CallPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    bool performRewrite = [&]() {
      for (auto arg : call.getOperands())
        if (cudaq::quake::isQuantumReferenceType(arg.getType()))
          return true;
      return false;
    }();
    if (!performRewrite) {
      LLVM_DEBUG(llvm::dbgs() << call << " is not a candidate.\n");
      return failure();
    }

    // Check every argument before creating anything, so a call this pattern
    // cannot handle is left untouched rather than half-rewritten.
    for (auto arg : call.getOperands())
      if (cudaq::quake::isQuantumReferenceType(arg.getType()) &&
          failed(checkQuantumArg(arg))) {
        LLVM_DEBUG(llvm::dbgs() << arg << " cannot be put in wire form.\n");
        return failure();
      }

    auto loc = call.getLoc();
    auto *ctx = rewriter.getContext();
    auto refTy = cudaq::quake::RefType::get(ctx);
    auto wireTy = cudaq::quake::WireType::get(ctx);

    // Walk arguments and map them to value types and keep track of the new wire
    // types in left-to-right order.
    SmallVector<Value> newArgs;
    // The references behind each veq argument, in argument order. Kept so the
    // wrap-back loop below does not have to re-derive them.
    SmallVector<SmallVector<Value>> veqRefs;
    const std::size_t origCoarity = call.getResultTypes().size();
    SmallVector<Type> resultTys{call.getResultTypes().begin(),
                                call.getResultTypes().end()};
    for (auto arg : call.getOperands()) {
      Type argTy = arg.getType();
      if (argTy == refTy) {
        newArgs.push_back(
            cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, arg));
        resultTys.push_back(wireTy);
        continue;
      }
      if (isa<cudaq::quake::VeqType>(argTy)) {
        SmallVector<Value> refs;
        if (failed(collectVeqRefs(rewriter, loc, arg, refs)))
          return failure();
        auto cableTy = cudaq::quake::CableType::get(ctx, refs.size());
        SmallVector<Value> unwraps;
        for (auto ref : refs)
          unwraps.push_back(
              cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, ref));
        newArgs.push_back(cudaq::quake::BundleCableOp::create(
            rewriter, loc, cableTy, unwraps));
        resultTys.push_back(cableTy);
        veqRefs.push_back(std::move(refs));
        continue;
      }
      if (isa<cudaq::quake::StruqType>(argTy)) {
        auto mkStruq = arg.getDefiningOp<cudaq::quake::MakeStruqOp>();
        if (!mkStruq) {
          LLVM_DEBUG(llvm::dbgs() << arg << " is not a make_struq.\n");
          return failure();
        }
        std::size_t cableSize = 0;
        SmallVector<Value> unwraps;
        for (auto strArg : mkStruq.getVeqs()) {
          auto strArgTy = strArg.getType();
          if (isa<cudaq::quake::RefType>(strArgTy)) {
            unwraps.push_back(
                cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, strArg));
            cableSize++;
            continue;
          }
          if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(strArgTy)) {
            if (auto relax = strArg.getDefiningOp<cudaq::quake::RelaxSizeOp>())
              strArg = relax.getInputVec();
            auto concat = strArg.getDefiningOp<cudaq::quake::ConcatOp>();
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
                  cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, carg));
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << strArg << " is not supported.\n");
          return failure();
        }
        auto cableTy = cudaq::quake::CableType::get(ctx, cableSize);
        newArgs.push_back(cudaq::quake::BundleCableOp::create(
            rewriter, loc, cableTy, unwraps));
        resultTys.push_back(cableTy);
        continue;
      }
      // Pass non-quantum arguments as-is.
      newArgs.push_back(arg);
    }

    // Create a quake.call_by_ref operation.
    auto callByRef = cudaq::quake::CallByRefOp::create(
        rewriter, loc, call.getCalleeAttr(), resultTys, newArgs);

    // Wrap the wires and cables.
    std::size_t i = origCoarity;
    std::size_t veqIdx = 0;
    SmallVector<Value> results{callByRef.getResults().begin(),
                               callByRef.getResults().end()};
    for (auto arg : call.getOperands()) {
      Type argTy = arg.getType();
      if (argTy == refTy) {
        cudaq::quake::WrapOp::create(rewriter, loc, results[i++], arg);
        continue;
      }
      if (isa<cudaq::quake::VeqType>(argTy)) {
        ArrayRef<Value> refs = veqRefs[veqIdx++];
        SmallVector<Type> wireTys(refs.size(), wireTy);
        auto split = cudaq::quake::SplitCableOp::create(rewriter, loc, wireTys,
                                                        results[i++]);
        for (auto [wire, ref] : llvm::zip(split.getResults(), refs))
          cudaq::quake::WrapOp::create(rewriter, loc, wire, ref);
      }
      if (isa<cudaq::quake::StruqType>(argTy)) {
        auto mkStruq = arg.getDefiningOp<cudaq::quake::MakeStruqOp>();
        const std::size_t cableSize =
            cast<cudaq::quake::CableType>(resultTys[i]).getSize();
        SmallVector<Type> wireTys(cableSize);
        std::fill(wireTys.begin(), wireTys.end(), wireTy);
        auto split = cudaq::quake::SplitCableOp::create(rewriter, loc, wireTys,
                                                        results[i++]);
        std::size_t j = 0;
        SmallVector<Value> splitResults{split.getResults().begin(),
                                        split.getResults().end()};
        for (auto strArg : mkStruq.getVeqs()) {
          auto strArgTy = strArg.getType();
          if (isa<cudaq::quake::RefType>(strArgTy)) {
            cudaq::quake::WrapOp::create(rewriter, loc, splitResults[j++],
                                         strArg);
            continue;
          }
          if (isa<cudaq::quake::VeqType>(strArgTy)) {
            if (auto relax = strArg.getDefiningOp<cudaq::quake::RelaxSizeOp>())
              strArg = relax.getInputVec();
            auto concat = strArg.getDefiningOp<cudaq::quake::ConcatOp>();
            SmallVector<Value> concatTargs{concat.getTargets().begin(),
                                           concat.getTargets().end()};
            for (std::size_t k = 0, K = concatTargs.size(); k < K; ++k)
              cudaq::quake::WrapOp::create(rewriter, loc, splitResults[j++],
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
    cudaq::quake::ExtractRefOp::getCanonicalizationPatterns(patterns, ctx);
    cudaq::quake::GetMemberOp::getCanonicalizationPatterns(patterns, ctx);
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // A call left in reference form would reach the backend with its qubits
    // never threaded through it.
    bool hasUnconvertedCall = false;
    funcOp.walk([&](func::CallOp call) {
      for (auto arg : call.getOperands())
        if (cudaq::quake::isQuantumReferenceType(arg.getType())) {
          call.emitOpError("cannot be put in wire form. The qubits passed to "
                           "a call must be a statically sized set");
          hasUnconvertedCall = true;
          break;
        }
    });
    if (hasUnconvertedCall)
      signalPassFailure();
  }
};
} // namespace
