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
// quake.apply pattern.
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
//   %d = quake.apply @callee(%c, %cst) : (!quake.cable<2>, f32) ->
//                                        !quake.cable<2>
//   %e, %f = quake.split_cable %d : (!quake.cable<2>) ->
//                                    (!quake.wire, !quake.wire)
//   quake.wrap %e to %0 : !quake.wire, !quake.ref  // [%0/%4]
//   quake.wrap %f to %1 : !quake.wire, !quake.ref  // [%1/%5]
//
// The `quake.apply` op emitted here is a plain, unpredicated call (no adj, no
// controls), so `apply-op-specialization` resolves it straight to a direct
// call of the original callee -- a subsequent run of that pass (scheduled
// right after this one) is required to eliminate it before it reaches
// codegen.
//

namespace {

// Collect the individual quantum references behind a `veq` argument. The
// cable's arity has to be known here, so the argument must resolve to a
// statically sized collection of references.
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
  // bounds. Materialize one reference per element.
  auto veqTy = cast<cudaq::quake::VeqType>(arg.getType());
  if (!veqTy.hasSpecifiedSize()) {
    LLVM_DEBUG(llvm::dbgs() << arg << " does not have a static size.\n");
    return failure();
  }
  for (std::size_t i = 0, n = veqTy.getSize(); i < n; ++i)
    refs.push_back(cudaq::quake::ExtractRefOp::create(rewriter, loc, arg, i));
  return success();
}

static LogicalResult checkQuantumArg(Value arg);

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
      if (failed(checkQuantumArg(member)))
        return failure();
    return success();
  }
  return failure();
}

// Collect the quantum references behind a struq argument, member by member
// and in order.
static LogicalResult collectStruqRefs(PatternRewriter &rewriter, Location loc,
                                      Value arg, SmallVectorImpl<Value> &refs) {
  auto mkStruq = arg.getDefiningOp<cudaq::quake::MakeStruqOp>();
  if (!mkStruq) {
    LLVM_DEBUG(llvm::dbgs() << arg << " is not a make_struq.\n");
    return failure();
  }
  auto refTy = cudaq::quake::RefType::get(rewriter.getContext());
  for (auto member : mkStruq.getVeqs()) {
    if (member.getType() == refTy) {
      refs.push_back(member);
      continue;
    }
    if (failed(collectVeqRefs(rewriter, loc, member, refs)))
      return failure();
  }
  return success();
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
    // The references behind each cabled argument, in argument order. Kept so
    // the wrap-back loop below does not have to re-derive them.
    SmallVector<SmallVector<Value>> cableRefs;
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
        cableRefs.push_back(std::move(refs));
        continue;
      }
      if (isa<cudaq::quake::StruqType>(argTy)) {
        SmallVector<Value> refs;
        if (failed(collectStruqRefs(rewriter, loc, arg, refs)))
          return failure();
        auto cableTy = cudaq::quake::CableType::get(ctx, refs.size());
        SmallVector<Value> unwraps;
        for (auto ref : refs)
          unwraps.push_back(
              cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, ref));
        newArgs.push_back(cudaq::quake::BundleCableOp::create(
            rewriter, loc, cableTy, unwraps));
        resultTys.push_back(cableTy);
        cableRefs.push_back(std::move(refs));
        continue;
      }
      // Pass non-quantum arguments as-is.
      newArgs.push_back(arg);
    }

    // Create a quake.apply operation.
    auto apply = cudaq::quake::ApplyOp::create(rewriter, loc, resultTys,
                                               call.getCallee(), newArgs);

    // Wrap the wires and cables.
    std::size_t i = origCoarity;
    std::size_t cableIdx = 0;
    SmallVector<Value> results{apply.getResults().begin(),
                               apply.getResults().end()};
    for (auto arg : call.getOperands()) {
      Type argTy = arg.getType();
      if (argTy == refTy) {
        cudaq::quake::WrapOp::create(rewriter, loc, results[i++], arg);
        continue;
      }
      if (isa<cudaq::quake::VeqType>(argTy) ||
          isa<cudaq::quake::StruqType>(argTy)) {
        ArrayRef<Value> refs = cableRefs[cableIdx++];
        SmallVector<Type> wireTys(refs.size(), wireTy);
        auto split = cudaq::quake::SplitCableOp::create(rewriter, loc, wireTys,
                                                        results[i++]);
        for (auto [wire, ref] : llvm::zip(split.getResults(), refs))
          cudaq::quake::WrapOp::create(rewriter, loc, wire, ref);
      }
    }

    rewriter.replaceOp(
        call, apply.getResults().drop_back(resultTys.size() - origCoarity));
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
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
