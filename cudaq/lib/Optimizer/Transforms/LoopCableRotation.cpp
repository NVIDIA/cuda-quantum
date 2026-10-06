/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPCABLEROTATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "loop-cable-rotation"

using namespace mlir;

/**
   \file

   See the `LoopCableRotation` description in Passes.td for the motivation.

   This pass is independent of `cable-rough-in` and does not modify it:
   `cable-rough-in` converts a whole `veq` passed to a *call* into wire form;
   this pass converts a `veq` walked qubit-by-qubit by a counted loop's own
   `extract_ref`. Nor does this pass linearize the gate ops inside the loop
   body itself - the detached wire is bound to a fresh reference with
   `quake.wrap_new`, so every existing use of the old `extract_ref` result
   keeps working completely unchanged in reference form; the `memtoreg` pass,
   run afterwards, threads that fresh reference into proper wire SSA form the
   same way it already does elsewhere in the pipeline.
 */

namespace {

struct MatchedLoop {
  cudaq::quake::ExtractRefOp extractRef;
  Value veq;            // the statically sized veq (relax_size stripped if any)
  std::size_t size = 0; // C
};

// Recognize: a compile-time counted loop, no early exit, whose single-block
// body contains exactly one `extract_ref` indexed by the loop's own
// induction variable on a statically sized veq whose size matches the trip
// count, and that veq has no other use anywhere in the loop.
static std::optional<MatchedLoop> matchLoop(cudaq::cc::LoopOp loop) {
  if (!cudaq::opt::isaCountedLoop(loop))
    return std::nullopt;
  auto components = cudaq::opt::getLoopComponents(loop);
  if (!components || !components->induction)
    return std::nullopt;
  auto tripCount = components->getIterationsConstant();
  if (!tripCount || *tripCount == 0)
    return std::nullopt;

  Region &body = loop.getBodyRegion();
  if (!body.hasOneBlock())
    return std::nullopt;
  Block &bodyBlock = body.front();
  unsigned inductionIdx = *components->induction;
  if (inductionIdx >= bodyBlock.getNumArguments())
    return std::nullopt;
  Value inductionArg = bodyBlock.getArgument(inductionIdx);

  cudaq::quake::ExtractRefOp match;
  for (auto extractOp : bodyBlock.getOps<cudaq::quake::ExtractRefOp>()) {
    if (match)
      return std::nullopt; // more than one extract_ref in the body
    match = extractOp;
  }
  if (!match || match.getIndex() != inductionArg)
    return std::nullopt;

  Value veq = match.getVeq();
  Value sizedVeq = veq;
  if (auto relax = veq.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    sizedVeq = relax.getInputVec();
  auto veqTy = dyn_cast<cudaq::quake::VeqType>(sizedVeq.getType());
  if (!veqTy || !veqTy.hasSpecifiedSize() || veqTy.getSize() != *tripCount)
    return std::nullopt;

  // Escape check: `veq` (the value extract_ref actually operates on, before
  // any relax_size-stripping) must have no other use reachable from
  // anywhere inside the loop besides `match` itself.
  auto isInsideLoop = [&](Operation *op) {
    for (Operation *p = op; p; p = p->getParentOp())
      if (p == loop.getOperation())
        return true;
    return false;
  };
  for (OpOperand &use : veq.getUses())
    if (use.getOwner() != match.getOperation() && isInsideLoop(use.getOwner()))
      return std::nullopt;

  return MatchedLoop{match, sizedVeq, veqTy.getSize()};
}

// Append `passthrough` as the trailing operand of every `cc.condition` in
// `region` (the while region always has exactly one block).
static void appendConditionOperand(Region &region, Value passthrough,
                                   PatternRewriter &rewriter) {
  auto cond = cast<cudaq::cc::ConditionOp>(region.front().back());
  SmallVector<Value> results(cond.getResults());
  results.push_back(passthrough);
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(cond);
  rewriter.replaceOpWithNewOp<cudaq::cc::ConditionOp>(cond, cond.getCondition(),
                                                      results);
}

// Append `passthrough` as the trailing operand of every `cc.continue` that
// terminates a block with no successors in `region` (step/else may, in
// principle, have more than one block).
static void appendContinueOperands(Region &region, Value passthrough,
                                   PatternRewriter &rewriter) {
  for (auto &block : region)
    if (block.hasNoSuccessors())
      if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(block.getTerminator())) {
        SmallVector<Value> operands(cont.getOperands());
        operands.push_back(passthrough);
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(cont);
        rewriter.replaceOpWithNewOp<cudaq::cc::ContinueOp>(cont, operands);
      }
}

class ThreadCablePattern : public OpRewritePattern<cudaq::cc::LoopOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    auto matched = matchLoop(loop);
    if (!matched)
      return failure();

    auto loc = loop.getLoc();
    auto *ctx = rewriter.getContext();
    std::size_t C = matched->size;
    Value veq = matched->veq;
    auto refTy = cudaq::quake::RefType::get(ctx);
    auto wireTy = cudaq::quake::WireType::get(ctx);
    auto cableTy = cudaq::quake::CableType::get(ctx, C);
    auto cableMinusOneTy = cudaq::quake::CableType::get(ctx, C - 1);

    // Before the loop: unwrap every ref of `veq`, in order, and bundle them
    // into the initial cable.
    rewriter.setInsertionPoint(loop);
    SmallVector<Value> refs;
    SmallVector<Value> wires;
    for (std::size_t j = 0; j < C; ++j) {
      auto ref = cudaq::quake::ExtractRefOp::create(rewriter, loc, veq, j);
      refs.push_back(ref);
      wires.push_back(
          cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy, ref));
    }
    Value cable0 =
        cudaq::quake::BundleCableOp::create(rewriter, loc, cableTy, wires);

    // Add a trailing cable-typed block argument to every non-empty region's
    // entry block.
    for (Region *region : loop.getRegions())
      if (!region->empty())
        region->front().addArgument(cableTy, loc);

    // Body: replace the matched extract_ref with detach_wire/wrap_new, and
    // append unwrap/attach_wire just before the region's terminator.
    Block &bodyBlock = loop.getBodyRegion().front();
    Value cableArg = bodyBlock.getArguments().back();
    rewriter.setInsertionPoint(matched->extractRef);
    auto detach = cudaq::quake::DetachWireOp::create(
        rewriter, loc, wireTy, cableMinusOneTy, cableArg, /*index=*/0);
    auto wrapNew = cudaq::quake::WrapNewOp::create(rewriter, loc, refTy,
                                                   detach.getWireOut());
    rewriter.replaceOp(matched->extractRef, wrapNew.getResult());

    auto bodyContinue = cast<cudaq::cc::ContinueOp>(bodyBlock.getTerminator());
    rewriter.setInsertionPoint(bodyContinue);
    auto unwrapBack = cudaq::quake::UnwrapOp::create(rewriter, loc, wireTy,
                                                     wrapNew.getResult());
    auto attach = cudaq::quake::AttachWireOp::create(
        rewriter, loc, cableTy, unwrapBack.getResult(), detach.getCableOut(),
        /*index=*/C - 1);
    SmallVector<Value> bodyOperands(bodyContinue.getOperands());
    bodyOperands.push_back(attach.getResult());
    rewriter.replaceOpWithNewOp<cudaq::cc::ContinueOp>(bodyContinue,
                                                       bodyOperands);

    // While/step/else: the cable simply passes through unchanged.
    appendConditionOperand(loop.getWhileRegion(),
                           loop.getWhileRegion().front().getArguments().back(),
                           rewriter);
    if (loop.hasStep())
      appendContinueOperands(loop.getStepRegion(),
                             loop.getStepRegion().front().getArguments().back(),
                             rewriter);
    if (loop.hasPythonElse())
      appendContinueOperands(loop.getElseRegion(),
                             loop.getElseRegion().front().getArguments().back(),
                             rewriter);

    // Rebuild the loop with the cable appended to its carried values, move
    // the (already-rewritten) regions across, and redirect the original
    // results.
    SmallVector<Value> newInitArgs(loop.getInitialArgs());
    newInitArgs.push_back(cable0);
    SmallVector<Type> newResultTypes(loop.getResultTypes());
    newResultTypes.push_back(cableTy);

    rewriter.setInsertionPoint(loop);
    auto newLoop = cudaq::cc::LoopOp::create(
        rewriter, loc, newResultTypes, newInitArgs, loop.isPostConditional(),
        [](OpBuilder &, Location, Region &) {},
        [](OpBuilder &, Location, Region &) {},
        /*stepBuilder=*/nullptr);
    newLoop->setDiscardableAttrs(loop->getDiscardableAttrDictionary());
    newLoop.getWhileRegion().takeBody(loop.getWhileRegion());
    newLoop.getBodyRegion().takeBody(loop.getBodyRegion());
    newLoop.getStepRegion().takeBody(loop.getStepRegion());
    newLoop.getElseRegion().takeBody(loop.getElseRegion());

    for (unsigned i = 0, n = loop.getNumResults(); i < n; ++i)
      loop.getResult(i).replaceAllUsesWith(newLoop.getResult(i));
    Value finalCable = newLoop.getResults().back();

    // After the loop: split the final cable and wrap each wire back to its
    // original ref - the cable is back in its original order after C
    // detach-front/attach-back cycles.
    rewriter.setInsertionPointAfter(newLoop);
    SmallVector<Type> wireTys(C, wireTy);
    auto split =
        cudaq::quake::SplitCableOp::create(rewriter, loc, wireTys, finalCable);
    for (auto [wire, ref] : llvm::zip(split.getResults(), refs))
      cudaq::quake::WrapOp::create(rewriter, loc, wire, ref);

    rewriter.eraseOp(loop);
    return success();
  }
};

class LoopCableRotationPass
    : public cudaq::opt::impl::LoopCableRotationBase<LoopCableRotationPass> {
public:
  using LoopCableRotationBase::LoopCableRotationBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ThreadCablePattern>(ctx);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
