/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// These patterns are used by classical-optimization pass and cc-loop-unroll
// pass.

// This file must be included after a `using namespace mlir;` as it uses bare
// identifiers from that namespace.

inline std::pair<Block *, Block *> findCloneRange(Block *first, Block *last) {
  return {first->getNextNode(), last->getPrevNode()};
}

static std::size_t
unrollLoopByValue(cudaq::cc::LoopOp loop,
                  const cudaq::opt::LoopComponents &components) {
  auto c = components.compareValue.getDefiningOp<arith::ConstantOp>();
  if (loop->hasAttr(cudaq::opt::NormalizedLoopAttr))
    return cast<IntegerAttr>(c.getValue()).getInt();
  if (components.hasAlwaysFalseCondition())
    return 0;
  auto resultOpt = components.getIterationsConstant();
  assert(resultOpt.has_value() && "must be counted loop");
  return *resultOpt;
}

static bool exceedsThresholdValue(cudaq::cc::LoopOp loop, std::size_t threshold,
                                  bool emitWarning = true) {
  auto components = cudaq::opt::getLoopComponents(loop);
  if (components->hasAlwaysTrueCondition()) {
    if (emitWarning)
      loop->emitWarning("Loop condition is always true. This loop is not "
                        "supported in a kernel.");
    return true;
  }
  auto upperBound = unrollLoopByValue(loop, *components);
  return upperBound >= threshold;
}

static bool loopCanBeUnrolled(cudaq::cc::LoopOp loop, bool allowBreak,
                              std::size_t threshold, bool emitWarning = true) {
  if (!allowBreak && !cudaq::opt::isaCountedLoop(loop))
    return false;
  if (allowBreak && !cudaq::opt::isaIndefiniteCountedLoop(loop))
    return false;
  return !exceedsThresholdValue(loop, threshold, emitWarning);
}

static bool valueDependsOnRegionBlockArgument(Value value, Region &region) {
  if (!value)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getOwner()->getParent() == &region;
  Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  for (Value operand : defOp->getOperands())
    if (valueDependsOnRegionBlockArgument(operand, region))
      return true;
  return false;
}

static bool loopTripCountDependsOnRegionArgs(cudaq::cc::LoopOp loop,
                                             Region &region) {
  auto components = cudaq::opt::getLoopComponents(loop);
  if (!components)
    return false;
  auto depends = [&](Value value) {
    return valueDependsOnRegionBlockArgument(value, region);
  };
  return depends(components->initialValue) ||
         depends(components->compareValue) || depends(components->stepValue) ||
         (components->addendValue && depends(components->addendValue)) ||
         (components->scaleValue && depends(components->scaleValue));
}

/// Returns true if \p loop accesses quantum data with a dynamic
/// (non-constant) index or slice and therefore blocks conversion to value
/// (wire) semantics: it contains a `quake.extract_ref` with a dynamic index or
/// a `quake.subveq` with a dynamic bound in its own body. The walk does NOT
/// descend into nested `cc.loop` ops -- a nested loop is classified (and
/// unrolled) on its own, and once it is unrolled its accesses become
/// constant-index, at which point this loop no longer blocks. This lets a
/// purely-repeating outer loop stay rolled while inner index-dependent loops
/// are unrolled. A loop with only constant-index references/slices (or only
/// `!quake.wire` values) does not block: mem2reg can thread those qubits as
/// loop iter-args.
static bool loopBlocksWireConversion(cudaq::cc::LoopOp loop) {
  bool found = false;
  loop->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (op != loop.getOperation() && isa<cudaq::cc::LoopOp>(op))
      return WalkResult::skip(); // nested loops are handled separately
    if (auto ext = dyn_cast<cudaq::quake::ExtractRefOp>(op))
      if (ext.getRawIndex() == cudaq::quake::ExtractRefOp::kDynamicIndex) {
        found = true;
        return WalkResult::interrupt();
      }
    if (auto sub = dyn_cast<cudaq::quake::SubVeqOp>(op))
      if (!sub.hasConstantLowerBound() || !sub.hasConstantUpperBound()) {
        found = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  return found;
}

static bool
loopHasParentDependentWireBlockingNestedLoop(cudaq::cc::LoopOp loop) {
  bool found = false;
  Region &body = loop.getBodyRegion();
  loop->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
    if (op == loop.getOperation())
      return WalkResult::advance();
    auto nested = dyn_cast<cudaq::cc::LoopOp>(op);
    if (!nested)
      return WalkResult::advance();
    if (loopBlocksWireConversion(nested) &&
        loopTripCountDependsOnRegionArgs(nested, body)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool loopMustUnrollForWireConversion(cudaq::cc::LoopOp loop) {
  return loopBlocksWireConversion(loop) ||
         loopHasParentDependentWireBlockingNestedLoop(loop);
}

namespace {

/// We fully unroll a counted loop (so marked with the counted attribute) as
/// long as the number of iterations is constant and that constant is less than
/// the threshold value.
///
/// Assumptions are made that the counted loop has a particular structural
/// layout as is consistent with the factory producing the counted loop.
///
/// After this pass, all loops marked counted will be unrolled or marked
/// invariant. An invariant loop means the loop must execute exactly some
/// specific number of times, even if that number is only known at runtime.
struct UnrollCountedLoop : public OpRewritePattern<cudaq::cc::LoopOp> {
  explicit UnrollCountedLoop(MLIRContext *ctx, std::size_t t, bool sf, bool ab,
                             unsigned &p, bool unrollOnlyWireBlocking = false)
      : OpRewritePattern(ctx), threshold(t), signalFailure(sf), allowBreak(ab),
        progress(p), unrollOnlyWireBlockingLoops(unrollOnlyWireBlocking) {}

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    // When the signalFailure flag is set, all loops are matched since that flag
    // requires that all LoopOp operations be rewritten. Despite the setting of
    // this flag, it may not be possible to fully unroll every LoopOp anyway.
    // Check for cases that are clearly not going to be unrolled.
    if (loop->hasAttr(cudaq::opt::DeadLoopAttr))
      return failure();
    // Opt-in selective unrolling for value-semantics pipelines: a loop that
    // does not itself index or slice qubits dynamically is left rolled unless
    // it must be unrolled first to make a nested wire-blocking loop's trip
    // count constant. Otherwise its qubits can be threaded as wire iter-args.
    if (unrollOnlyWireBlockingLoops) {
      // Filter out loops that cannot be unrolled before doing the more
      // expensive quantum-access walks below. These skips are intentionally
      // quiet: the final selective-mode validation diagnoses any remaining
      // wire-blocking loops.
      if (!loopCanBeUnrolled(loop, allowBreak, threshold,
                             /*emitWarning=*/false))
        return failure();
      if (!loopMustUnrollForWireConversion(loop))
        return failure();
    }
    if (!allowBreak && !cudaq::opt::isaCountedLoop(loop)) {
      if (signalFailure)
        loop.emitOpError("not a simple counted loop");
      return failure();
    }
    if (allowBreak && !cudaq::opt::isaIndefiniteCountedLoop(loop)) {
      if (signalFailure)
        loop.emitOpError("not a constant upper bound loop");
      return failure();
    }
    if (exceedsThresholdValue(loop, threshold)) {
      if (signalFailure)
        loop.emitOpError("loop bounds exceed iteration threshold");
      return failure();
    }

    // At this point, we're ready to unroll the loop and replace it with a
    // sequence of blocks. Each block will receive a block argument that is the
    // iteration number. The original cc.loop will be replaced by a constant,
    // the total number of iterations.
    // TODO: Allow the threading of other block arguments to the result.
    auto components = cudaq::opt::getLoopComponents(loop);
    assert(components && "counted loop must have components");
    if (!components->induction.has_value())
      return failure();
    auto unrollBy = unrollLoopByValue(loop, *components);
    Type inductionTy = loop.getOperands()[*components->induction].getType();
    LLVM_DEBUG(llvm::dbgs()
               << "unrolling loop by " << unrollBy << " iterations\n");
    auto loc = loop.getLoc();
    // Split the basic block in which this cc.loop appears.
    auto *insBlock = rewriter.getInsertionBlock();
    auto insPos = rewriter.getInsertionPoint();
    auto *endBlock = rewriter.splitBlock(insBlock, insPos);
    auto argTys = loop.getResultTypes();
    SmallVector<Location> argLocs(argTys.size(), loop.getLoc());
    endBlock->addArguments(argTys, argLocs);
    rewriter.setInsertionPointToEnd(insBlock);
    Value iterCount = getIntegerConstant(loc, inductionTy, 0, rewriter);
    SmallVector<Location> locsRange(loop.getNumResults(), loc);
    auto &bodyRegion = loop.getBodyRegion();
    SmallVector<Value> iterationOpers = loop.getOperands();
    auto setIterationOpers = [&](auto from) {
      assert(iterationOpers.size() == from.size());
      for (auto i : llvm::enumerate(from))
        iterationOpers[i.index()] = i.value();
    };

    // Make a constant number of copies of the body.
    Block *contBlock = nullptr;
    Value nextIterCount;
    for (std::size_t i = 0u; i < unrollBy; ++i) {
      // 1. Clone the while region.
      rewriter.cloneRegionBefore(loop.getWhileRegion(), endBlock);
      Block *whileBlock = insBlock->getNextNode();
      // 2. Clone the body region.
      rewriter.cloneRegionBefore(bodyRegion, endBlock);
      // Replace the ConditionOp in the while region clone with a direct branch.
      // This makes the comparison there dead. DCE will delete any unneeded code
      // associated with it.
      auto cond = cast<cudaq::cc::ConditionOp>(whileBlock->getTerminator());
      rewriter.setInsertionPoint(cond);
      rewriter.replaceOpWithNewOp<cf::BranchOp>(cond, whileBlock->getNextNode(),
                                                cond.getResults());
      auto cloneRange = findCloneRange(insBlock, endBlock);
      // 3. If the loop has a step region, clone it as well. Otherwise create an
      // empty block to target as the next "continue" block.
      if (loop.hasStep()) {
        contBlock = endBlock->getPrevNode();
        rewriter.cloneRegionBefore(loop.getStepRegion(), endBlock);
        contBlock = contBlock->getNextNode();
      } else {
        contBlock = rewriter.createBlock(endBlock, argTys, argLocs);
      }
      // Replace any continue and (possibly) break ops in the body region. They
      // are replaced with branches to the continue block or exit block, resp.
      for (Block *b = cloneRange.first; b != contBlock; b = b->getNextNode()) {
        auto *term = b->getTerminator();
        if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(term)) {
          auto termOpers = cont.getOperands();
          rewriter.setInsertionPoint(cont);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(cont, contBlock, termOpers);
        } else if (allowBreak) {
          if (auto brk = dyn_cast<cudaq::cc::BreakOp>(term)) {
            auto termOpers = brk.getOperands();
            rewriter.setInsertionPoint(brk);
            rewriter.replaceOpWithNewOp<cf::BranchOp>(brk, endBlock, termOpers);
          }
        }
      }
      // If there was a step region, its entry block is the continue block.
      // However, it may have multiple exit blocks. Thread each of these to a
      // merge block. The continue block is updated to this new empty merge
      // block.
      if (loop.hasStep()) {
        Block *mergeBlock = rewriter.createBlock(endBlock, argTys, argLocs);
        for (Block *b = contBlock; b != mergeBlock; b = b->getNextNode())
          if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(b->getTerminator())) {
            auto termOpers = cont.getOperands();
            rewriter.setInsertionPoint(cont);
            rewriter.replaceOpWithNewOp<cf::BranchOp>(cont, mergeBlock,
                                                      termOpers);
          }
        contBlock = mergeBlock;
      }
      // At this point, the continue block is a new, empty block. Generate the
      // next iteration number in this continue block.
      rewriter.setInsertionPointToEnd(contBlock);
      nextIterCount = getIntegerConstant(loc, inductionTy, i + 1, rewriter);
      rewriter.setInsertionPointToEnd(insBlock);
      // Propagate the previous iteration number into the new block. This makes
      // any unneeded computation dead. DCE will clean that up as well.
      iterationOpers[*components->induction] = iterCount;
      cf::BranchOp::create(rewriter, loc, cloneRange.first, iterationOpers);
      // Bookkeeping for the next iteration, which uses the new continue block,
      // `conBlock`, and its arguments.
      setIterationOpers(contBlock->getArguments());
      iterCount = nextIterCount;
      insBlock = contBlock;
    }

    // Finish up the last block.
    rewriter.setInsertionPointToEnd(insBlock);
    if (contBlock) {
      iterationOpers[*components->induction] = nextIterCount;
      setIterationOpers(contBlock->getArguments());
    }
    [[maybe_unused]] auto lastBranch =
        cf::BranchOp::create(rewriter, loc, endBlock, iterationOpers);
    rewriter.replaceOp(loop, endBlock->getArguments());

    LLVM_DEBUG(llvm::dbgs() << "after unrolling a loop:\n";
               lastBranch->getParentOfType<func::FuncOp>().dump());
    progress++;
    return success();
  }

  static Value getIntegerConstant(Location loc, Type ty, std::int64_t val,
                                  PatternRewriter &rewriter) {
    auto attr = rewriter.getIntegerAttr(ty, val);
    return arith::ConstantOp::create(rewriter, loc, ty, attr);
  }

  std::size_t threshold;
  bool signalFailure;
  bool allowBreak;
  unsigned &progress;
  bool unrollOnlyWireBlockingLoops;
};
} // namespace
