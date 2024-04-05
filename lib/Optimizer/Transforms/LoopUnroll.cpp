/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPUNROLL
#define GEN_PASS_DEF_UPDATEREGISTERNAMES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-unroll"

using namespace mlir;

inline std::pair<Block *, Block *> findCloneRange(Block *first, Block *last) {
  return {first->getNextNode(), last->getPrevNode()};
}

static std::size_t
unrollLoopByValue(cudaq::cc::LoopOp loop,
                  const cudaq::opt::LoopComponents &components) {
  auto c = components.compareValue.getDefiningOp<arith::ConstantOp>();
  return cast<IntegerAttr>(c.getValue()).getInt();
}

static std::size_t unrollLoopByValue(cudaq::cc::LoopOp loop) {
  auto components = cudaq::opt::getLoopComponents(loop);
  return unrollLoopByValue(loop, *components);
}

static bool exceedsThresholdValue(cudaq::cc::LoopOp loop,
                                  std::size_t threshold) {
  auto upperBound = unrollLoopByValue(loop);
  return upperBound >= threshold;
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
                             unsigned &p)
      : OpRewritePattern(ctx), threshold(t), signalFailure(sf), allowBreak(ab),
        progress(p) {}

  LogicalResult matchAndRewrite(cudaq::cc::LoopOp loop,
                                PatternRewriter &rewriter) const override {
    // When the signalFailure flag is set, all loops are matched since that flag
    // requires that all LoopOp operations be rewritten. Despite the setting of
    // this flag, it may not be possible to fully unroll every LoopOp anyway.
    // Check for cases that are clearly not going to be unrolled.
    if (!allowBreak && !cudaq::opt::isaCountedLoop(loop)) {
      if (signalFailure)
        loop.emitOpError("not a simple counted loop");
      return failure();
    }
    if (allowBreak && !cudaq::opt::isaConstantUpperBoundLoop(loop)) {
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
    auto unrollBy = unrollLoopByValue(loop, *components);
    if (components->isClosedIntervalForm())
      ++unrollBy;
    Type inductionTy = loop.getOperands()[components->induction].getType();
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
      // are repalced with branches to the continue block or exit block, resp.
      for (Block *b = cloneRange.first; b != contBlock; b = b->getNextNode()) {
        auto *term = b->getTerminator();
        if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(term)) {
          auto termOpers = cont.getOperands();
          rewriter.setInsertionPoint(cont);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(cont, contBlock, termOpers);
        }
        if (allowBreak) {
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
      iterationOpers[components->induction] = iterCount;
      rewriter.create<cf::BranchOp>(loc, cloneRange.first, iterationOpers);
      // Bookkeeping for the next iteration, which uses the new continue block,
      // `conBlock`, and its arguments.
      setIterationOpers(contBlock->getArguments());
      iterCount = nextIterCount;
      insBlock = contBlock;
    }

    // Finish up the last block.
    rewriter.setInsertionPointToEnd(insBlock);
    if (contBlock) {
      iterationOpers[components->induction] = nextIterCount;
      setIterationOpers(contBlock->getArguments());
    }
    [[maybe_unused]] auto lastBranch =
        rewriter.create<cf::BranchOp>(loc, endBlock, iterationOpers);
    rewriter.replaceOp(loop, endBlock->getArguments());

    LLVM_DEBUG(llvm::dbgs() << "after unrolling a loop:\n";
               lastBranch->getParentOfType<func::FuncOp>().dump());
    progress++;
    return success();
  }

  static Value getIntegerConstant(Location loc, Type ty, std::int64_t val,
                                  PatternRewriter &rewriter) {
    auto attr = rewriter.getIntegerAttr(ty, val);
    return rewriter.create<arith::ConstantOp>(loc, ty, attr);
  }

  std::size_t threshold;
  bool signalFailure;
  bool allowBreak;
  unsigned &progress;
};

/// The loop unrolling pass will fully unroll a `cc::LoopOp` when the loop is
/// known to always execute a constant number of iterations. That is, the loop
/// is a counted loop. (A threshold value can be used to bound the legal range
/// of iterations. The default is 50.)
class LoopUnrollPass : public cudaq::opt::impl::LoopUnrollBase<LoopUnrollPass> {
public:
  using LoopUnrollBase::LoopUnrollBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto *op = getOperation();
    auto numLoops = countLoopOps(op);
    unsigned progress = 0;
    if (numLoops) {
      PassManager pm(ctx);
      pm.addPass(createCanonicalizerPass());
      RewritePatternSet patterns(ctx);
      patterns.insert<UnrollCountedLoop>(ctx, threshold,
                                         /*signalFailure=*/false, allowBreak,
                                         progress);
      FrozenRewritePatternSet frozen(std::move(patterns));
      // Iterate over the loops until a fixed-point is reached. Some loops can
      // only be unrolled if other loops are unrolled first and the constants
      // iteratively propagated.
      do {
        progress = 0;
        (void)applyPatternsAndFoldGreedily(op, frozen);
        if (failed(pm.run(op)))
          break;
      } while (progress);
    }
    numLoops = countLoopOps(op);
    if (numLoops && signalFailure) {
      RewritePatternSet patterns(ctx);
      patterns.insert<UnrollCountedLoop>(ctx, threshold, signalFailure,
                                         allowBreak, progress);
      (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
      emitError(UnknownLoc::get(ctx), "did not unroll loops");
      signalPassFailure();
    }
  }

  static unsigned countLoopOps(Operation *op) {
    unsigned result = 0;
    op->walk([&](cudaq::cc::LoopOp loop) { result++; });
    LLVM_DEBUG(llvm::dbgs() << "Total number of loops: " << result << '\n');
    return result;
  }
};

/// After unrolling the loops, there may be duplicate registerName attributes in
/// use. This pass will assign them unique names by appending a counter.
class UpdateRegisterNamesPass
    : public cudaq::opt::impl::UpdateRegisterNamesBase<
          UpdateRegisterNamesPass> {
public:
  using UpdateRegisterNamesBase::UpdateRegisterNamesBase;

  void runOnOperation() override {
    auto *mod = getOperation();

    // First save the op's that contain a registerName attribute
    DenseMap<StringRef, SmallVector<Operation *>> regOps;
    mod->walk([&](mlir::Operation *walkOp) {
      if (auto prevAttr = walkOp->getAttr("registerName")) {
        auto registerName = prevAttr.cast<StringAttr>().getValue();
        regOps[registerName].push_back(walkOp);
      }
      return WalkResult::advance();
    });

    // Now apply new labels, appending a counter if necessary
    for (auto &[registerName, opVec] : regOps) {
      if (opVec.size() == 1)
        continue; // don't rename individual qubit measurements
      auto strLen = std::to_string(opVec.size()).size();
      int bit = 0;
      for (auto &regOp : opVec)
        if (auto prevAttr = regOp->getAttr("registerName")) {
          auto suffix = std::to_string(bit++);
          if (suffix.size() < strLen)
            suffix = std::string(strLen - suffix.size(), '0') + suffix;
          // Note Quantinuum can't support a ":" delimiter, so use '%'
          auto newAttr = OpBuilder(&getContext())
                             .getStringAttr(registerName + "%" + suffix);
          regOp->setAttr("registerName", newAttr);
        }
    }
  }
};

/// Unrolling pass pipeline command-line options. These options are similar to
/// the LoopUnroll pass options, but have different default settings.
struct UnrollPipelineOptions
    : public PassPipelineOptions<UnrollPipelineOptions> {
  PassOptions::Option<unsigned> threshold{
      *this, "threshold",
      llvm::cl::desc("Maximum iterations to unroll. (default: 1024)"),
      llvm::cl::init(1024)};
  PassOptions::Option<bool> signalFailure{
      *this, "signal-failure-if-any-loop-cannot-be-completely-unrolled",
      llvm::cl::desc(
          "Signal failure if pass can't unroll all loops. (default: true)"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> allowBreak{
      *this, "allow-early-exit",
      llvm::cl::desc("Allow unrolling of loop with early exit (i.e. break "
                     "statement). (default: true)"),
      llvm::cl::init(true)};
  PassOptions::Option<bool> allowClosedInterval{
      *this, "allow-closed-interval",
      llvm::cl::desc("Allow unrolling of loop with a closed interval form. "
                     "(default: true)"),
      llvm::cl::init(true)};
};
} // namespace

/// Add a pass pipeline to apply the requisite passes to fully unroll loops.
/// When converting to a quantum circuit, the static control program is fully
/// expanded to eliminate control flow. This pipeline will raise an error if any
/// loop in the module cannot be fully unrolled and signalFailure is set.
static void createUnrollingPipeline(OpPassManager &pm, unsigned threshold,
                                    bool signalFailure, bool allowBreak,
                                    bool allowClosedInterval) {
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  cudaq::opt::LoopNormalizeOptions lno{allowClosedInterval, allowBreak};
  pm.addPass(cudaq::opt::createLoopNormalize(lno));
  pm.addPass(createCanonicalizerPass());
  cudaq::opt::LoopUnrollOptions luo{threshold, signalFailure, allowBreak};
  pm.addPass(cudaq::opt::createLoopUnroll(luo));
  pm.addPass(cudaq::opt::createUpdateRegisterNames());
}

void cudaq::opt::registerUnrollingPipeline() {
  PassPipelineRegistration<UnrollPipelineOptions>(
      "unrolling-pipeline",
      "Fully unroll loops that can be completely unrolled.",
      [](OpPassManager &pm, const UnrollPipelineOptions &upo) {
        createUnrollingPipeline(pm, upo.threshold, upo.signalFailure,
                                upo.allowBreak, upo.allowClosedInterval);
      });
}
