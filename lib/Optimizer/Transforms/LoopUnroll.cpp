/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPUNROLL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-unroll"

using namespace mlir;

#include "LoopUnrollPatterns.inc"

namespace {
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
      RewritePatternSet patterns(ctx);
      for (auto *dialect : ctx->getLoadedDialects())
        dialect->getCanonicalizationPatterns(patterns);
      for (RegisteredOperationName op : ctx->getRegisteredOperations())
        op.getCanonicalizationPatterns(patterns, ctx);
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
      } while (progress);
    }

    if (signalFailure) {
      numLoops = countLoopOps(op);
      if (numLoops) {
        op->emitOpError("did not unroll loops");
        signalPassFailure();
      }
    }
  }

  static unsigned countLoopOps(Operation *op) {
    unsigned result = 0;
    op->walk([&](cudaq::cc::LoopOp loop) {
      if (!loop->hasAttr(cudaq::opt::DeadLoopAttr))
        result++;
    });
    LLVM_DEBUG(llvm::dbgs() << "Total number of loops: " << result << '\n');
    return result;
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
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  cudaq::opt::LoopNormalizeOptions lno{allowClosedInterval, allowBreak};
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize(lno));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  cudaq::opt::LoopUnrollOptions luo{threshold, signalFailure, allowBreak};
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll(luo));
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUpdateRegisterNames());
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
