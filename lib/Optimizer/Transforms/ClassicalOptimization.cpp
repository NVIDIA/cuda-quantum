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
#define GEN_PASS_DEF_CLASSICALOPTIMIZATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "classical-optimizations"

using namespace mlir;

#include "LiftArrayAllocPatterns.inc"
#include "LoopNormalizePatterns.inc"
#include "LoopUnrollPatterns.inc"
#include "LowerToCFGPatterns.inc"
#include "WriteAfterWriteEliminationPatterns.inc"

namespace {

/// The classical optimization pass performs a number of classical
/// optimizations greedily until changes no more changes can be done.
class ClassicalOptimizationPass
    : public cudaq::opt::impl::ClassicalOptimizationBase<
          ClassicalOptimizationPass> {
public:
  using ClassicalOptimizationBase::ClassicalOptimizationBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto *op = getOperation();
    DominanceInfo domInfo(op);
    auto func = dyn_cast<func::FuncOp>(op);
    auto numLoops = countLoopOps(op);
    unsigned progress = 0;

    RewritePatternSet patterns(ctx);
    for (auto *dialect : ctx->getLoadedDialects())
      dialect->getCanonicalizationPatterns(patterns);
    for (RegisteredOperationName op : ctx->getRegisteredOperations())
      op.getCanonicalizationPatterns(patterns, ctx);

    // Add patterns that help const prop loop boundaries computed
    // in conditional statements, other loops, or arrays.
    patterns.insert<RewriteIf>(ctx, /*rewriteOnlyIfConst=*/true);
    patterns.insert<LoopPat>(ctx, allowClosedInterval, allowBreak);
    patterns.insert<AllocaPattern>(
        ctx, domInfo, func == nullptr ? "unknown" : func.getName());
    if (numLoops)
      patterns.insert<UnrollCountedLoop>(ctx, threshold,
                                         /*signalFailure=*/false, allowBreak,
                                         progress);

    FrozenRewritePatternSet frozen(std::move(patterns));
    // Iterate over the loops until a fixed-point is reached. Some loops can
    // only be unrolled if other loops are unrolled first and the constants
    // iteratively propagated.
    do {
      // Remove overridden writes.
      auto analysis = SimplifyWritesAnalysis(domInfo, op);
      analysis.removeOverriddenStores();
      // Clean up dead code.
      {
        auto builder = OpBuilder(op);
        IRRewriter rewriter(builder);
        [[maybe_unused]] auto unused =
            simplifyRegions(rewriter, op->getRegions());
      }
      progress = 0;
      (void)applyPatternsAndFoldGreedily(op, frozen);
    } while (progress);
  }

  static unsigned countLoopOps(Operation *op) {
    unsigned result = 0;
    op->walk([&](cudaq::cc::LoopOp loop) { result++; });
    LLVM_DEBUG(llvm::dbgs() << "Total number of loops: " << result << '\n');
    return result;
  }
};

/// Classical optimization pipeline command-line options. These options are
/// similar to the ClassicalOptimization pass options, but have different
/// default settings.
struct ClassicalOptimizationPipelineOptions
    : public PassPipelineOptions<ClassicalOptimizationPipelineOptions> {
  PassOptions::Option<unsigned> threshold{
      *this, "threshold",
      llvm::cl::desc("Maximum iterations to unroll. (default: 1024)"),
      llvm::cl::init(1024)};
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

static void createClassicalOptPipeline(
    OpPassManager &pm, const ClassicalOptimizationPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createSROA());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());

  // Run classical optimization twice with a cse in between to optimize more
  // code.
  // TODO: run cse as a part of classical-optimization when we update the llvm
  // version.
  cudaq::opt::ClassicalOptimizationOptions opts;
  opts.threshold = options.threshold;
  opts.allowClosedInterval = options.allowClosedInterval;
  opts.allowBreak = options.allowBreak;
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalOptimization(opts));
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalOptimization(opts));
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUpdateRegisterNames());
}

void cudaq::opt::createClassicalOptimizationPipeline(
    OpPassManager &pm, std::optional<unsigned> threshold,
    std::optional<bool> allowBreak, std::optional<bool> allowClosedInterval) {
  ClassicalOptimizationPipelineOptions options;
  if (threshold.has_value())
    options.threshold = *threshold;
  if (allowClosedInterval.has_value())
    options.allowClosedInterval = *allowClosedInterval;
  if (allowBreak.has_value())
    options.allowBreak = *allowBreak;
  ::createClassicalOptPipeline(pm, options);
}

void cudaq::opt::registerClassicalOptimizationPipeline() {
  PassPipelineRegistration<ClassicalOptimizationPipelineOptions>(
      "classical-optimization-pipeline", "Fully optimize classical code.",
      [](OpPassManager &pm,
         const ClassicalOptimizationPipelineOptions &options) {
        ::createClassicalOptPipeline(pm, options);
      });
}
