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
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CANONICALIZEKERNEL
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

static RewritePatternSet canonicalizationPatterns(MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  for (auto *dialect : ctx->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
  return patterns;
}

// A straight-line SSA block has no unreachable blocks, removable block
// arguments, or cyclic dead uses. The greedy worklist removes dead leaf ops.
static bool isFlatKernel(func::FuncOp func) {
  return func.getBody().hasOneBlock() &&
         isa<func::ReturnOp>(func.front().getTerminator()) &&
         llvm::none_of(func.front(),
                       [](Operation &op) { return op.getNumRegions() != 0; });
}

// Only CanonicalizeKernel publishes this analysis, after the greedy driver
// converges on a gate list. A pass that changes the IR invalidates it.
struct CanonicalGateList {
  explicit CanonicalGateList(Operation *) {}
};

static bool isGateList(func::FuncOp func) {
  return llvm::all_of(func.front(), [](Operation &op) {
#define QUAKE_OP(name) cudaq::quake::name
    return isa<QUANTUM_OPS(QUAKE_OP), cudaq::quake::AllocaOp,
               cudaq::quake::DeallocOp, cudaq::quake::ExtractRefOp,
               cudaq::quake::EvinceOp, cudaq::quake::NullWireOp,
               cudaq::quake::SinkOp, arith::ConstantOp, func::ReturnOp>(op);
#undef QUAKE_OP
  });
}

class CanonicalizeKernelPass
    : public cudaq::opt::impl::CanonicalizeKernelBase<CanonicalizeKernelPass> {
  LogicalResult initialize(MLIRContext *ctx) override {
    patterns = FrozenRewritePatternSet(canonicalizationPatterns(ctx));
    return success();
  }

  void runOnOperation() override {
    if (getCachedAnalysis<CanonicalGateList>()) {
      markAllAnalysesPreserved();
      return;
    }
    auto func = getOperation();
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    bool flat = isFlatKernel(func);
    config.setRegionSimplificationLevel(
        flat ? GreedySimplifyRegionLevel::Disabled
             : GreedySimplifyRegionLevel::Normal);
    bool changed = false;
    auto converged = applyPatternsGreedily(func, patterns, config, &changed);
    // A rewrite can introduce control flow into an initially flat kernel.
    if (changed && flat && !isFlatKernel(func)) {
      config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);
      (void)applyPatternsGreedily(func, patterns, config);
    } else if (flat && succeeded(converged) && isGateList(func)) {
      (void)getAnalysis<CanonicalGateList>();
      markAnalysesPreserved<CanonicalGateList>();
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

  FrozenRewritePatternSet patterns;
};

/// The classical optimization pass performs a number of classical
/// optimizations greedily until changes no more changes can be done.
class ClassicalOptimizationPass
    : public cudaq::opt::impl::ClassicalOptimizationBase<
          ClassicalOptimizationPass> {
public:
  using ClassicalOptimizationBase::ClassicalOptimizationBase;

  void runOnOperation() override {
    auto *op = getOperation();
    if (unrollOnlyAliasingQuantumAccessLoops && unrollOnlyIndexUseLoops) {
      op->emitOpError("unroll-only-aliasing-quantum-access-loops and "
                      "unroll-only-index-use-loops are mutually exclusive");
      signalPassFailure();
      return;
    }
    // The additional patterns only match classical control flow or memory.
    // CSE still runs when requested and may expose new canonicalizations.
    bool changed = !getCachedAnalysis<CanonicalGateList>() && optimize(op);
    if (cseAndRepeat) {
      DominanceInfo dominance(op);
      IRRewriter rewriter(&getContext());
      bool cseChanged = false;
      eliminateCommonSubExpressions(rewriter, dominance, op, &cseChanged);
      changed |= cseChanged;
      if (changed)
        optimize(op);
    }
    if (!changed)
      markAllAnalysesPreserved();
  }

private:
  bool optimize(Operation *op) {
    auto *ctx = &getContext();
    DominanceInfo domInfo(op);
    auto func = dyn_cast<func::FuncOp>(op);
    auto numLoops = countLoopOps(op);
    unsigned progress = 0;

    auto patterns = canonicalizationPatterns(ctx);

    // Add patterns that help const prop loop boundaries computed
    // in conditional statements, other loops, or arrays.
    patterns.insert<RewriteIf>(ctx, /*rewriteOnlyIfConst=*/true);
    patterns.insert<LoopPat>(ctx, allowClosedInterval, allowBreak);
    patterns.insert<AllocaPattern>(
        ctx, domInfo, func == nullptr ? "unknown" : func.getName());
    if (numLoops && !disableLoopUnrolling)
      patterns.insert<UnrollCountedLoop>(
          ctx, threshold,
          /*signalFailure=*/false, allowBreak, progress,
          unrollOnlyAliasingQuantumAccessLoops, unrollOnlyIndexUseLoops);

    FrozenRewritePatternSet frozen(std::move(patterns));
    // Iterate over the loops until a fixed-point is reached. Some loops can
    // only be unrolled if other loops are unrolled first and the constants
    // iteratively propagated.
    bool changed = false;
    do {
      // Remove overridden writes.
      auto analysis = SimplifyWritesAnalysis(domInfo, op);
      changed |= analysis.removeOverriddenStores();
      // Clean up dead code.
      {
        auto builder = OpBuilder(op);
        IRRewriter rewriter(builder);
        changed |= succeeded(simplifyRegions(rewriter, op->getRegions()));
      }
      progress = 0;
      bool rewritten = false;
      if (failed(applyPatternsGreedily(op, frozen, {}, &rewritten)))
        return true;
      changed |= rewritten;
    } while (progress);
    return changed;
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
  PassOptions::Option<bool> disableLoopUnrolling{
      *this, "no-loop-unroll",
      llvm::cl::desc("Disable loop unrolling and preserve cc.loop operations. "
                     "(default: false)"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> unrollOnlyAliasingQuantumAccessLoops{
      *this, "unroll-only-aliasing-quantum-access-loops",
      llvm::cl::desc(
          "Unroll only loops containing aliasing quantum accesses or "
          "exp_pauli operands that must be resolved for wire-set lowering. "
          "(default: false)"),
      llvm::cl::init(false)};
  PassOptions::Option<bool> unrollOnlyIndexUseLoops{
      *this, "unroll-only-index-use-loops",
      llvm::cl::desc("Unroll only loops whose induction variable is used in "
                     "the loop body. (default: false)"),
      llvm::cl::init(false)};
};
} // namespace

static void createClassicalOptPipeline(
    OpPassManager &pm, const ClassicalOptimizationPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCanonicalizeKernel());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createSROA());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());

  // CSE can expose a second round of classical optimizations.
  cudaq::opt::ClassicalOptimizationOptions opts;
  opts.cseAndRepeat = true;
  opts.threshold = options.threshold;
  opts.allowClosedInterval = options.allowClosedInterval;
  opts.allowBreak = options.allowBreak;
  opts.disableLoopUnrolling = options.disableLoopUnrolling;
  opts.unrollOnlyAliasingQuantumAccessLoops =
      options.unrollOnlyAliasingQuantumAccessLoops;
  opts.unrollOnlyIndexUseLoops = options.unrollOnlyIndexUseLoops;
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalOptimization(opts));
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUpdateRegisterNames());
}

void cudaq::opt::createClassicalOptimizationPipeline(
    OpPassManager &pm, std::optional<unsigned> threshold,
    std::optional<bool> allowBreak, std::optional<bool> allowClosedInterval,
    std::optional<bool> disableLoopUnrolling,
    std::optional<bool> unrollOnlyAliasingQuantumAccessLoops,
    std::optional<bool> unrollOnlyIndexUseLoops) {
  ClassicalOptimizationPipelineOptions options;
  if (threshold.has_value())
    options.threshold = *threshold;
  if (allowClosedInterval.has_value())
    options.allowClosedInterval = *allowClosedInterval;
  if (allowBreak.has_value())
    options.allowBreak = *allowBreak;
  if (disableLoopUnrolling.has_value())
    options.disableLoopUnrolling = *disableLoopUnrolling;
  if (unrollOnlyAliasingQuantumAccessLoops.has_value())
    options.unrollOnlyAliasingQuantumAccessLoops =
        *unrollOnlyAliasingQuantumAccessLoops;
  if (unrollOnlyIndexUseLoops.has_value())
    options.unrollOnlyIndexUseLoops = *unrollOnlyIndexUseLoops;
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
