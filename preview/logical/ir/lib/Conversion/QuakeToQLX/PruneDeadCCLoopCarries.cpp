//===- PruneDeadCCLoopCarries.cpp - Quake loop cleanup --------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#include "qlx/Conversion/QuakeToQLXPasses.h"

#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace qlx {
#define GEN_PASS_DEF_PRUNEDEADCCLOOPCARRIES
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"
} // namespace qlx

using namespace mlir;

namespace {

static LogicalResult validateSupportedLoop(cudaq::cc::LoopOp loop) {
  auto reject = [&](StringRef reason) -> LogicalResult {
    return loop.emitOpError("cannot prune dead carries: ") << reason;
  };

  if (loop.getPostCondition())
    return reject("post-condition loops are outside the supported subset");
  if (!loop.getElseRegion().empty())
    return reject("Python loop-else is outside the supported subset");
  if (!loop.getWhileRegion().hasOneBlock() ||
      !loop.getBodyRegion().hasOneBlock() ||
      !loop.getStepRegion().hasOneBlock())
    return reject("while/do/step regions must each contain one block");

  bool hasBreak = false;
  loop.getBodyRegion().walk([&](cudaq::cc::BreakOp) { hasBreak = true; });
  if (hasBreak)
    return reject("break is outside the supported subset");

  unsigned arity = loop.getInitialArgs().size();
  Block &whileBlock = loop.getWhileRegion().front();
  Block &bodyBlock = loop.getBodyRegion().front();
  Block &stepBlock = loop.getStepRegion().front();
  auto condition = dyn_cast<cudaq::cc::ConditionOp>(whileBlock.getTerminator());
  auto bodyContinue =
      dyn_cast<cudaq::cc::ContinueOp>(bodyBlock.getTerminator());
  auto stepContinue =
      dyn_cast<cudaq::cc::ContinueOp>(stepBlock.getTerminator());
  if (!condition || !bodyContinue || !stepContinue)
    return reject("regions must end in cc.condition/cc.continue");
  if (loop.getNumResults() != arity || whileBlock.getNumArguments() != arity ||
      bodyBlock.getNumArguments() != arity ||
      stepBlock.getNumArguments() != arity ||
      condition.getResults().size() != arity ||
      bodyContinue.getNumOperands() != arity ||
      stepContinue.getNumOperands() != arity)
    return reject("loop carry arity is inconsistent across regions");
  return success();
}

static bool hasUndefInitialCarry(cudaq::cc::LoopOp loop) {
  return llvm::any_of(loop.getInitialArgs(), [](Value value) {
    return value.getDefiningOp<cudaq::cc::UndefOp>() != nullptr;
  });
}

static bool containsScope(func::FuncOp function) {
  return function
      .walk([](cudaq::cc::ScopeOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

class PruneDeadCCLoopCarriesPass
    : public qlx::impl::PruneDeadCCLoopCarriesBase<PruneDeadCCLoopCarriesPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<cudaq::cc::CCDialect, mlir::func::FuncDialect,
                    mlir::ub::UBDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> cleanupRoots;
    llvm::SmallPtrSet<Operation *, 4> seenRoots;
    WalkResult validation = module.walk([&](cudaq::cc::LoopOp loop) {
      if (failed(validateSupportedLoop(loop)))
        return WalkResult::interrupt();
      if (!hasUndefInitialCarry(loop))
        return WalkResult::advance();

      auto function = loop->getParentOfType<func::FuncOp>();
      if (!function) {
        loop.emitOpError(
            "cannot prune dead carries outside an enclosing func.func");
        return WalkResult::interrupt();
      }
      if (containsScope(function)) {
        loop.emitOpError(
            "cannot prune dead carries while cc.scope remains; lower the "
            "scope before the Quake-to-P0 handoff");
        return WalkResult::interrupt();
      }
      if (seenRoots.insert(function.getOperation()).second)
        cleanupRoots.push_back(function);
      return WalkResult::advance();
    });
    if (validation.wasInterrupted())
      return signalPassFailure();
    if (cleanupRoots.empty())
      return;

    // Let upstream MLIR establish liveness and poison non-live forwarded
    // operands. Limit the pass to affected, isolated functions: running it at
    // module scope would also canonicalize unrelated cc.scope operations and
    // introduce the CF dialect into this narrow handoff preparation.
    OpPassManager deadValuePipeline(func::FuncOp::getOperationName());
    deadValuePipeline.addPass(createRemoveDeadValuesPass());
    for (func::FuncOp root : cleanupRoots)
      if (failed(runPipeline(deadValuePipeline, root.getOperation())))
        return signalPassFailure();

    // Finish scenario 2 of remove-dead-values by structurally erasing each
    // dead tied set: loop operand/result, region arguments, and terminator
    // operands. This is MLIR's generic implementation, not a QLX deadness rule.
    RewritePatternSet patterns(&getContext());
    populateRegionBranchOpInterfaceCanonicalizationPatterns(
        patterns, cudaq::cc::LoopOp::getOperationName());
    FrozenRewritePatternSet frozen(std::move(patterns));
    for (func::FuncOp root : cleanupRoots)
      if (failed(applyPatternsGreedily(root, frozen))) {
        signalPassFailure();
        return;
      }
  }
};

} // namespace
