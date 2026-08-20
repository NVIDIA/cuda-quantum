/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LoopAnalysis.h"
#include "PassDetails.h"
#include "QuakeOperatorUtilities.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_CONSOLIDATEBROADCASTS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "consolidate-broadcasts"

using namespace mlir;

namespace {

/// Does \p region hold nothing but the loop's own control: no side effects,
/// and no way out of the loop?
bool isControlOnly(Region &region) {
  return !region
              .walk([](Operation *op) {
                if (!isMemoryEffectFree(op) ||
                    isa<cudaq::cc::BreakOp, cudaq::cc::UnwindBreakOp>(op))
                  return WalkResult::interrupt();
                return WalkResult::advance();
              })
              .wasInterrupted();
}

/// Match `for (i = 0; i < N; ++i) { op(v[i]); ... }`, where `v` is a veq of
/// size `N`, and replace the entire loop with `op(v); ...`.
LogicalResult rollLoop(cudaq::cc::LoopOp loop) {
  // A counted loop runs a constant number of iterations from 0, stepping by 1,
  // with no early exit and no `do while` form. It must also leave nothing
  // behind.
  if (!cudaq::opt::isaCountedLoop(loop) ||
      !llvm::all_of(loop->getResults(), [](Value v) { return v.use_empty(); }))
    return failure();
  auto components = cudaq::opt::getLoopComponents(loop);
  assert(components && "counted loop must have components");
  auto iterations = components->getIterationsConstant();
  if (!iterations)
    return failure();

  // An `else` region must not be dropped, and the while and step regions,
  // which the rewrite also drops, must hold nothing but the loop's control.
  if (loop.hasPythonElse() || !isControlOnly(loop.getWhileRegion()) ||
      !isControlOnly(loop.getStepRegion()))
    return failure();

  Region &body = loop.getBodyRegion();
  if (!body.hasOneBlock())
    return failure();
  Block &block = body.front();
  // Only one argument, the induction variable
  if (block.getNumArguments() != 1)
    return failure();

  // In the loop body, we're matching on the form
  // %var = quake.extract_ref [%induction_var] %veq
  // quake.op %var
  // ...
  // cc.continue
  // Where each op is a broadcast operator
  auto extract = dyn_cast<cudaq::quake::ExtractRefOp>(block.front());
  if (!extract || extract.getIndex() != block.getArgument(0))
    return failure();

  auto isBroadcastable = [&extract](Operation &op) {
    auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op);
    if (!gate)
      return false;
    if (!cudaq::opt::isBroadcastOperator(gate))
      return false;
    if (!gate.getControls().empty() || gate.getTargets().size() != 1 ||
        gate.getTargets()[0] != extract.getRef())
      return false;

    return true;
  };

  SmallVector<cudaq::quake::OperatorInterface> broadcastable;

  for (Operation &op : block.without_terminator()) {
    if (extract == &op)
      continue;
    if (!isBroadcastable(op))
      return failure();
    broadcastable.emplace_back(&op);
  }
  if (broadcastable.empty())
    return failure();

  // The loop must walk the whole vector.
  Value veq = extract.getVeq();
  if (cudaq::quake::getVeqSize(veq) != iterations)
    return failure();

  // Given the body above, the operators' parameters are all defined outside
  // the loop, so the clones are well-formed there. The sole target is the last
  // operand.
  OpBuilder builder(loop);
  for (auto gate : broadcastable) {
    Operation *broadcast = builder.clone(*gate.getOperation());
    broadcast->setOperand(broadcast->getNumOperands() - 1, veq);
  }
  loop.erase();
  return success();
}

struct ConsolidateBroadcastsPass
    : public cudaq::opt::impl::ConsolidateBroadcastsBase<
          ConsolidateBroadcastsPass> {
  using ConsolidateBroadcastsBase::ConsolidateBroadcastsBase;

  void runOnOperation() override {
    SmallVector<cudaq::cc::LoopOp> loops;
    getOperation().walk([&](cudaq::cc::LoopOp loop) { loops.push_back(loop); });
    for (auto loop : loops)
      (void)rollLoop(loop);
  }
};
} // namespace
