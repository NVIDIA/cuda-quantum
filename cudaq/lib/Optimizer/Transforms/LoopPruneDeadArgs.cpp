/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Characteristics.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LOOPPRUNEDEADARGS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-loop-prune-dead-args"

using namespace mlir;

namespace {

using LoopCarriedSlot = std::pair<Operation *, unsigned>;

/// Return the slot a use passes its value into, if passing it along is all the
/// use does (a loop's initial argument, or a region terminator forwarding it
/// around the loop).
std::optional<LoopCarriedSlot> forwardedSlot(OpOperand &use) {
  Operation *user = use.getOwner();
  unsigned pos = use.getOperandNumber();
  if (auto loop = dyn_cast<cudaq::cc::LoopOp>(user))
    return LoopCarriedSlot{loop.getOperation(), pos};
  auto loop = dyn_cast_or_null<cudaq::cc::LoopOp>(user->getParentOp());
  if (!loop)
    return std::nullopt;
  if (isa<cudaq::cc::ConditionOp>(user)) {
    if (pos == 0)
      return std::nullopt;
    return LoopCarriedSlot{loop.getOperation(), pos - 1};
  }
  if (isa<cudaq::cc::ContinueOp, cudaq::cc::BreakOp>(user))
    return LoopCarriedSlot{loop.getOperation(), pos};
  return std::nullopt;
}

/// Collect every value occupying slot pos of loop (the loop's result and the
/// matching entry block argument of each of its regions).
SmallVector<Value> valuesInSlot(cudaq::cc::LoopOp loop, unsigned pos) {
  SmallVector<Value> values;
  if (pos < loop.getNumResults())
    values.push_back(loop.getResult(pos));
  for (auto *region : loop.getRegions()) {
    if (region->empty())
      continue;
    Block &entry = region->front();
    if (pos < entry.getNumArguments())
      values.push_back(entry.getArgument(pos));
  }
  return values;
}

/// The slots in dead can be removed from loop's signature only if every
/// terminator forwarding values around the loop sits directly in one of its
/// regions, and nothing reads the block arguments being dropped.
bool canEraseDeadSlots(cudaq::cc::LoopOp loop,
                       const llvm::SmallBitVector &dead) {
  bool ok = true;
  loop.getOperation()->walk([&](Operation *op) {
    if (!isa<cudaq::cc::ConditionOp, cudaq::cc::BreakOp>(op) ||
        op->getParentOp() == loop.getOperation() || !op->getNumOperands())
      return;
    // A break nested in a scope still exits this loop and carries its values.
    for (auto *p = op->getParentOp(); p; p = p->getParentOp()) {
      if (p == loop.getOperation()) {
        ok = false;
        return;
      }
      if (isa<cudaq::cc::LoopOp>(p))
        return;
    }
  });
  if (!ok)
    return false;
  for (auto *region : loop.getRegions()) {
    if (region->empty())
      continue;
    Block &entry = region->front();
    for (unsigned pos = 0, end = dead.size(); pos != end; ++pos)
      if (dead[pos] && pos < entry.getNumArguments() &&
          !entry.getArgument(pos).use_empty())
        return false;
  }
  return true;
}

/// Rebuild loop without the slots marked in dead.
void eraseDeadSlots(cudaq::cc::LoopOp loop, const llvm::SmallBitVector &dead) {
  // Trim the operands each region terminator forwards.
  for (auto *region : loop.getRegions())
    for (Block &block : *region) {
      if (!block.hasNoSuccessors())
        continue;
      Operation *term = block.getTerminator();
      if (term->getParentOp() != loop.getOperation() ||
          !isa<cudaq::cc::ConditionOp, cudaq::cc::ContinueOp,
               cudaq::cc::BreakOp>(term))
        continue;
      unsigned offset = isa<cudaq::cc::ConditionOp>(term) ? 1 : 0;
      SmallVector<Value> keep;
      for (unsigned i = 0, n = term->getNumOperands(); i != n; ++i)
        if (i < offset || dead.size() <= i - offset || !dead[i - offset])
          keep.push_back(term->getOperand(i));
      term->setOperands(keep);
    }

  // Drop the matching block arguments.
  for (auto *region : loop.getRegions()) {
    if (region->empty())
      continue;
    Block &entry = region->front();
    for (int pos = dead.size() - 1; pos >= 0; --pos)
      if (dead[pos] && static_cast<unsigned>(pos) < entry.getNumArguments())
        entry.eraseArgument(pos);
  }

  SmallVector<Value> newInitArgs;
  SmallVector<Type> newResultTypes;
  for (unsigned pos = 0, end = loop.getInitialArgs().size(); pos != end; ++pos)
    if (!dead[pos]) {
      newInitArgs.push_back(loop.getInitialArgs()[pos]);
      if (pos < loop.getNumResults())
        newResultTypes.push_back(loop.getResultTypes()[pos]);
    }

  OpBuilder builder(loop);
  auto newLoop = cudaq::cc::LoopOp::create(
      builder, loop.getLoc(), newResultTypes, newInitArgs,
      loop.isPostConditional(), [](OpBuilder &, Location, Region &) {},
      [](OpBuilder &, Location, Region &) {},
      /*stepBuilder=*/nullptr);
  newLoop->setDiscardableAttrs(loop->getDiscardableAttrDictionary());
  newLoop.getWhileRegion().takeBody(loop.getWhileRegion());
  newLoop.getBodyRegion().takeBody(loop.getBodyRegion());
  newLoop.getStepRegion().takeBody(loop.getStepRegion());
  newLoop.getElseRegion().takeBody(loop.getElseRegion());

  unsigned newPos = 0;
  for (unsigned pos = 0, end = loop.getNumResults(); pos != end; ++pos)
    if (!dead[pos])
      loop.getResult(pos).replaceAllUsesWith(newLoop.getResult(newPos++));
  loop.erase();
}

void pruneDeadLoopCarriedValues(func::FuncOp func) {
  SmallVector<cudaq::cc::LoopOp> loops;
  func.walk([&](cudaq::cc::LoopOp loop) { loops.push_back(loop); });
  if (loops.empty())
    return;

  // Reaching the loop-carried values means going through the arithmetic that
  // computes them, and that arithmetic reads the loop's own block arguments.
  SmallVector<Value> candidates;
  auto isPrunableComputation = [](Operation *op) {
    return op->getNumRegions() == 0 && !op->hasTrait<OpTrait::IsTerminator>() &&
           !cudaq::opt::hasQuantum(*op) && isMemoryEffectFree(op);
  };
  for (auto loop : loops)
    for (unsigned pos = 0, end = loop.getInitialArgs().size(); pos != end;
         ++pos)
      llvm::append_range(candidates, valuesInSlot(loop, pos));
  func.walk([&](Operation *op) {
    if (isPrunableComputation(op))
      llvm::append_range(candidates, op->getResults());
  });

  // Start from the assumption that every candidate is dead. Record the
  // reverse liveness dependencies, then propagate from opaque uses with a
  // worklist. A value is live if
  // 1. it has an opaque use
  // 2. it is forwarded into a slot already known to be live
  // 3. it feeds a computation whose own result is already known to be live
  // Each dependency is visited only when its source becomes live.
  DenseSet<Value> live;
  DenseMap<Value, SmallVector<Value>> livenessDependents;
  DenseMap<LoopCarriedSlot, SmallVector<Value>> slotForwarders;
  SmallVector<Value> worklist;
  auto markLive = [&](Value value) {
    if (live.insert(value).second)
      worklist.push_back(value);
  };
  for (Value val : candidates)
    for (OpOperand &use : val.getUses()) {
      if (auto forwarded = forwardedSlot(use)) {
        slotForwarders[*forwarded].push_back(val);
        continue;
      }
      Operation *user = use.getOwner();
      if (!isPrunableComputation(user)) {
        markLive(val);
        continue;
      }
      for (Value result : user->getResults())
        livenessDependents[result].push_back(val);
    }
  for (auto &[slot, forwarders] : slotForwarders) {
    auto loop = cast<cudaq::cc::LoopOp>(slot.first);
    for (Value member : valuesInSlot(loop, slot.second))
      llvm::append_range(livenessDependents[member], forwarders);
  }
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    auto iter = livenessDependents.find(value);
    if (iter == livenessDependents.end())
      continue;
    for (Value dependent : iter->second)
      markLive(dependent);
  }

  auto slotIsLive = [&](LoopCarriedSlot slot) {
    auto loop = cast<cudaq::cc::LoopOp>(slot.first);
    return llvm::any_of(valuesInSlot(loop, slot.second),
                        [&](Value val) { return live.count(val); });
  };

  auto deleteDeadComputations = [&]() {
    DenseSet<Operation *> quantumContainingOps;
    func.walk([&](Operation *op) {
      if (!op->hasTrait<cudaq::QuantumGate>())
        return;
      for (Operation *ancestor = op; ancestor;
           ancestor = ancestor->getParentOp())
        if (!quantumContainingOps.insert(ancestor).second)
          break;
    });

    SmallVector<Operation *> worklist;
    DenseSet<Operation *> queued;
    auto addIfDead = [&](Operation *op) {
      if (op && !queued.count(op) && !quantumContainingOps.count(op) &&
          isOpTriviallyDead(op)) {
        queued.insert(op);
        worklist.push_back(op);
      }
    };

    // Pre-order plus a LIFO worklist erases nested operations before a dead
    // parent that owns them.
    func.walk<WalkOrder::PreOrder>(addIfDead);
    while (!worklist.empty()) {
      Operation *op = worklist.pop_back_val();
      queued.erase(op);
      SmallVector<Operation *> definingOps;
      for (Value operand : op->getOperands())
        if (Operation *definingOp = operand.getDefiningOp())
          definingOps.push_back(definingOp);
      Operation *parent = op->getParentOp();
      op->erase();

      // Removing the final use can make an operand's definition dead. A
      // region-owning parent can also become dead when its last effecting
      // nested operation disappears.
      for (Operation *definingOp : definingOps)
        addIfDead(definingOp);
      addIfDead(parent);
    }
  };

  // Short-circuit each dead slot with the loop's initial value for that slot.
  // The initial value is an operand of the loop, so it dominates every use we
  // rewrite.
  for (auto loop : loops)
    for (auto [pos, initialArg] : llvm::enumerate(loop.getInitialArgs())) {
      if (slotIsLive(LoopCarriedSlot{loop.getOperation(), pos}))
        continue;
      for (auto *region : loop.getRegions())
        for (Block &block : *region) {
          if (!block.hasNoSuccessors())
            continue;
          Operation *term = block.getTerminator();
          if (term->getParentOp() != loop.getOperation())
            continue;
          unsigned operandPos =
              isa<cudaq::cc::ConditionOp>(term) ? pos + 1 : pos;
          if (isa<cudaq::cc::ConditionOp, cudaq::cc::ContinueOp,
                  cudaq::cc::BreakOp>(term) &&
              operandPos < term->getNumOperands())
            term->setOperand(operandPos, initialArg);
        }
    }

  // Delete the computations the short-circuiting just made dead. This has to
  // happen before the slots are erased. While a dead slot's block argument
  // still feeds arithmetic, `canEraseDeadSlots` refuses to drop it.
  deleteDeadComputations();

  // Drop each dead slot from the loop's signature. Erasing an inner loop's
  // slots is what frees up the enclosing loop's, so keep going until nothing
  // more can be dropped.
  for (bool changed = true; changed;) {
    changed = false;
    SmallVector<cudaq::cc::LoopOp> current;
    func.walk([&](cudaq::cc::LoopOp loop) { current.push_back(loop); });
    for (auto loop : llvm::reverse(current)) {
      llvm::SmallBitVector dead(loop.getInitialArgs().size());
      for (unsigned pos = 0, end = dead.size(); pos != end; ++pos)
        if (!slotIsLive(LoopCarriedSlot{loop.getOperation(), pos}))
          dead.set(pos);
      if (dead.none() || !canEraseDeadSlots(loop, dead))
        continue;
      eraseDeadSlots(loop, dead);
      changed = true;
      break;
    }
  }

  deleteDeadComputations();
}

class LoopPruneDeadArgsPass
    : public cudaq::opt::impl::LoopPruneDeadArgsBase<LoopPruneDeadArgsPass> {
public:
  using LoopPruneDeadArgsBase::LoopPruneDeadArgsBase;

  void runOnOperation() override { pruneDeadLoopCarriedValues(getOperation()); }
};
} // namespace
