/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_NORMALIZEATOMICQUANTUMREGIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "normalize-atomic-quantum-regions"

using namespace mlir;

namespace {
/// \returns true if any operation in \p region uses a value-semantics (wire,
/// cable, or control) quantum type anywhere in its operands or results. This
/// pass only reasons about reference-semantics (`quake.alloca`/
/// `quake.dealloc`) Quake IR.
static bool hasValueSemantics(Region &region) {
  bool found = false;
  region.walk([&](Operation *op) {
    auto isQuantumValue = [](Value v) {
      return cudaq::quake::isQuantumValueType(v.getType());
    };
    if (llvm::any_of(op->getOperands(), isQuantumValue) ||
        llvm::any_of(op->getResults(), isQuantumValue)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Check that is it safe to sink a measurement.
static bool isMeasurementSafeToSink(Operation *measureOp, Block &block) {
  auto measure = cast<cudaq::quake::MeasurementInterface>(measureOp);
  llvm::SmallPtrSet<Value, 4> targets;
  for (Value target : measure.getTargets())
    targets.insert(target);

  bool afterMeasure = false;
  for (Operation &op : block) {
    if (&op == measureOp) {
      afterMeasure = true;
      continue;
    }
    if (!afterMeasure)
      continue;
    if (auto dealloc = dyn_cast<cudaq::quake::DeallocOp>(op))
      if (targets.contains(dealloc.getReference()))
        continue;
    if (llvm::any_of(op.getOperands(),
                     [&](Value v) { return targets.contains(v); }))
      return false;
  }
  return true;
}

/// Attempt to hoist `quake.alloca`s (and sink their paired `quake.dealloc`s
/// and any measurement's transitive closure of users) out of \p scope, a
/// `cc.scope` carrying the `atomic_quantum_region` attribute. Returns true if
/// \p scope was rewritten.
static bool processAtomicScope(cudaq::cc::ScopeOp scope) {
  // For now, only rewrite a single-block, zero-result atomic scope. This
  // sidesteps having to thread `cc.continue` operands (which would flow
  // either through the still-nested atomic scope or through the sunk tail)
  // and having to reason about which exit a multi-block region's allocations
  // are live across.
  if (scope.getNumResults() != 0 || !scope.getRegion().hasOneBlock())
    return false;

  if (hasValueSemantics(scope.getRegion()))
    return false;

  Block &block = scope.getRegion().front();

  SmallVector<cudaq::quake::AllocaOp> allocas;
  for (Operation &op : block)
    if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op))
      allocas.push_back(alloc);
  if (allocas.empty())
    return false;

  llvm::DenseSet<Operation *> allocaSet;
  for (auto alloc : allocas)
    allocaSet.insert(alloc);

  // A dynamically-sized alloca's `%size` operand must already dominate the
  // scope (i.e. not be defined inside the block being hoisted out of); it is
  // not itself something we try to hoist.
  for (auto alloc : allocas)
    if (Value size = alloc.getSize())
      if (Operation *def = size.getDefiningOp())
        if (def->getBlock() == &block)
          return false;

  // Deallocations of the hoisted allocations get sunk alongside them.
  llvm::DenseSet<Operation *> sinkSet;
  for (Operation &op : block)
    if (auto dealloc = dyn_cast<cudaq::quake::DeallocOp>(op))
      if (Operation *def = dealloc.getReference().getDefiningOp())
        if (allocaSet.contains(def))
          sinkSet.insert(dealloc);

  // Measurements, and the transitive closure of every op that consumes a
  // measurement's results, get sunk as well. If any such consumer lives
  // outside this block (e.g. nested in nested control flow within the atomic
  // scope), bail rather than attempt an unsound partial hoist.
  SmallVector<Operation *> worklist;
  for (Operation &op : block) {
    if (!isa<cudaq::quake::MeasurementInterface>(op))
      continue;
    if (!isMeasurementSafeToSink(&op, block)) {
      op.emitWarning(
          "measurement in an atomic quantum region will not be sunk out of "
          "the region because the measured reference is used again "
          "afterwards");
      continue;
    }
    worklist.push_back(&op);
  }
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!sinkSet.insert(op).second)
      continue;
    if (op->getBlock() != &block)
      return false;
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers())
        worklist.push_back(user);
  }

  // Every op being sunk must only depend on values defined outside the block
  // or on other ops in the hoisted/sunk sets -- otherwise sinking it would
  // reference a value left behind inside the (still nested) atomic scope,
  // which is not visible from the new outer scope.
  for (Operation *op : sinkSet)
    for (Value operand : op->getOperands())
      if (Operation *def = operand.getDefiningOp())
        if (def->getBlock() == &block && !allocaSet.contains(def) &&
            !sinkSet.contains(def))
          return false;

  // Collect the ops to sink in their original relative order.
  SmallVector<Operation *> sinkOpsInOrder;
  for (Operation &op : block)
    if (sinkSet.contains(&op))
      sinkOpsInOrder.push_back(&op);

  OpBuilder builder(scope);
  builder.setInsertionPoint(scope);
  auto outerScope = cudaq::cc::ScopeOp::create(builder, scope.getLoc(),
                                               [](OpBuilder &, Location) {});
  Block &outerBlock = outerScope.getRegion().front();
  OpBuilder terminatorBuilder(&outerBlock, outerBlock.end());
  cudaq::cc::ContinueOp::create(terminatorBuilder, scope.getLoc());
  Operation *terminator = &outerBlock.back();

  scope->moveBefore(terminator);
  for (auto alloc : allocas)
    alloc->moveBefore(scope);
  for (Operation *op : sinkOpsInOrder)
    op->moveBefore(terminator);

  return true;
}

class NormalizeAtomicQuantumRegionsPass
    : public cudaq::opt::impl::NormalizeAtomicQuantumRegionsBase<
          NormalizeAtomicQuantumRegionsPass> {
public:
  using NormalizeAtomicQuantumRegionsBase::NormalizeAtomicQuantumRegionsBase;

  void runOnOperation() override {
    Operation *op = getOperation();
    SmallVector<cudaq::cc::ScopeOp> atomicScopes;
    op->walk([&](cudaq::cc::ScopeOp scope) {
      if (scope.getAtomicQuantumRegionAttr())
        atomicScopes.push_back(scope);
    });
    for (auto scope : atomicScopes)
      processAtomicScope(scope);
  }
};
} // namespace
