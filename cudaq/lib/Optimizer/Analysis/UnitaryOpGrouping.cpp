/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Support/TypeID.h"

#define DEBUG_TYPE "unitary-op-grouping-analysis"

using namespace mlir;

namespace {
static bool isUnitaryOp(Operation *op) {
  return op->hasTrait<cudaq::QuantumGate>() && !isa<cudaq::quake::ResetOp>(op);
}

/// If we've hit this function, that means we've reached an op that cannot be
/// added to the current group of unitary ops. So, we do one of two things.
/// 1) If the current group is empty, we don't do anything and just return
/// 2) If the current group is non-empty, we have have a UnitaryOpsGroup to
///    populate. We create the struct and then push it back to the
///    UnitaryOpGroups vector.
static void
flushGroupIfNonEmpty(cudaq::quake::detail::UnitaryOpGroups &groups,
                     Block &block,
                     llvm::SmallVectorImpl<Operation *> &currUnitaryOps) {
  if (currUnitaryOps.empty())
    return;

  cudaq::quake::detail::UnitaryOpGroup group;
  group.block = &block;
  group.firstOp = currUnitaryOps.front();
  group.lastOp = currUnitaryOps.back();
  group.ops.append(currUnitaryOps.begin(), currUnitaryOps.end());

  LLVM_DEBUG(llvm::dbgs() << "Found unitary group with " << group.ops.size()
                          << " op(s)\n");

  groups.push_back(std::move(group));
  currUnitaryOps.clear();
}
} // namespace

const mlir::Block *
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getBlockForGroup(
    const UnitaryOpGroup &group) const {
  return group.block;
}

const cudaq::quake::detail::UnitaryOpGroup *
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getGroupContainingOp(
    mlir::Operation *op) const {
  if (!op || !isUnitaryOp(op))
    return nullptr;

  for (const UnitaryOpGroup &group : groups)
    if (llvm::is_contained(group.ops, op))
      return &group;

  return nullptr;
}

llvm::SmallVector<const cudaq::quake::detail::UnitaryOpGroup *>
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getGroupsIn(
    const mlir::Block *block) const {
  llvm::SmallVector<const UnitaryOpGroup *> groupsInBlock;
  for (const UnitaryOpGroup &group : groups) {
    if (group.block == block)
      groupsInBlock.push_back(&group);
  }
  return groupsInBlock;
}

bool cudaq::quake::detail::UnitaryOpGroupingAnalysis::inSameGroup(
    mlir::Operation *op1, mlir::Operation *op2) const {
  if (!op1 || !op2 || !isUnitaryOp(op1) || !isUnitaryOp(op2))
    return false;

  const UnitaryOpGroup *group1 = getGroupContainingOp(op1);
  const UnitaryOpGroup *group2 = getGroupContainingOp(op2);

  return group1 && group2 && group1 == group2;
}

void cudaq::quake::detail::UnitaryOpGroupingAnalysis::scanBlock(Block &block) {
  llvm::SmallVector<Operation *> currUnitaryOps;

  for (Operation &op : block) {
    if (isUnitaryOp(&op)) {
      currUnitaryOps.push_back(&op);
      continue;
    }

    flushGroupIfNonEmpty(groups, block, currUnitaryOps);
  }

  flushGroupIfNonEmpty(groups, block, currUnitaryOps);
}

void cudaq::quake::detail::UnitaryOpGroupingAnalysis::scanRegion(
    Region &region) {
  for (Block &block : region) {
    scanBlock(block);

    // Nested regions, such as cc.if branches or cc.loop bodies, are scanned as
    // independent blocks. MVP 1 never groups across the parent control-flow op.
    for (Operation &op : block)
      for (Region &nestedRegion : op.getRegions())
        scanRegion(nestedRegion);
  }
}

void cudaq::quake::detail::UnitaryOpGroupingAnalysis::performAnalysis(
    Operation *operation) {
  auto funcOp = dyn_cast<func::FuncOp>(operation);
  if (!funcOp)
    return;

  LLVM_DEBUG(llvm::dbgs() << "Function to analyze: " << funcOp.getName()
                          << '\n');

  /// initially start with the func op region, then recursively scan nested
  /// regions
  for (Region &region : funcOp->getRegions())
    scanRegion(region);

  LLVM_DEBUG(llvm::dbgs() << "Found " << groups.size()
                          << " unitary group(s)\n");
}
