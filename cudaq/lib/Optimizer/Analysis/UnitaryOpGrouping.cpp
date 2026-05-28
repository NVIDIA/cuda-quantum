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

#define DEBUG_TYPE "unitary-op-grouping-analysis"

using namespace mlir;

static bool isUnitaryOp(Operation *op) {
  return op->hasTrait<cudaq::QuantumGate>() && !isa<cudaq::quake::ResetOp>(op);
}

/// If we've hit this function, that means we've reached an op that cannot be
/// added to the current group of unitary ops. So, we do one of two things.
/// 1) If the current group is empty, we don't do anything and just return
/// 2) If the current group is non-empty, we have have a UnitaryOpsGroup to
///    populate. We create the struct, push it back to the UnitaryOpGroups
///    vector, and update the lookup tables for the new group.
void cudaq::quake::detail::UnitaryOpGroupingAnalysis::flushGroupIfNonEmpty(
    Block &block, llvm::SmallVectorImpl<Operation *> &currUnitaryOps) {
  if (currUnitaryOps.empty())
    return;

  // size of groups before insertion is the index of the new group to be
  // inserted.
  unsigned groupIndex = static_cast<unsigned>(groups.size());

  UnitaryOpGroup group;
  group.block = &block;
  group.ops.append(currUnitaryOps.begin(), currUnitaryOps.end());

  LLVM_DEBUG(llvm::dbgs() << "Found unitary group with " << group.ops.size()
                          << " op(s)\n");

  groups.push_back(std::move(group));

  /// populate lookup tables now that new group has been created.
  const UnitaryOpGroup &insertedGroup = groups.back();
  blockToGroupIndices[insertedGroup.block].push_back(groupIndex);
  for (Operation *op : insertedGroup.ops)
    opToGroupIndex.try_emplace(op, groupIndex);

  currUnitaryOps.clear();
}

const mlir::Block *
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getBlockForGroup(
    const UnitaryOpGroup &group) const {
  return group.block;
}

const cudaq::quake::detail::UnitaryOpGroup *
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getGroupContainingOp(
    mlir::Operation *op) const {
  if (!op)
    return nullptr;

  auto iter = opToGroupIndex.find(op);
  if (iter == opToGroupIndex.end())
    return nullptr;

  return &groups[iter->second];
}

llvm::SmallVector<const cudaq::quake::detail::UnitaryOpGroup *>
cudaq::quake::detail::UnitaryOpGroupingAnalysis::getGroupsIn(
    const mlir::Block *block) const {
  llvm::SmallVector<const UnitaryOpGroup *> groupsInBlock;
  if (!block)
    return groupsInBlock;

  auto iter = blockToGroupIndices.find(block);
  if (iter == blockToGroupIndices.end())
    return groupsInBlock;

  for (unsigned groupIndex : iter->second)
    groupsInBlock.push_back(&groups[groupIndex]);

  return groupsInBlock;
}

bool cudaq::quake::detail::UnitaryOpGroupingAnalysis::inSameGroup(
    mlir::Operation *op1, mlir::Operation *op2) const {
  if (!op1 || !op2)
    return false;

  auto group1 = opToGroupIndex.find(op1);
  if (group1 == opToGroupIndex.end())
    return false;

  auto group2 = opToGroupIndex.find(op2);
  if (group2 == opToGroupIndex.end())
    return false;

  return group1->second == group2->second;
}

/// Scan a single block and form groups from maximal contiguous runs of
/// unitary operations. Any non-unitary operation terminates the current run,
/// but it is not added to a group itself. If that non-unitary operation owns
/// nested regions, those regions are scanned immediately after the current
/// block-local run is flushed.
void cudaq::quake::detail::UnitaryOpGroupingAnalysis::scanBlock(Block &block) {
  llvm::SmallVector<Operation *> currUnitaryOps;

  for (Operation &op : block) {
    if (isUnitaryOp(&op)) {
      currUnitaryOps.push_back(&op);
      continue;
    }

    flushGroupIfNonEmpty(block, currUnitaryOps);

    // Nested regions, such as cc.if branches or cc.loop bodies, are scanned as
    // independent regions at the parent op boundary; this prevents groups
    // across the parent control-flow op.
    for (Region &nestedRegion : op.getRegions())
      scanRegion(nestedRegion);
  }

  flushGroupIfNonEmpty(block, currUnitaryOps);
}

/// Scan every block in a region. Nested regions are reached from scanBlock when
/// their owning operation is encountered.
void cudaq::quake::detail::UnitaryOpGroupingAnalysis::scanRegion(
    Region &region) {
  for (Block &block : region)
    scanBlock(block);
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
