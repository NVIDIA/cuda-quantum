/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
using UnitaryOpGroup = cudaq::quake::detail::UnitaryOpGroup;
using UnitaryOpGroupingAnalysis =
    cudaq::quake::detail::UnitaryOpGroupingAnalysis;

static void printBool(bool value) {
  llvm::errs() << (value ? "true" : "false");
}

static void printOperationMembership(
    Operation &op, const UnitaryOpGroupingAnalysis &analysis,
    const llvm::DenseMap<const UnitaryOpGroup *, unsigned> &groupNumbers) {
  // Exercise getGroupContainingOp for both unitary and non-unitary ops. A
  // unitary op should map back to the group that owns it; measurements,
  // classical ops, terminators, and parent control-flow ops should print
  // "none" instead of pretending they belong to a unitary group.
  const UnitaryOpGroup *group = analysis.getGroupContainingOp(&op);

  llvm::errs() << "    " << op.getName().getStringRef() << " -> ";
  if (!group) {
    llvm::errs() << "none\n";
    return;
  }

  auto iter = groupNumbers.find(group);
  if (iter == groupNumbers.end()) {
    llvm::errs() << "unknown\n";
    return;
  }

  llvm::errs() << "group #" << iter->second << '\n';
}

static void printOperationMemberships(
    Operation *root, const UnitaryOpGroupingAnalysis &analysis,
    const llvm::DenseMap<const UnitaryOpGroup *, unsigned> &groupNumbers) {
  // Walk the operation tree recursively so nested regions, such as cc.if
  // branches, are checked too. This keeps the test honest about the analysis
  // finding nested unitary groups while rejecting the non-unitary wrapper ops.
  for (Region &region : root->getRegions())
    for (Block &block : region)
      for (Operation &op : block) {
        printOperationMembership(op, analysis, groupNumbers);
        printOperationMemberships(&op, analysis, groupNumbers);
      }
}

static void
printAdjacentInSameGroupChecks(Operation *root,
                               const UnitaryOpGroupingAnalysis &analysis) {
  // Exercise inSameGroup on neighboring operations in each block. This gives
  // simple true cases for consecutive unitary ops in one run, and false cases
  // where a measurement, classical op, terminator, or control-flow wrapper
  // breaks the run. Recurse so nested-region blocks are checked independently.
  //
  // Recursively walking the operation tree also allows us to check nested
  // regions, e.g., cc.if
  for (Region &region : root->getRegions())
    for (Block &block : region) {
      Operation *previous = nullptr;
      for (Operation &op : block) {
        if (previous) {
          llvm::errs() << "    " << previous->getName().getStringRef() << ", "
                       << op.getName().getStringRef() << " -> ";
          printBool(analysis.inSameGroup(previous, &op));
          llvm::errs() << '\n';
        }

        printAdjacentInSameGroupChecks(&op, analysis);
        previous = &op;
      }
    }
}

static void printBlockSummaries(const UnitaryOpGroupingAnalysis &analysis) {
  // Exercise the block-oriented query helpers. Each unitary group records the
  // block it came from, so collect the distinct blocks that contain at least
  // one group and then print how many groups the analysis reports in each one.
  // The numbering here is local to this test output; it is only meant to make
  // the FileCheck expectations deterministic without printing block pointers.
  llvm::SmallVector<const Block *> blocks;
  for (const UnitaryOpGroup &group : analysis.getGroups()) {
    const Block *block = analysis.getBlockForGroup(group);
    if (!llvm::is_contained(blocks, block))
      blocks.push_back(block);
  }

  llvm::errs() << "  blocks:\n";
  for (auto indexedBlock : llvm::enumerate(blocks)) {
    llvm::SmallVector<const UnitaryOpGroup *> groupsInBlock =
        analysis.getGroupsIn(indexedBlock.value());
    llvm::errs() << "    block #" << indexedBlock.index() << " -- contains "
                 << groupsInBlock.size() << " unitary groups\n";
  }
}

static void printGroupMembershipConsistencyChecks(
    const UnitaryOpGroupingAnalysis &analysis) {
  // Exercise inSameGroup using the groups discovered by the analysis itself.
  // For multi-op groups, the first and last op should be reported as belonging
  // together. For neighboring groups, their first ops should never be reported
  // as belonging together, even when the groups are in the same block.
  const auto &groups = analysis.getGroups();

  llvm::errs() << "  group membership consistency checks:\n";
  for (auto indexedGroup : llvm::enumerate(groups)) {
    const UnitaryOpGroup &group = indexedGroup.value();
    if (group.ops.size() < 2)
      continue;

    llvm::errs() << "    group #" << indexedGroup.index() << " first/last -> ";
    printBool(analysis.inSameGroup(group.ops.front(), group.ops.back()));
    llvm::errs() << '\n';
  }

  for (unsigned i = 0, end = groups.size(); i + 1 < end; ++i) {
    llvm::errs() << "    group #" << i << " first, group #" << i + 1
                 << " first -> ";
    printBool(
        analysis.inSameGroup(groups[i].ops.front(), groups[i + 1].ops.front()));
    llvm::errs() << '\n';
  }
}

class TestUnitaryOpGroupingPass
    : public PassWrapper<TestUnitaryOpGroupingPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestUnitaryOpGroupingPass)

  StringRef getArgument() const final { return "test-unitary-op-grouping"; }

  StringRef getDescription() const final {
    return "Print found groups of unitary operations";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (auto funcOp : module.getOps<func::FuncOp>()) {
      auto &analysis =
          getChildAnalysis<UnitaryOpGroupingAnalysis>(funcOp.getOperation());

      llvm::DenseMap<const UnitaryOpGroup *, unsigned> groupNumbers;
      llvm::errs() << "func @" << funcOp.getName() << '\n';

      unsigned groupIndex = 0;
      for (const auto &group : analysis.getGroups()) {
        groupNumbers.try_emplace(&group, groupIndex);
        llvm::errs() << "  unitary group #" << groupIndex++ << " -- found "
                     << group.ops.size() << " ops\n";
        for (Operation *op : group.ops)
          llvm::errs() << "    " << op->getName().getStringRef() << '\n';
      }

      printBlockSummaries(analysis);

      llvm::errs() << "  op membership:\n";
      printOperationMemberships(funcOp.getOperation(), analysis, groupNumbers);

      llvm::errs() << "  adjacent inSameGroup checks:\n";
      printAdjacentInSameGroupChecks(funcOp.getOperation(), analysis);

      printGroupMembershipConsistencyChecks(analysis);
    }

    markAllAnalysesPreserved();
  }
};
} // namespace

namespace cudaq::test {
void registerTestUnitaryOpGroupingPass() {
  PassRegistration<TestUnitaryOpGroupingPass>();
}
} // namespace cudaq::test
