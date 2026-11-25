/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OutlinePartitions.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Analysis/UnitaryOpGrouping.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_OUTLINEUNITARIES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {

/// Return whether the group already occupies an entire lambda body.
bool isAlreadyOutlined(const cudaq::quake::detail::UnitaryOpGroup &group) {
  auto *block = group.block;
  if (!mlir::isa<cudaq::cc::CreateLambdaOp>(block->getParentOp()))
    return false;

  // The block must contain only the unitary ops and a single return op.
  // The +1 accounts for the cc.return op that gets added during
  // unitary outlining and isn't present in the original group.
  return block->getParent()->hasOneBlock() && !block->empty() &&
         mlir::isa<cudaq::cc::ReturnOp>(block->back()) &&
         block->getOperations().size() == group.ops.size() + 1;
}

class OutlineUnitariesPass
    : public cudaq::opt::impl::OutlineUnitariesBase<OutlineUnitariesPass> {
public:
  using OutlineUnitariesBase::OutlineUnitariesBase;

  void runOnOperation() override {
    const auto &analysis =
        getAnalysis<cudaq::quake::detail::UnitaryOpGroupingAnalysis>();
    const cudaq::quake::detail::UnitaryOpGroups &unitaryOpGroups =
        analysis.getGroups();

    // Preserve the analysis's supplied operation order, which may differ from
    // block order. Leave trailing delimiters outside the outlined partitions.
    // Every nonempty unitary group must already be outlined or be outlined
    // here; unsupported groups fail the pass.
    mlir::SmallVector<cudaq::opt::PartitionToOutline> partitionsToOutline;
    for (const auto &group : unitaryOpGroups) {
      // Delimiter-only groups contain no unitaries to outline.
      if (group.ops.empty())
        continue;
      if (isAlreadyOutlined(group))
        continue;
      partitionsToOutline.push_back(
          {.ops = {group.ops.begin(), group.ops.end()},
           .mustPrecede = group.trailingDelimiterOps.empty()
                              ? group.hardBoundaryOp
                              : group.trailingDelimiterOps.front()});
    }

    // A failed batch may retain successful rewrites. Do not consult the
    // grouping analysis after this call mutates the IR.
    auto reportFailure = [&](mlir::Operation *op, llvm::StringRef reason) {
      (op ? op : getOperation().getOperation())
              ->emitError("failed to outline unitary group: ")
          << reason;
    };
    if (mlir::failed(cudaq::opt::outlinePartitions(partitionsToOutline,
                                                   reportFailure))) {
      signalPassFailure();
    }
  }
};
} // namespace
