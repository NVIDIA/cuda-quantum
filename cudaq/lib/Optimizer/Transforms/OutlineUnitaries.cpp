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
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_OUTLINEUNITARIES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {
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
    // TODO: Define and document the policy for unsupported groups, including
    // reference-semantics operations: skip them or fail the pass explicitly.
    mlir::SmallVector<cudaq::opt::PartitionToOutline> partitionsToOutline;
    for (const auto &group : unitaryOpGroups) {
      // note that some groups may contain only delimiter ops
      if (group.ops.empty())
        continue;
      partitionsToOutline.push_back(
          {.ops = {group.ops.begin(), group.ops.end()},
           // TODO: Once UnitaryOpGroupingAnalysis records segmentEnd, use it
           // instead of nullptr when there are no trailing delimiters. Default
           // placement can otherwise move the call across a hard boundary to
           // a later external consumer. Keep boundaries outside all partitions.
           .mustPrecede = group.trailingDelimiterOps.empty()
                              ? nullptr
                              : group.trailingDelimiterOps.front()});
    }

    // A failed batch may retain successful rewrites. Do not consult the
    // grouping analysis after this call mutates the IR.
    if (mlir::failed(cudaq::opt::outlinePartitions(partitionsToOutline))) {
      getOperation().emitError("failed to outline unitary groups");
      signalPassFailure();
    }
  }
};
} // namespace
