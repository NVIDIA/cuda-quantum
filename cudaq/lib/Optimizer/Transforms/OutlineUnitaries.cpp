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
    const cudaq::quake::detail::UnitaryOpGroups &groups = analysis.getGroups();

    // TODO: Build PartitionToOutline requests from the groups. Preserve the
    // analysis's supplied operation order, which may differ from block order.
    // Decide how to handle trailingDelimiterOps and delimiter-only groups.

    // TODO: Choose each partition's mustPrecede boundary. Boundaries must not
    // be selected by any partition in the batch.
    // TODO: Call cudaq::opt::outlinePartitions and handle failure, remembering
    // that a failed batch may retain successful rewrites. Do not consult the
    // grouping analysis after mutating the IR.
    (void)groups;
  }
};
} // namespace
