/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

namespace cudaq::opt {

struct PartitionToOutline {
  /// Selected operations in cloning order. Each internal producer must precede
  /// its consumers. Duplicate pointers and invalid def-use order are rejected.
  /// The caller must preserve additional semantic ordering constraints.
  mlir::SmallVector<mlir::Operation *> ops;
  /// Optional operation before which the generated lambda and call must be
  /// inserted. Must belong to the partition's block and be outside ops.
  /// An earlier external consumer takes precedence as the insertion point.
  /// Outlining fails if captured inputs are unavailable at the chosen point.
  /// With nullptr, placement follows external consumers or falls back to just
  /// after the physically last selected operation.
  /// This pointer is non-owning and must remain valid until this partition is
  /// processed. outlinePartitions rejects the batch if this operation is
  /// selected by any partition in the batch.
  mlir::Operation *mustPrecede = nullptr;
};

/// Outline supported operations from one block into a lambda and a call.
/// Operations are cloned in the supplied def-use order, with all external
/// inputs captured. Placement is constrained by partition.mustPrecede.
/// Non-wire results precede wire results. Wire results follow the first-use
/// order of their corresponding incoming wires. Returns failure without
/// modifying the IR if the selection cannot be outlined.
mlir::FailureOr<cudaq::cc::CreateLambdaOp>
outlinePartition(const PartitionToOutline &partition);

/// Outline each supplied partition in its supplied def-use order using its own
/// mustPrecede boundary, returning failure if any is rejected.
/// Batch validation rejects duplicate or overlapping selections and boundaries
/// selected by any partition, returning failure without modifying the IR.
/// Once validation succeeds, partitions are outlined individually; successful
/// rewrites are retained even if another partition fails to outline.
mlir::LogicalResult
outlinePartitions(mlir::ArrayRef<PartitionToOutline> partitions);

} // namespace cudaq::opt
