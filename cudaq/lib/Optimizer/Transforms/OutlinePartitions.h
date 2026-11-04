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
  mlir::SmallVector<mlir::Operation *> ops;
  /// Optional operation before which the generated lambda and call must be
  /// inserted. Must belong to the partition's block and be outside ops.
  /// An earlier external consumer takes precedence as the insertion point.
  /// Outlining fails if captured inputs are unavailable at the chosen point.
  /// With nullptr, placement follows external consumers or falls back to just
  /// after the physically last selected operation.
  /// This pointer is non-owning and must remain valid until this partition is
  /// processed; earlier partitions in a batch must not erase the operation.
  mlir::Operation *mustPrecede = nullptr;
  /// If true, clone ops in the supplied order, rejecting duplicates and any
  /// internal producer placed after its consumer. The caller must preserve
  /// additional semantic ordering constraints. If false, ignore duplicates and
  /// clone operations in block order.
  bool acceptOrderedPartition = false;
};

/// Outline supported operations from one block into a lambda and a call.
/// Cloning order is controlled by partition.acceptOrderedPartition, and
/// placement is constrained by partition.mustPrecede. All external inputs are
/// captured.
/// Non-wire results precede wire results. Wire results follow the first-use
/// order of their corresponding incoming wires. Returns failure without
/// modifying the IR if the selection cannot be outlined.
mlir::FailureOr<cudaq::cc::CreateLambdaOp>
outlinePartition(const PartitionToOutline &partition);

/// Outline each supplied partition using its own acceptOrderedPartition and
/// mustPrecede settings, returning failure if any is rejected.
/// Successful rewrites are retained even if another partition is rejected.
mlir::LogicalResult
outlinePartitions(mlir::ArrayRef<PartitionToOutline> partitions);

} // namespace cudaq::opt
