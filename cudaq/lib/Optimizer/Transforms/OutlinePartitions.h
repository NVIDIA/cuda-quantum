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

/// Outline supported operations from one block into a lambda and a call.
/// By default, duplicate operations are ignored and operations are cloned in
/// block order. If acceptOrderedPartition is true, operations are cloned in the
/// supplied order, which must contain no duplicates and place each internal
/// producer before its consumers. The caller must preserve any additional
/// semantic ordering constraints. All external inputs are captured.
/// Non-wire results precede wire results. Wire results follow the first-use
/// order of their corresponding incoming wires. Returns failure without
/// modifying the IR if the selection cannot be outlined.
mlir::FailureOr<cudaq::cc::CreateLambdaOp>
outlinePartition(llvm::ArrayRef<mlir::Operation *> partition,
                 const bool acceptOrderedPartition = false);

/// Outline each supplied partition using outlinePartition's ordering policy,
/// returning failure if any is rejected.
/// Successful rewrites are retained even if another partition is rejected.
mlir::LogicalResult outlinePartitions(
    llvm::ArrayRef<llvm::SmallVector<mlir::Operation *>> partitions,
    const bool acceptOrderedPartition = false);

} // namespace cudaq::opt
