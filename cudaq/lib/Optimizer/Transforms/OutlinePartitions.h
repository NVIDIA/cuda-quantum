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
/// Operations are cloned in block order, with captured non-wire inputs and
/// explicit wire parameters. Non-wire results precede wire results. Returns
/// failure without modifying the IR if the selection cannot be outlined.
mlir::FailureOr<cudaq::cc::CreateLambdaOp>
outlinePartition(llvm::ArrayRef<mlir::Operation *> partition);

/// Outline each supplied partition, returning failure if any is rejected.
/// Successful rewrites are retained even if another partition is rejected.
mlir::LogicalResult outlinePartitions(
    llvm::ArrayRef<llvm::SmallVector<mlir::Operation *>> partitions);

} // namespace cudaq::opt
