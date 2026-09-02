/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace cudaq::opt {

/// Trivial greedy qubit-partition analysis.
///
/// Walks wire-semantics operations in block order and assigns them to
/// partitions. A partition grows as long as the number of distinct qubit wire
/// timelines it uses does not exceed \p maxQubits. When an operation would
/// exceed that limit, the current partition is closed and a new one begins.
///
/// A "qubit wire timeline" entering a partition is counted once:
///   - External wire values consumed by a partition op whose defining op is
///     outside the partition.
///   - Wire values produced by wire-source ops (null_wire, borrow_wire) that
///     reside inside the partition.
struct GreedyOpPartitioner {
  explicit GreedyOpPartitioner(mlir::Operation *op, unsigned maxQubits);

  mlir::ArrayRef<llvm::DenseSet<mlir::Operation *>> getPartitions() const {
    return partitions;
  }

private:
  void partitionBlock(mlir::Block &block, unsigned maxQubits);

  mlir::SmallVector<llvm::DenseSet<mlir::Operation *>> partitions;
};

} // namespace cudaq::opt
