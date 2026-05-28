/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/TypeID.h"

namespace mlir {
class Operation;
class Block;
class Region;
} // namespace mlir

namespace cudaq::quake::detail {
struct UnitaryOpGroup {
  UnitaryOpGroup() = default;

  mlir::Block *block = nullptr;
  llvm::SmallVector<mlir::Operation *> ops;
};

using UnitaryOpGroups = llvm::SmallVector<cudaq::quake::detail::UnitaryOpGroup>;

/// Analysis to group unitary operations within blocks.
///
/// This analysis finds maximal contiguous runs of unitary quantum operations.
/// Groups are never formed across block boundaries or across non-unitary
/// operations such as measurements, resets, classical operations, terminators,
/// or region-owning control-flow operations.
struct UnitaryOpGroupingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitaryOpGroupingAnalysis)

  /// Construct the analysis for a function operation.
  ///
  /// If \p op is not a `func.func`, the analysis leaves the group collection
  /// empty.
  explicit UnitaryOpGroupingAnalysis(mlir::Operation *op) {
    performAnalysis(op);
  }

  /// Return all unitary operation groups discovered by the analysis.
  ///
  /// The groups are ordered by the recursive block scan order used by the
  /// analysis.
  const UnitaryOpGroups &getGroups() const { return groups; }

  /// Return the block that contains \p group.
  const mlir::Block *getBlockForGroup(const UnitaryOpGroup &group) const;

  /// Return the group containing \p op, or nullptr if \p op is not grouped.
  ///
  /// Non-unitary operations, null operations, and operations from outside the
  /// analyzed function return nullptr.
  const UnitaryOpGroup *getGroupContainingOp(mlir::Operation *op) const;

  /// Return the unitary groups contained in \p block.
  ///
  /// Returns an empty vector when \p block is null or no groups were found in
  /// the block.
  llvm::SmallVector<const UnitaryOpGroup *>
  getGroupsIn(const mlir::Block *block) const;

  /// Return true if both operations are in the same unitary group.
  ///
  /// Returns false if either operation is null, non-unitary, or not part of a
  /// discovered unitary group.
  bool inSameGroup(mlir::Operation *op1, mlir::Operation *op2) const;

private:
  void performAnalysis(mlir::Operation *operation);
  void scanRegion(mlir::Region &region);
  void scanBlock(mlir::Block &block);
  void buildLookupTables();

  UnitaryOpGroups groups;
  llvm::DenseMap<mlir::Operation *, unsigned> opToGroupIndex;
  llvm::DenseMap<const mlir::Block *, llvm::SmallVector<unsigned>>
      blockToGroupIndices;
};
} // namespace cudaq::quake::detail
