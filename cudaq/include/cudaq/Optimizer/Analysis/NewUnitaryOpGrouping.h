/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include <optional>

namespace mlir {
class Operation;
class Block;
} // namespace mlir

namespace cudaq::quake::detail {

// output of the Unitary Op Grouping Analysis
struct NewUnitaryOpGroup {
  NewUnitaryOpGroup() = default;

  mlir::Block *containingBlock;
  mlir::SmallVector<mlir::Operation *, 8> ops;
  mlir::SmallVector<mlir::Operation *, 2> trailingDelimiterOps;
};

using NewUnitaryOpGroups = mlir::SmallVector<NewUnitaryOpGroup>;

// Role of an operation during segment creation
enum class SegmentOpRole { Unitary, MsmtDelimiter, ResetDelimiter };

enum class OrderingMode {
  Textual, // use the order as it appears in the IR block
  WireDataflow
};

/// Window in this context is a maximal contiguous portion of the
/// `containingBlock` consisting of unitary operations.
struct QuantumOpSegment {
  mlir::Block *containingBlock = nullptr;
  mlir::SmallVector<mlir::Operation *> opsInBlockOrder;
};

/// Helper data structure for creating order for `CanonicalSegmentOrder`
struct SegmentDependencyGraph {
  mlir::Block *containingBlock = nullptr;
  OrderingMode mode = OrderingMode::Textual;

  // all nodes in the segment in the order they appear in the IR block
  mlir::SmallVector<mlir::Operation *> nodesInBlockOrder;

  // map of op to original position in the IR block
  mlir::DenseMap<mlir::Operation *, unsigned> originalPositionByOp;

  // Adjacency list
  mlir::DenseMap<mlir::Operation *, mlir::SmallVector<mlir::Operation *, 4>>
      successorsByOp;

  // number of predecessors of each op in the segment
  mlir::DenseMap<mlir::Operation *, unsigned> predecessorCountByOp;
};

/// Becomes the canonical order of the segment after policies are ordering
/// policies are applied to the `SegmentDependencyGraph`
struct CanonicalSegmentOrder {
  mlir::Block *containingBlock = nullptr;
  OrderingMode mode = OrderingMode::Textual;
  mlir::SmallVector<mlir::Operation *> opsInCanonicalOrder;
};

struct NewUnitaryOpGroupingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NewUnitaryOpGroupingAnalysis)

  explicit NewUnitaryOpGroupingAnalysis(mlir::Operation *op) {
    performAnalysis(op);
  }

  const NewUnitaryOpGroups &getGroups() const { return unitaryOpGroups; }
  std::optional<unsigned> getGroupIndexForOp(mlir::Operation *op) const;
  const mlir::Block *getBlockForGroup(const NewUnitaryOpGroup &group) const;
  const NewUnitaryOpGroup *getGroupContainingOp(mlir::Operation *op) const;
  mlir::SmallVector<const NewUnitaryOpGroup *>
  getGroupsIn(const mlir::Block *block) const;
  bool inSameGroup(mlir::Operation *lhs, mlir::Operation *rhs) const;

private:
  NewUnitaryOpGroups unitaryOpGroups;
  mlir::DenseMap<mlir::Operation *, unsigned> opToGroupIndex;
  mlir::DenseMap<const mlir::Block *, mlir::SmallVector<unsigned>>
      blockToGroupIndices;

  void performAnalysis(mlir::Operation *op);
  void analyzeBlock(mlir::Block &block);
  CanonicalSegmentOrder
  computeCanonicalSegmentOrder(SegmentDependencyGraph &sdg);
  void formUnitaryGroups(const CanonicalSegmentOrder &cso);

  mlir::DenseMap<mlir::Operation *, mlir::Operation *> dependenciesByOp;
};
} // namespace cudaq::quake::detail
