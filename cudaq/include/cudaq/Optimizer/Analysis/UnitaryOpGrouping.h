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

/// A group in one op segment's virtual canonical order.
///
/// A group contains zero or more unitary operations followed by zero or more
/// measurement/reset delimiters. Delimiter-only groups are valid. Hard-boundary
/// operations are excluded. All pointers are non-owning and remain valid only
/// while the analyzed IR remains valid.
struct UnitaryOpGroup {
  UnitaryOpGroup() = default;

  /// Block containing every operation in this group.
  mlir::Block *block = nullptr;

  /// Unitary operations in virtual canonical order.
  mlir::SmallVector<mlir::Operation *, 8> ops;

  /// Consecutive measurements/resets that terminate this group.
  mlir::SmallVector<mlir::Operation *, 2> trailingDelimiterOps;
};

using UnitaryOpGroups = mlir::SmallVector<UnitaryOpGroup>;

/// Classification of an operation retained in a quantum-operation segment.
enum class SegmentOpRole { Unitary, MsmtDelimiter, ResetDelimiter };

enum class OrderingMode {
  /// Preserve the order in which operations appear in the block.
  Textual,

  /// Apply dependency ordering for scalar wires with known qubit identities.
  /// The ordering policy is:
  /// 1. prefer unitary ops over resets/measurements
  /// 2. prefer ops that happen earlier in block order
  WireDataflow
};

/// A maximal contiguous run of unitary, measurement, and reset operations in
/// one block. Any other operation ends the segment and is excluded from it.
struct QuantumOpSegment {
  mlir::Block *containingBlock = nullptr;
  mlir::SmallVector<mlir::Operation *> opsInBlockOrder;
};

/// Dependency graph for one segment. An edge A -> B means A must precede B in
/// the virtual canonical order.
struct SegmentDependencyGraph {
  mlir::Block *containingBlock = nullptr;
  OrderingMode mode = OrderingMode::Textual;

  /// All nodes in the order they appear in the block.
  mlir::SmallVector<mlir::Operation *> nodesInBlockOrder;

  /// Zero-based rank within this segment's original block order.
  mlir::DenseMap<mlir::Operation *, unsigned> originalPositionByOp;

  /// Directed predecessor-to-successor adjacency list.
  mlir::DenseMap<mlir::Operation *, mlir::SmallVector<mlir::Operation *, 4>>
      successorsByOp;

  /// In-degree table consumed while computing canonical order.
  mlir::DenseMap<mlir::Operation *, unsigned> predecessorCountByOp;
};

/// Virtual deterministic order produced for one segment. The IR is not
/// modified.
struct CanonicalSegmentOrder {
  mlir::Block *containingBlock = nullptr;
  OrderingMode mode = OrderingMode::Textual;
  mlir::SmallVector<mlir::Operation *> opsInCanonicalOrder;
};

/// Analyze a function and form block-local unitary-operation groups.
///
/// The analysis recursively visits blocks in nested regions. Groups never
/// cross blocks or unclassified hard-boundary operations. Segments containing
/// only supported scalar-wire operations with known qubit identities use
/// dependency ordering; other segments preserve textual block order. Analysis
/// computes only a virtual order and does not mutate the IR. A non-function
/// input produces no groups.
struct UnitaryOpGroupingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitaryOpGroupingAnalysis)

  explicit UnitaryOpGroupingAnalysis(mlir::Operation *op) {
    performAnalysis(op);
  }

  /// Return all groups in recursive block-visitation order.
  const UnitaryOpGroups &getGroups() const { return unitaryOpGroups; }

  /// Return the index of the group containing \p op, if one exists.
  ///
  /// Both unitary and trailing-delimiter operations are group members.
  std::optional<unsigned> getGroupIndexForOp(mlir::Operation *op) const;

  /// Return the block containing p group.
  const mlir::Block *getBlockForGroup(const UnitaryOpGroup &group) const;

  /// Return the group containing \p op, or nullptr when it is not grouped.
  const UnitaryOpGroup *getGroupContainingOp(mlir::Operation *op) const;

  /// Return groups contained directly in  block.
  /// A null block or a block without groups produces an empty vector.
  mlir::SmallVector<const UnitaryOpGroup *>
  getGroupsIn(const mlir::Block *block) const;

  /// Return true when both operations belong to the same group.
  ///
  /// Returns false when either operation is null or ungrouped.
  bool inSameGroup(mlir::Operation *lhs, mlir::Operation *rhs) const;

private:
  UnitaryOpGroups unitaryOpGroups;
  mlir::DenseMap<mlir::Operation *, unsigned> opToGroupIndex;
  mlir::DenseMap<const mlir::Block *, mlir::SmallVector<unsigned>>
      blockToGroupIndices;

  void performAnalysis(mlir::Operation *op);
  void analyzeBlock(mlir::Block &block);
  CanonicalSegmentOrder
  computeCanonicalSegmentOrder(SegmentDependencyGraph &sdg);
  void formUnitaryGroups(const CanonicalSegmentOrder &cso);
};
} // namespace cudaq::quake::detail
