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

/// A nonempty group emitted from one quantum-operation segment's selected
/// virtual order.
///
/// A group contains a possibly empty run of unitary operations followed by a
/// possibly empty run of measurement/reset delimiters. At least one run is
/// nonempty, so delimiter-only groups are valid. No group crosses a segment or
/// block boundary, and hard-boundary operations are excluded.
///
/// The block and operation pointers are non-owning. The grouping result remains
/// meaningful only while the analyzed IR is unchanged.
struct UnitaryOpGroup {
  UnitaryOpGroup() = default;

  /// Non-owning block containing every operation in this group.
  /// This is non-null for every group emitted by the analysis.
  mlir::Block *block = nullptr;

  /// Possibly empty unitary run in the selected virtual order.
  /// In wire-dataflow mode, this order may differ from physical block order.
  mlir::SmallVector<mlir::Operation *, 8> ops;

  /// Possibly empty measurement/reset run following the unitary run in the
  /// selected virtual order. These delimiter operations are full group members.
  mlir::SmallVector<mlir::Operation *, 2> trailingDelimiterOps;
};

using UnitaryOpGroups = mlir::SmallVector<UnitaryOpGroup>;

/// Transient classification of an operation retained in a quantum-operation
/// segment. An operation without a role terminates the segment and is excluded
/// from its dependency graph.
enum class SegmentOpRole { Unitary, MsmtDelimiter, ResetDelimiter };

/// Transient strategy used to select one segment's virtual order.
enum class OrderingMode {
  /// Preserve the order in which segment operations appear in the block.
  Textual,

  /// Topologically order scalar-wire operations using intra-segment SSA
  /// def-use and known-logical-qubit dependencies. Among dependency-ready
  /// operations, prefer a unitary over a measurement/reset delimiter; within
  /// either class, prefer original segment order.
  WireDataflow
};

/// A transient maximal contiguous run of unitary, measurement, and reset
/// operations in one block. Any other operation ends the segment and is
/// excluded from it.
struct QuantumOpSegment {
  mlir::Block *containingBlock = nullptr;
  mlir::SmallVector<mlir::Operation *> opsInBlockOrder;
};

/// Transient dependency graph for one segment. An edge A -> B means A must
/// precede B in the virtual canonical order.
struct SegmentDependencyGraph {
  mlir::Block *containingBlock = nullptr;
  OrderingMode mode = OrderingMode::Textual;

  /// All nodes in their original segment order.
  mlir::SmallVector<mlir::Operation *> nodesInBlockOrder;

  /// Zero-based rank in original segment order and authoritative membership
  /// map.
  mlir::DenseMap<mlir::Operation *, unsigned> originalPositionByOp;

  /// Directed predecessor-to-successor adjacency list, including empty entries
  /// for nodes with no successors.
  mlir::DenseMap<mlir::Operation *, mlir::SmallVector<mlir::Operation *, 4>>
      successorsByOp;

  /// In-degree table, including zero entries, consumed by canonical ordering.
  mlir::DenseMap<mlir::Operation *, unsigned> predecessorCountByOp;
};

/// Transient deterministic virtual order produced for one segment. Computing
/// this order does not modify the IR.
struct CanonicalSegmentOrder {
  mlir::Block *containingBlock = nullptr;
  OrderingMode mode = OrderingMode::Textual;
  mlir::SmallVector<mlir::Operation *> opsInCanonicalOrder;
};

/// Analyze a function and form block-local groups of unitary operations and
/// their trailing measurement/reset delimiters.
///
/// The analysis recursively visits blocks in nested regions. Any operation not
/// classified as a unitary, measurement, or reset is a hard boundary: it is
/// excluded from groups, and groups never cross it or a block boundary.
///
/// A segment uses wire-dataflow ordering only when every operation has
/// supported scalar-wire flow and every quantum input has a known logical
/// identity. Otherwise, the entire segment preserves textual order. The
/// analysis computes only virtual orders and does not mutate the IR.
struct UnitaryOpGroupingAnalysis {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnitaryOpGroupingAnalysis)

  /// Construct the analysis for \p op.
  ///
  /// If \p op is not itself a `func.func`, the result is empty. Results remain
  /// meaningful only while the analyzed IR is unchanged.
  explicit UnitaryOpGroupingAnalysis(mlir::Operation *op) {
    performAnalysis(op);
  }

  /// Return all groups in deterministic recursive operation/region discovery
  /// order.
  ///
  /// The returned reference is borrowed and valid only for this analysis
  /// object's lifetime.
  const UnitaryOpGroups &getGroups() const { return unitaryOpGroups; }

  /// Return `i` such that `getGroups()[i]` contains \p op.
  ///
  /// Return `std::nullopt` when \p op is null or ungrouped. Both unitary and
  /// trailing-delimiter operations are indexed. Indices are stable only for
  /// this analysis instance.
  std::optional<unsigned> getGroupIndexForOp(mlir::Operation *op) const;

  /// Return the non-owning block pointer recorded in \p group.
  const mlir::Block *getBlockForGroup(const UnitaryOpGroup &group) const;

  /// Return a borrowed pointer to the group containing \p op.
  ///
  /// Both unitary and trailing-delimiter operations are group members. Return
  /// `nullptr` when \p op is null or ungrouped. The returned pointer is valid
  /// only for this analysis object's lifetime.
  const UnitaryOpGroup *getGroupContainingOp(mlir::Operation *op) const;

  /// Return borrowed pointers to groups contained directly in \p block,
  /// excluding groups in nested blocks and preserving discovery order.
  ///
  /// A null block or a block without groups produces an empty vector. Returned
  /// pointers are valid only for this analysis object's lifetime.
  mlir::SmallVector<const UnitaryOpGroup *>
  getGroupsIn(const mlir::Block *block) const;

  /// Return true when both operations, including trailing delimiters, belong to
  /// the same group.
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
