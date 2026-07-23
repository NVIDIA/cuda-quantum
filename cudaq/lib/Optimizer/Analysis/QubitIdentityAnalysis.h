/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Value.h"
#include <cstdint>
#include <optional>

namespace mlir {
class Block;
}

namespace cudaq::quake::detail {

/// Assigns analysis-local identifiers to virtual qubits represented by scalar
/// `!quake.wire` and `!quake.control` SSA values within one block of valid
/// Quake value-form IR. Block-local quantum analyses can use these identifiers
/// to determine whether operations act on the same or disjoint virtual qubits.
///
/// Block arguments, `quake.null_wire`, and `quake.borrow_wire` establish local
/// identities. The analysis propagates them through supported operators,
/// measurements, resets, and wire/control conversions. Identity does not imply
/// quantum-state equivalence. For example, measurement and reset preserve the
/// virtual-qubit identity while changing its state.
///
/// The analysis does not propagate identifiers through calls, references,
/// aggregates, or block edges. Values that cannot be identified unambiguously
/// remain unidentified. Any mutation of the block invalidates the analysis.
class QubitIdentityAnalysis {
public:
  using QubitId = std::uint32_t;

  explicit QubitIdentityAnalysis(mlir::Block &block);

  /// Return the analysis-local qubit identifier, or no value when identity
  /// cannot be propagated unambiguously.
  std::optional<QubitId> getQubitId(mlir::Value value) const;

private:
  llvm::DenseMap<mlir::Value, QubitId> qubitIds;
};

} // namespace cudaq::quake::detail
