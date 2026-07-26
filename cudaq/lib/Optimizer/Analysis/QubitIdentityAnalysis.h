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
#include "mlir/IR/ValueRange.h"
#include <cstdint>
#include <optional>

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace cudaq::quake::detail {

/// Assigns analysis-local identifiers to virtual qubits represented by scalar
/// `!quake.wire` values within one block of valid Quake value-form IR.
/// `CommutationAnalysis` uses these identifiers to determine whether operations
/// act on the same or disjoint virtual qubits.
///
/// Block arguments, `quake.null_wire`, and `quake.borrow_wire` establish local
/// identities. The analysis propagates them through operators whose controls
/// and targets are all scalar wires, and through scalar-wire results of
/// measurement instruments and reset channels.
///
/// The analysis does not propagate identifiers through reusable controls,
/// conversions, calls, references, aggregates, unsupported non-unitary quantum
/// operations, or block edges. Values that cannot be identified unambiguously
/// remain unidentified. The
/// commutation-aware rewrite driver selectively maintains identities for
/// verified identity-preserving insertions, replacements, and erasures.
/// Unsupported mutations invalidate the owning analysis.
class QubitIdentityAnalysis {
public:
  using QubitId = std::uint32_t;

  explicit QubitIdentityAnalysis(mlir::Block &block);

  /// Return the analysis-local qubit identifier, or no value when identity
  /// cannot be propagated unambiguously.
  std::optional<QubitId> getQubitId(mlir::Value value) const;

  /// Return true only for equal-size ranges whose corresponding values have
  /// the same known analysis-local qubit identities.
  bool haveSameOrderedQubitIdentities(mlir::ValueRange lhs,
                                      mlir::ValueRange rhs) const;

  /// Propagate result identities for a supported all-wire operation.
  /// Classical-only operations succeed without changing identity state.
  /// Return false for unsupported or ambiguous quantum propagation.
  bool registerOperation(mlir::Operation &operation);

  /// Return true when result arity is unchanged and each quantum result is
  /// replaced by a value with the same known analysis-local identity.
  bool replacementPreservesIdentities(mlir::Operation &operation,
                                      mlir::ValueRange replacement) const;

  /// Remove identity mappings for the operation's results only.
  void eraseOperation(mlir::Operation &operation);

private:
  llvm::DenseMap<mlir::Value, QubitId> qubitIds;
};

} // namespace cudaq::quake::detail
