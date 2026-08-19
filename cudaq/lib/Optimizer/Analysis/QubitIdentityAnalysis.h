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
/// `!quake.wire` values within one block of valid Quake IR. The supported mixed
/// subset contains scalar wire operators and the reference boundaries described
/// below. `CommutationAnalysis` uses these identifiers to determine whether
/// operations act on the same or disjoint virtual qubits.
///
/// `quake.null_wire`, `quake.borrow_wire`, and direct scalar `quake.alloca`
/// operations establish local identities. An allocation identity reaches wires
/// through later unwraps in the same block and remains stable across a matching
/// wrap and unwrap cycle. `quake.wrap_new` binds its reference only when its
/// input wire already has a known identity. A wrap through an untracked
/// reference invalidates active reference bindings because it may write through
/// an alias. Call-like, region-owning, and otherwise unsupported effectful
/// operations also invalidate active reference bindings because the analysis
/// has no summary of captured reference effects. The analysis propagates
/// identities through operators whose controls and targets are all scalar
/// wires, and through scalar-wire results of measurement instruments and reset
/// channels.
///
/// Block arguments remain unidentified because valid IR does not guarantee
/// that their incoming wires are distinct. The analysis also does not propagate
/// identifiers through reusable `!quake.control` values, conversions, calls,
/// reference arguments, reference selections, aggregates, unsupported
/// non-unitary quantum operations, or block edges. Vector allocations and
/// references derived through `quake.extract_ref` or `quake.concat` remain
/// unidentified. Values that cannot be identified unambiguously remain
/// unidentified. Any mutation of the block invalidates the analysis.
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
