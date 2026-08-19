/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <utility>

namespace mlir {
class Block;
class Operation;
} // namespace mlir

namespace cudaq::quake::detail {

class QubitIdentityAnalysis;

/// The outcome of a commutation query.
enum class commutation_status { commutes, does_not_commute, indeterminate };

/// The rule or limitation that produced a commutation status.
enum class commutation_reason {
  // Reasons paired with commutation_status::commutes.
  /// The operations have disjoint block-local quantum support.
  disjoint_support,
  /// The recognized operations have the same structural action and placement,
  /// optionally with opposite adjoint states.
  same_operation,
  /// Both operations are diagonal in the computational basis.
  computational_diagonal,
  /// Both operations rotate about the same axis.
  same_axis,
  /// The unitary channel is diagonal in the recognized measurement
  /// instrument's basis.
  measurement_instrument_basis,
  /// The unitary channel preserves the reset channel's output state.
  preserved_reset_state,
  /// Pauli products have even anti-commutation parity on shared targets.
  even_pauli_parity,
  /// A diagonal operation overlaps the other operation only on controls.
  diagonal_on_controls,
  /// Controlled operations have commuting target actions and no target-control
  /// crossover.
  compatible_controlled_targets,
  /// Opposite polarity on a shared control makes the control predicates
  /// mutually exclusive.
  mutually_exclusive_controls,

  // Reasons paired with commutation_status::does_not_commute.
  /// Exact Pauli operators have odd anti-commutation parity on shared targets.
  odd_pauli_parity,

  // Reasons paired with commutation_status::indeterminate.
  /// At least one query operation is null.
  null_operation,
  /// At least one operation is outside the analyzed block.
  different_blocks,
  /// At least one operation has no supported analysis operation view.
  unsupported_operation_kind,
  /// A quantum operand is not a supported scalar wire value.
  unsupported_quantum_operand_type,
  /// A quantum operand has no analysis-local qubit identifier.
  unmapped_qubit_id,
  /// An operation uses the same virtual qubit in more than one control or
  /// target position.
  duplicate_qubit_operand,
  /// An `ExpPauli` word is dynamic.
  unsupported_pauli_word,
  /// Supported operations did not satisfy an available structural rule.
  no_applicable_rule
};

/// Return the stable textual identifier for a commutation reason.
llvm::StringRef getCommutationReasonId(commutation_reason reason);

/// A structural commutation outcome and its classification.
struct CommutationResult {
  commutation_status status;
  commutation_reason reason;
};

/// Exact block-local commutation analysis for supported Quake quantum
/// operations.
///
/// A query asks whether two operations have the same induced action when
/// composed in either order. Gate relations require exact unitary-operator
/// equality, including global phase. Supported measurement instruments require
/// equality outcome by outcome, preserving each classical outcome label and
/// conditional output state. Reset relations use quantum-channel equality.
///
/// The block must contain verifier-valid Quake IR.
///
/// Candidate operations use scalar `!quake.wire` values and must implement
/// Quake `OperatorInterface` or be single-target measurement instruments, reset
/// channels, or sinks. Identity can be established by `quake.null_wire`,
/// `quake.borrow_wire`, direct scalar allocation followed by unwrap, and
/// `quake.wrap_new` from a known wire. A matching wrap and unwrap preserves the
/// allocation binding. Supported scalar-wire operations propagate identity.
///
/// Operations on disjoint qubits commute regardless of their supported
/// operation kind. For overlapping qubits, the analysis applies structural
/// rules for recognized built-in Quake operators, computational-basis
/// measurement instruments, and unitary channels that preserve the reset
/// output state.
/// Custom unitaries with the same defining symbol, exact parameters, controls,
/// and targets are also recognized as the same operation. The analysis does not
/// inspect custom-unitary matrices, analyze arbitrary quantum-channel or
/// measurement-instrument representations, or infer overlapping-support
/// semantics from different custom definitions or dynamic Pauli words.
///
/// `does_not_commute` is returned only for the limited cases where an available
/// rule proves that the operations do not commute. `indeterminate` means that
/// the available rules established neither result. It does not imply either
/// commutation or a failure to commute.
///
/// Compiler transformations must treat both `does_not_commute` and
/// `indeterminate` as not safe to reorder. The separate statuses preserve the
/// distinction between a proven failure to commute and the absence of a proof.
///
/// A wrap through an unidentified reference, a call-like operation, a region
/// owner, or another unsupported operation with memory effects invalidates
/// active reference bindings because the analysis has no alias or
/// captured-effect summary.
/// Shared-support non-unitary rules cover only single-target Mz instruments
/// with computational-basis-diagonal unitaries and reset-channel relations;
/// sinks and pairs of non-unitary operations remain indeterminate. Reusable
/// `!quake.control` values, conversions, call results, reference arguments,
/// reference selections, aggregates, unsupported non-unitary operations, and
/// references derived through `quake.extract_ref` or `quake.concat` do not
/// establish identity. Each wire block argument remains unidentified because
/// function entries, CFG edges, and nested regions do not guarantee that
/// incoming wires are distinct. Operations reached only from those arguments
/// therefore remain indeterminate.
///
/// Any mutation of the block invalidates the analysis instance. The caller
/// must discard it before querying the changed block.
class CommutationAnalysis {
public:
  explicit CommutationAnalysis(mlir::Block &block);
  ~CommutationAnalysis();

  CommutationAnalysis(const CommutationAnalysis &) = delete;
  CommutationAnalysis &operator=(const CommutationAnalysis &) = delete;

  /// Return the detailed symmetric relation between two operations.
  CommutationResult getResult(mlir::Operation *lhs, mlir::Operation *rhs);

  /// Return true only when exact commutation has been proven.
  bool canCommute(mlir::Operation *lhs, mlir::Operation *rhs);

private:
  mlir::Block *block;
  std::unique_ptr<QubitIdentityAnalysis> qubitIdentity;
  llvm::DenseMap<std::pair<mlir::Operation *, mlir::Operation *>,
                 CommutationResult>
      cache;
};

} // namespace cudaq::quake::detail
