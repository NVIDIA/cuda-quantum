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
enum class CommutationStatus { Commutes, DoesNotCommute, Indeterminate };

/// The rule or limitation that produced a commutation status.
enum class CommutationReason {
  // Reasons paired with CommutationStatus::Commutes.
  /// The operations have disjoint block-local quantum support.
  DisjointSupport,
  /// The recognized operations have the same structural action and placement,
  /// optionally with opposite adjoint states.
  SameOperation,
  /// Both operations are diagonal in the computational basis.
  ComputationalDiagonal,
  /// Both operations rotate about the same axis.
  SameAxis,
  /// The unitary channel is diagonal in the recognized measurement
  /// instrument's basis.
  MeasurementInstrumentBasis,
  /// The unitary channel preserves the reset channel's output state.
  PreservedResetState,
  /// Pauli products have even anti-commutation parity on shared targets.
  EvenPauliParity,
  /// A diagonal operation overlaps the other operation only on controls.
  DiagonalOnControls,
  /// Controlled operations have commuting target actions and no target-control
  /// crossover.
  CompatibleControlledTargets,
  /// Opposite polarity on a shared control makes the control predicates
  /// mutually exclusive.
  MutuallyExclusiveControls,

  // Reasons paired with CommutationStatus::DoesNotCommute.
  /// Exact Pauli operators have odd anti-commutation parity on shared targets.
  OddPauliParity,

  // Reasons paired with CommutationStatus::Indeterminate.
  /// At least one query operation is null.
  NullOperation,
  /// At least one operation is outside the analyzed block.
  DifferentBlocks,
  /// At least one operation has no supported analysis operation view.
  UnsupportedOperationKind,
  /// A quantum operand is not a supported scalar wire value.
  UnsupportedQuantumOperandType,
  /// A quantum operand has no analysis-local qubit identifier.
  UnmappedQubitId,
  /// An operation uses the same virtual qubit in more than one control or
  /// target position.
  DuplicateQubitOperand,
  /// An `ExpPauli` word is dynamic.
  UnsupportedPauliWord,
  /// Supported operations did not satisfy an available structural rule.
  NoApplicableRule
};

/// Return the stable textual identifier for a commutation reason.
llvm::StringRef getCommutationReasonId(CommutationReason reason);

/// A structural commutation outcome and its classification.
struct CommutationResult {
  CommutationStatus status;
  CommutationReason reason;

  /// True only when structural analysis proved exact commutation.
  explicit operator bool() const {
    return status == CommutationStatus::Commutes;
  }
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
/// rules for recognized built-in Quake operators, matching-basis measurement
/// instruments, and unitary channels that preserve the reset output state.
/// Custom unitaries with the same defining symbol, exact parameters, controls,
/// and targets are also recognized as the same operation. The analysis does not
/// inspect custom-unitary matrices, analyze arbitrary quantum-channel or
/// measurement-instrument representations, or infer overlapping-support
/// semantics from different custom definitions or dynamic Pauli words.
///
/// `DoesNotCommute` is returned only for the limited cases where an available
/// rule proves that the operations do not commute. `Indeterminate` means that
/// the available rules established neither result. It does not imply either
/// commutation or a failure to commute.
///
/// Compiler transformations must treat both `DoesNotCommute` and
/// `Indeterminate` as not safe to reorder. The separate statuses preserve the
/// distinction between a proven failure to commute and the absence of a proof.
///
/// A wrap through an unidentified reference, a call-like operation, a region
/// owner, or another unsupported effectful operation invalidates active
/// reference bindings because the analysis has no alias or captured-effect
/// summary.
/// Shared-support non-unitary rules cover only single-target matching-basis
/// measurement instruments and reset-channel relations with unitary channels;
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
