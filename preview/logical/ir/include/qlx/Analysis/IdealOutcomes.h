/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_ANALYSIS_IDEALOUTCOMES_H
#define CUDAQ_LOGICAL_ANALYSIS_IDEALOUTCOMES_H

#include "qlx/Target/Fabric/EmitStim.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace cudaq::logical {

/// An ideal measurement bit as an affine function of independent uniformly
/// random bits. Bit (k % 64) of `randomWords[k / 64]` is the coefficient of
/// random variable k. A reset can introduce variables without an exposed
/// measurement record. Every returned expression has the final random width.
struct IdealAffineBit {
  bool constant = false;
  std::vector<std::uint64_t> randomWords;
};

/// Encoded boundary state, in the code profile's canonical basis. Every row
/// has 2 * carriers.size() bits in (X | Z) order. Input stabilizers have
/// independent random signs (one explicit input syndrome bit each); logical
/// X/Z rows are Choi-paired with inaccessible references. Output analysis
/// measures only the independent stabilizers.
struct IdealBoundaryPort {
  std::vector<unsigned> carriers;
  std::vector<llvm::SmallBitVector> keptStabilizers;
  std::vector<llvm::SmallBitVector> logicalX;
  std::vector<llvm::SmallBitVector> logicalZ;
};

/// A bounded, exact logical objective, independent of its realization.
/// Bindings map expected quantum formal indices to the flattened input/output
/// logical boundary indices and must cover each boundary exactly once. Empty
/// bindings retain the legacy identity convention only for Clifford actions.
///
/// For a Clifford action, row r < arity is the image of X_r, row arity + r
/// the image of Z_r; every row has 2 * arity bits in (X | Z) order and its
/// phase is the sign in the Hermitian Pauli convention.
///
/// Preparation describes one signed X/Z eigenstate. Measurement describes a
/// signed X/Z measurement of formal `target`, either discarding every input
/// or preserving all logical inputs. Its mapped classical result is the XOR
/// of the selected realization records, kept input syndrome bits, and the
/// affine constant. No outcome or hidden randomness is postselected.
struct IdealLogicalAction {
  unsigned arity = 0;
  std::vector<llvm::SmallBitVector> images;
  std::vector<bool> phases;
  enum class Kind { Clifford, Preparation, Measurement };
  enum class Basis { X, Z };
  Kind kind = Kind::Clifford;
  Basis basis = Basis::Z;
  bool negative = false;
  std::vector<unsigned> inputBindings;
  std::vector<unsigned> outputBindings;
  unsigned target = 0;
  std::vector<unsigned> outcomeRecords;
  std::vector<unsigned> outcomeInputSyndromes;
  bool outcomeConstant = false;
  bool destructive = true;
};

/// Outcome of comparing one expected Choi stabilizer with the realization's
/// post-gadget Choi state.
enum class IdealActionRowStatus {
  /// The expected signed Pauli is in the final stabilizer group, with the
  /// declared sign, independent of every random outcome.
  Verified,
  /// The expected Pauli, up to sign, is not a stabilizer of the final state.
  NotStabilized,
  /// The expected Pauli is stabilized but with the opposite deterministic sign.
  SignMismatch,
  /// The expected Pauli is stabilized, but its sign depends on random ideal
  /// outcomes.
  RandomSign,
};

/// Result of the optional logical-action comparison.
struct IdealActionReport {
  enum class Status { Unchecked, Unsupported, Checked };
  Status status = Status::Unchecked;
  /// Why the declared action could not be compared (Unsupported only).
  std::string reason;
  /// One entry per checked signed quantum/classical relation (Checked only).
  std::vector<IdealActionRowStatus> rows;
  /// Descriptive relation labels, in the same order as rows.
  std::vector<std::string> labels;

  bool verified() const {
    if (status != Status::Checked || rows.empty())
      return false;
    for (IdealActionRowStatus row : rows)
      if (row != IdealActionRowStatus::Verified)
        return false;
    return true;
  }
};

/// Exact measurement relations for a selected, flattened Clifford gadget.
/// Input stabilizer signs are independent variables, logical qubits are
/// Choi-paired with inaccessible references. A relation is accepted only if
/// it holds for every encoded logical input and input syndrome sector.
class IdealOutcomeAnalysis {
public:
  llvm::ArrayRef<IdealAffineBit> inputSyndromes() const {
    return llvm::ArrayRef<IdealAffineBit>(outcomes_).slice(0, inputCount_);
  }
  llvm::ArrayRef<IdealAffineBit> outputSyndromes() const {
    return llvm::ArrayRef<IdealAffineBit>(outcomes_).drop_front(inputCount_ +
                                                                recordCount_);
  }
  /// Concatenation of input syndrome bits, realization records, and output
  /// independent stabilizer checks.
  llvm::ArrayRef<IdealAffineBit> outcomes() const { return outcomes_; }
  unsigned independentRandomBits() const { return randomBits_; }

  /// Comparison of the post-gadget Choi state with the declared logical
  /// action; Unchecked when no action was supplied.
  const IdealActionReport &action() const { return action_; }

  /// Dimension of the linear space of deterministic full-outcome parities.
  /// Affine constants determine their values, not the dimension.
  unsigned outcomeDeterministicRelationDimension() const;

private:
  friend llvm::Expected<IdealOutcomeAnalysis>
  analyzeIdealOutcomes(const Projection &projection,
                       llvm::ArrayRef<IdealBoundaryPort> inputPorts,
                       llvm::ArrayRef<IdealBoundaryPort> outputPorts,
                       const IdealLogicalAction *action);

  std::vector<IdealAffineBit> outcomes_;
  IdealActionReport action_;
  unsigned inputCount_ = 0;
  unsigned recordCount_ = 0;
  unsigned randomBits_ = 0;
};

/// Analyze only the typed projection; this never parses Stim text or modifies
/// canonical MLIR. Invalid, ambiguous, or unsupported projections fail closed.
/// With a declared `action`, also compares the post-gadget Choi state (after
/// the output stabilizers are measured) with the Choi state of that action,
/// stabilizer by stabilizer and sign by sign. An action that cannot be
/// expressed over the boundary's logical qubits is reported as Unsupported
/// rather than as an error.
llvm::Expected<IdealOutcomeAnalysis>
analyzeIdealOutcomes(const Projection &projection,
                     llvm::ArrayRef<IdealBoundaryPort> inputPorts,
                     llvm::ArrayRef<IdealBoundaryPort> outputPorts,
                     const IdealLogicalAction *action = nullptr);

} // namespace cudaq::logical

#endif // CUDAQ_LOGICAL_ANALYSIS_IDEALOUTCOMES_H
