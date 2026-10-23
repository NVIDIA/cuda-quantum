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

  /// Dimension of the linear space of deterministic full-outcome parities.
  /// Affine constants determine their values, not the dimension.
  unsigned outcomeDeterministicRelationDimension() const;

private:
  friend llvm::Expected<IdealOutcomeAnalysis>
  analyzeIdealOutcomes(const Projection &projection,
                       llvm::ArrayRef<IdealBoundaryPort> inputPorts,
                       llvm::ArrayRef<IdealBoundaryPort> outputPorts);

  std::vector<IdealAffineBit> outcomes_;
  unsigned inputCount_ = 0;
  unsigned recordCount_ = 0;
  unsigned randomBits_ = 0;
};

/// Analyze only the typed projection; this never parses Stim text or modifies
/// canonical MLIR. Invalid, ambiguous, or unsupported projections fail closed.
llvm::Expected<IdealOutcomeAnalysis>
analyzeIdealOutcomes(const Projection &projection,
                     llvm::ArrayRef<IdealBoundaryPort> inputPorts,
                     llvm::ArrayRef<IdealBoundaryPort> outputPorts);

} // namespace cudaq::logical

#endif // CUDAQ_LOGICAL_ANALYSIS_IDEALOUTCOMES_H
