/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_LOGICAL_ANALYSIS_CODEPRESENTATION_H
#define CUDAQ_LOGICAL_ANALYSIS_CODEPRESENTATION_H

#include "qlx/Dialect/Fabric/IR/FabricOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace cudaq::logical {

/// Probability-independent GF(2) maps derived from a canonical code profile.
///
/// All physical Paulis use the (X | Z) bit order. The input tables have one
/// 2*n-bit representative per effective syndrome and one X/Z representative
/// per logical qubit. A redundant effective-syndrome basis vector need not be
/// a valid standalone syndrome: each representative is the canonical lift
/// through kept_from_effective, as in CodeProfile.encode_boundary. This is
/// a section on attainable coordinates; its extension to other bitstrings
/// does not make inconsistent redundant syndromes physically attainable.
///
/// This value owns its matrices and does not retain MLIR operations or
/// attributes. Construct it only from verified, linked Fabric code/profile
/// declarations; construction also rechecks every matrix needed by an EDEM.
class CodePresentation {
public:
  static mlir::FailureOr<CodePresentation>
  fromProfile(qlx::fabric::CodeProfileOp profile);

  /// Nominal EDEM type name, derived from the ordered boundary algebra using
  /// the Logical code-presentation/v1 encoding (independent of IR symbols).
  const std::string &name() const { return name_; }
  llvm::ArrayRef<bool> effectiveSigns() const { return effectiveSigns_; }

  unsigned physicalQubits() const { return physicalQubits_; }
  unsigned logicalQubits() const { return logicalQubits_; }
  unsigned effectiveSyndromes() const { return syndromeInputs_.size(); }
  unsigned keptSyndromes() const { return keptStabilizers_.size(); }

  /// Independent code stabilizers, expressed as physical (X | Z) rows.
  llvm::ArrayRef<llvm::SmallBitVector> keptStabilizerBasis() const {
    return keptStabilizers_;
  }
  /// Each effective stabilizer is the XOR of the selected kept basis rows.
  /// The row order matches outputSyndromeFunctionals().
  llvm::ArrayRef<llvm::SmallBitVector> effectiveToKeptDecomposition() const {
    return effectiveToKept_;
  }

  llvm::ArrayRef<llvm::SmallBitVector> inputSyndromeRepresentatives() const {
    return syndromeInputs_;
  }
  llvm::ArrayRef<llvm::SmallBitVector> inputLogicalXRepresentatives() const {
    return logicalXInputs_;
  }
  llvm::ArrayRef<llvm::SmallBitVector> inputLogicalZRepresentatives() const {
    return logicalZInputs_;
  }
  /// The logical input representatives are the canonical code basis rows.
  llvm::ArrayRef<llvm::SmallBitVector> logicalXBasis() const {
    return logicalXInputs_;
  }
  llvm::ArrayRef<llvm::SmallBitVector> logicalZBasis() const {
    return logicalZInputs_;
  }

  /// Each output bit is the GF(2) dot product of this functional with a
  /// physical (X | Z) Pauli. The logical X functionals are derived from the
  /// code's logical Z operators, and conversely for logical Z.
  llvm::ArrayRef<llvm::SmallBitVector> outputSyndromeFunctionals() const {
    return syndromeFunctionals_;
  }
  llvm::ArrayRef<llvm::SmallBitVector> outputLogicalXFunctionals() const {
    return logicalXFunctionals_;
  }
  llvm::ArrayRef<llvm::SmallBitVector> outputLogicalZFunctionals() const {
    return logicalZFunctionals_;
  }

private:
  CodePresentation(unsigned physicalQubits, unsigned logicalQubits)
      : physicalQubits_(physicalQubits), logicalQubits_(logicalQubits) {}

  std::string name_;
  llvm::SmallVector<bool> effectiveSigns_;
  unsigned physicalQubits_;
  unsigned logicalQubits_;
  llvm::SmallVector<llvm::SmallBitVector> keptStabilizers_;
  llvm::SmallVector<llvm::SmallBitVector> effectiveToKept_;
  llvm::SmallVector<llvm::SmallBitVector> syndromeInputs_;
  llvm::SmallVector<llvm::SmallBitVector> logicalXInputs_;
  llvm::SmallVector<llvm::SmallBitVector> logicalZInputs_;
  llvm::SmallVector<llvm::SmallBitVector> syndromeFunctionals_;
  llvm::SmallVector<llvm::SmallBitVector> logicalXFunctionals_;
  llvm::SmallVector<llvm::SmallBitVector> logicalZFunctionals_;
};

} // namespace cudaq::logical

#endif // CUDAQ_LOGICAL_ANALYSIS_CODEPRESENTATION_H
