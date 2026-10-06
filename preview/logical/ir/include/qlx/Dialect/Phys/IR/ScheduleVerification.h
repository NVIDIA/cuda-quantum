/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_PHYS_IR_SCHEDULEVERIFICATION_H
#define QLX_DIALECT_PHYS_IR_SCHEDULEVERIFICATION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>

namespace qlx::phys {

class ScheduleOp;

/// Neutral, non-owning semantic claims for one P3 schedule event.  Both the
/// portable row parser and the native scheduler adapt into this IR-owned form
/// before invoking the independent verifier below.
struct ScheduleClaim {
  llvm::StringRef id;
  llvm::StringRef kind;
  double start = 0.0;
  double duration = 0.0;
  llvm::SmallVector<llvm::StringRef, 4> resources;
  llvm::SmallVector<llvm::StringRef, 4> dependencies;
  llvm::SmallVector<llvm::StringRef, 4> dataDependencies;
  llvm::SmallVector<llvm::StringRef, 4> resourceDependencies;
  llvm::SmallVector<llvm::StringRef, 4> domainDependencies;
  llvm::StringRef parent;
  llvm::StringRef branch;
  llvm::StringRef condition;
  llvm::StringRef callee;
  llvm::StringRef instance;
  llvm::StringRef profile;
  llvm::StringRef templateEvent;
  llvm::StringRef attempt;
  llvm::StringRef attemptEvent;
  llvm::StringRef decisionEvent;
  llvm::StringRef commitPoint;
  llvm::StringRef exhaustion;
  llvm::StringRef successProbabilitySource;
  llvm::StringRef successProbabilityEvidence;
  std::optional<int64_t> maxAttempts;
  std::optional<int64_t> repeatCount;
  std::optional<double> repeatPeriod;
  std::optional<double> repeatEpilogue;
  std::optional<int64_t> maxIterations;
  std::optional<double> successProbability;
  bool hasDataDependencies = false;
  bool hasResourceDependencies = false;
  bool hasDomainDependencies = false;

  double finish() const { return start + duration; }
};

/// Structural work evidence from one invocation of the common semantic core.
struct ScheduleVerificationStats {
  uint64_t semanticVerifierRuns = 0;
  uint64_t claimsVisited = 0;
  uint64_t hierarchyPostorderVisits = 0;
  uint64_t frontierStateCopies = 0;
  uint64_t frontierJournalTouches = 0;
};

/// Independently reconstruct and verify all schedule semantics from claims.
/// The claims are not trusted scheduler state: the graph, allocation mappings,
/// folded envelopes, timing profile, dependencies, and resource exclusions
/// are re-derived from the ScheduleOp's referenced IR.
mlir::LogicalResult
verifyScheduleClaims(ScheduleOp schedule, llvm::ArrayRef<ScheduleClaim> claims,
                     ScheduleVerificationStats *stats = nullptr);

} // namespace qlx::phys

#endif // QLX_DIALECT_PHYS_IR_SCHEDULEVERIFICATION_H
