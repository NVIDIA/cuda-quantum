/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_DIALECT_PHYS_TRANSFORMS_SCHEDULEMODEL_H
#define QLX_DIALECT_PHYS_TRANSFORMS_SCHEDULEMODEL_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace qlx::phys {

using NativeScheduleResourceId = unsigned;

/// One immutable row produced by the native scheduler. StringRefs point into
/// the live MLIR module or scheduler-owned storage and are valid only for the
/// synchronous NativeScheduleView lifetime.
struct NativeScheduleRow {
  llvm::StringRef id;
  llvm::StringRef kind;
  double start = 0.0;
  double duration = 0.0;
  llvm::SmallVector<NativeScheduleResourceId, 4> resourceIds;
  llvm::SmallVector<std::pair<NativeScheduleResourceId, unsigned>, 4>
      inputResourcePositions;
  bool stateOnlyBoundary = true;
  llvm::SmallVector<llvm::StringRef, 4> dependencies;
  llvm::SmallVector<llvm::StringRef, 4> dataDependencies;
  llvm::SmallVector<llvm::StringRef, 4> resourceDependencies;
  llvm::SmallVector<llvm::StringRef, 4> domainDependencies;
  llvm::StringRef parent;
  llvm::StringRef branch;
  llvm::StringRef condition;
  std::optional<int64_t> maxAttempts;
  llvm::StringRef commitPoint;
  std::optional<int64_t> repeatCount;
  std::optional<double> repeatPeriod;
  std::optional<double> repeatEpilogue;
  std::optional<int64_t> maxIterations;
  llvm::StringRef callee;
  llvm::StringRef instance;
  llvm::StringRef profile;
  llvm::StringRef templateEvent;
  llvm::StringRef attempt;
  llvm::StringRef attemptEvent;
  llvm::StringRef decisionEvent;
  llvm::StringRef exhaustion;
  std::optional<double> successProbability;
  llvm::StringRef successProbabilitySource;
  llvm::StringRef successProbabilityEvidence;

  double finish() const { return start + duration; }
};

/// A non-owning synchronous handoff from native scheduling to native
/// estimation. It is never retained as an alternative program authority.
struct NativeScheduleView {
  llvm::ArrayRef<NativeScheduleRow> rows;
  llvm::ArrayRef<std::string> resourceLabels;
};

} // namespace qlx::phys

#endif // QLX_DIALECT_PHYS_TRANSFORMS_SCHEDULEMODEL_H
