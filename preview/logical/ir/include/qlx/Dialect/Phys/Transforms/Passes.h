/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

//===- Passes.h - Phys transform passes ------------------------*- C++ -*-===//

#ifndef QLX_DIALECT_PHYS_TRANSFORMS_PASSES_H
#define QLX_DIALECT_PHYS_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "qlx/Dialect/Phys/Transforms/ScheduleModel.h"

#include "llvm/ADT/StringRef.h"

#include <string>

namespace qlx::phys {

/// Derive the Tier-3 schedule estimate without mutating or cloning the
/// verified module.  The returned JSON contains only the small typed evidence
/// payload; schedule and graph semantics remain in the retained ModuleOp.
mlir::FailureOr<std::string>
estimateScheduleJSON(mlir::ModuleOp module, llvm::StringRef schedule,
                     llvm::StringRef lowerTier = {}, bool fullWorkload = false);

/// Derive a Tier-3 estimate from an in-process schedule whose complete module
/// was already authenticated and sealed by the compiler. Raw or replayed
/// portable inputs must use estimateScheduleJSON above.
mlir::FailureOr<std::string>
estimateVerifiedScheduleJSON(mlir::ModuleOp module, llvm::StringRef schedule,
                             llvm::StringRef lowerTier = {},
                             bool fullWorkload = false);

/// Derive the same estimate directly from the native scheduler's transient
/// typed rows. A ScheduleOp symbol must remain live for referenced P3 evidence
/// while estimation runs.
mlir::FailureOr<std::string> estimateScheduleJSON(mlir::ModuleOp module,
                                                  llvm::StringRef schedule,
                                                  llvm::StringRef lowerTier,
                                                  NativeScheduleView view,
                                                  bool fullWorkload = false);

/// Schedule one graph and immediately estimate the exact typed rows before
/// their scheduler-owned storage is released. The transient ScheduleOp is
/// independently verified through the same semantic core as portable rows,
/// then erased; only the compact estimate payload is returned.
mlir::FailureOr<std::string> scheduleAndEstimateJSON(
    mlir::ModuleOp module, llvm::StringRef graph, llvm::StringRef schedule,
    llvm::StringRef lowerTier = {}, bool fullWorkload = false);

/// Schedule and estimate a graph in a compiler-owned ModuleOp whose P3
/// refinement was already authenticated in the current transaction. The
/// transient schedule claims still receive the complete semantic proof.
mlir::FailureOr<std::string> scheduleVerifiedAndEstimateJSON(
    mlir::ModuleOp module, llvm::StringRef graph, llvm::StringRef schedule,
    llvm::StringRef lowerTier = {}, bool fullWorkload = false);

#define GEN_PASS_DECL
#include "qlx/Dialect/Phys/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "qlx/Dialect/Phys/Transforms/Passes.h.inc"

} // namespace qlx::phys

#endif // QLX_DIALECT_PHYS_TRANSFORMS_PASSES_H
