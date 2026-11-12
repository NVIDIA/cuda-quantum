/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
//
// Linear single-ownership analysis.
//
// Values whose type implements `cudaq::LinearTypeInterface` (logical qubits,
// patches, resource payloads, physical states, linear event handles, frames)
// must have exactly one consuming use on every execution path. Plain SSA does
// not enforce this. The analysis below does, driven only by the interface and
// the op traits declared in CUDA-Q core
// (cudaq/Optimizer/Dialect/LinearOwnershipInterfaces.td), so it names no
// dialect.
//
// Per value it checks for four possible violations:
//
//   double-consume        two consuming uses can both run.
//   loop-capture-consume  a value defined outside a LinearRepeatedRegions op
//                         is consumed inside it.
//   borrow-after-consume  a LinearBorrowsOperands use follows the consume.
//   unconsumed            no consuming use on some path: none at all, or a
//                         consume in only some regions of a
//                         LinearExclusiveRegions op. Exempt for mayDrop()
//                         types.
//
// Uses in distinct regions of a LinearExclusiveRegions op are exclusive and
// count as one dynamic consume. Other region-bearing ops are treated as
// scopes that run once. Uses in different blocks of a multi-block region may
// lie on exclusive CFG paths and are not compared.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "cudaq/Optimizer/Dialect/LinearOwnershipInterfaces.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

namespace qlx {

struct LinearOwnershipViolation {
  /// The scope op (a LinearOwnershipScope) the value belongs to.
  mlir::Operation *scope = nullptr;
  /// One of: double-consume, loop-capture-consume, borrow-after-consume,
  /// unconsumed.
  std::string kind;
  /// Short description of the offending value, e.g.
  /// "%0 : !qlx.logical_qubit (result of qlx.prepare)".
  std::string value;
  /// Complete diagnostic text.
  std::string message;
  /// Where the violation is reported: the offending use, or the value's
  /// definition for unconsumed values.
  mlir::Location location;
  /// A second related location (the earlier consume), when there is one.
  std::optional<mlir::Location> related;
};

struct LinearOwnershipReport {
  /// Scopes analyzed, in walk order. Declaration-only scopes are skipped.
  llvm::SmallVector<mlir::Operation *> checkedScopes;
  /// Number of linear values (block arguments and results) inspected.
  int64_t linearValues = 0;
  llvm::SmallVector<LinearOwnershipViolation> violations;

  bool ok() const { return violations.empty(); }
};

/// Analyze one LinearOwnershipScope op. Values inside nested scopes are left
/// to those scopes. Violations are appended to `report`; diagnostics are not
/// emitted. Stops after the first violation when `firstOnly` is set.
void analyzeLinearOwnershipScope(mlir::Operation *scope,
                                 LinearOwnershipReport &report,
                                 bool firstOnly = false);

/// Analyze every non-declaration-only LinearOwnershipScope op nested in
/// `root` (including `root` itself).
LinearOwnershipReport analyzeLinearOwnership(mlir::Operation *root);

/// Verifier entry point for one scope op: emits the first violation as an
/// error on the scope's context and fails.
mlir::LogicalResult verifyLinearOwnership(mlir::Operation *scope);

/// Emit one violation as an error diagnostic (with a note for `related`).
void emitLinearOwnershipViolation(const LinearOwnershipViolation &violation);

/// True for a scope whose body only declares a signature: a single block
/// holding nothing but an operand-free terminator, with unused arguments.
bool isDeclarationOnlyScope(mlir::Operation *scope);

} // namespace qlx
