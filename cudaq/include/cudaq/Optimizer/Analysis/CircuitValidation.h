/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include <cstddef>
#include <string>

namespace cudaq::opt {

/// Reasons a module can be rejected from the bounded-unitary validation domain.
///
/// The Optimization Validation Core accepts a rewrite only when the baseline
/// and candidate are straight-line, bounded-unitary Quake circuits whose
/// unitaries can be built and compared exactly. Any construct outside that
/// domain is rejected here so the `validator` fails closed rather than silently
/// validating something it cannot reason about.
enum class DomainRejectionKind {
  /// A measurement operation (`quake.mz`/`mx`/`my`, etc.) is present.
  Measurement,
  /// A `quake.reset` operation is present.
  Reset,
  /// A noise channel (`quake.apply_noise`) is present.
  Noise,
  /// Classical control flow (`cc.if`/`cc.loop`) is present. Only straight-line
  /// circuits are supported for now.
  DynamicControlFlow,
  /// An `un-inlined` call is present. The `callee's` body is not visible for
  /// exact
  /// unitary construction. Inline before validating.
  UnsupportedCall,
  /// A dynamically-sized `!quake.veq` is present. The qubit count is not
  /// statically knowable.
  DynamicQubitRegister,
  /// The kernel uses more qubits than the exact-unitary bound allows.
  TooManyQubits,
};

/// Return a stable, machine-consumable slug for \p kind (e.g. "measurement").
/// These strings are part of the `validator's` diagnostic contract and must
/// stay stable across releases.
llvm::StringRef toString(DomainRejectionKind kind);

/// A reason a kernel was rejected, with enough context to diagnose it.
struct DomainRejection {
  DomainRejectionKind kind;
  /// The kernel (function) symbol name the rejection was found in.
  std::string kernel;
  /// Context (e.g. the offending op name or qubit count).
  std::string detail;
  /// Source location of the offending construct, when available.
  mlir::Location loc;
};

/// Result of a bounded-unitary domain `preflight` over a whole module.
struct BoundedUnitaryDomainStatus {
  /// True iff every kernel with a body is in the supported domain.
  bool supported = true;
  /// The largest statically-known qubit count observed across kernels.
  std::size_t maxQubits = 0;
  /// All rejections found, in discovery order. Empty iff \c supported.
  llvm::SmallVector<DomainRejection> rejections;
};

/// Default upper bound on the number of qubits per kernel. A dense unitary of
/// n qubits is a 2^n x 2^n complex matrix, so this bounds memory/time of the
/// exact comparison.
inline constexpr unsigned kDefaultExactQubitBound = 14;

/// Determine whether every function-with-a-body in \p module is a
/// straight-line, bounded-unitary Quake circuit suitable for exact unitary
/// validation.
///
/// Declarations (empty bodies) are ignored. Each kernel is validated
/// independently. \p exactQubitBound applies per kernel. The check is a fast,
/// structural gate. The authoritative semantic check is the exact unitary
/// comparison performed separately once a module is in the supported domain.
BoundedUnitaryDomainStatus
checkBoundedUnitaryDomain(mlir::ModuleOp module,
                          unsigned exactQubitBound = kDefaultExactQubitBound);

/// Result of an exact unitary comparison of two straight-line kernels.
struct UnitaryComparisonResult {
  /// True iff both unitaries were built and have matching dimensions. When
  /// false, no comparison was performed and \c error explains why.
  bool computed = false;
  /// Element-wise equality within tolerance.
  bool strictEqual = false;
  /// Equality after dividing a global phase out of each unitary.
  bool equalUpToGlobalPhase = false;
  /// Relative global phase (radians, in (-pi, pi]) of \c candidate with respect
  /// to \c baseline. Only meaningful when \c equalUpToGlobalPhase is true.
  double phase = 0.0;
  /// True iff \c phase is within tolerance of zero.
  bool phaseIsZero = false;
  /// Populated only when \c computed is false.
  std::string error;
};

/// Compare the unitaries of two straight-line, bounded-unitary kernels exactly.
///
/// Each dense unitary is built directly from the IR (no simulator, no target
/// pipeline), then compared element-wise and up to a global phase. Current
/// CUDA-Q circuit results are not global-phase observable for a complete
/// kernel, so \c equalUpToGlobalPhase is the acceptance signal while \c phase /
/// \c phaseIsZero record the delta for callers that need it.
///
/// Callers should confirm both kernels are in the supported domain (see
/// \c checkBoundedUnitaryDomain) first. On a build failure or dimension
/// mismatch the result reports \c computed == false rather than a false
/// equivalence.
UnitaryComparisonResult compareUnitaries(mlir::func::FuncOp baseline,
                                         mlir::func::FuncOp candidate,
                                         double rtol = 1e-5,
                                         double atol = 1e-8);

} // namespace cudaq::opt
