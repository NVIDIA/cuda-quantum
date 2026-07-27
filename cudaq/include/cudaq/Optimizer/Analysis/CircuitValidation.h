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
  /// All rejections found, in discovery order. Empty iff supported.
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
  /// false, no comparison was performed and error explains why.
  bool computed = false;
  /// Element-wise equality within tolerance.
  bool strictEqual = false;
  /// Equality after dividing a global phase out of each unitary.
  bool equalUpToGlobalPhase = false;
  /// Relative global phase (radians, in (-pi, pi]) of candidate with respect
  /// to baseline. Only meaningful when equalUpToGlobalPhase is true.
  double phase = 0.0;
  /// True iff phase is within tolerance of zero.
  bool phaseIsZero = false;
  /// Populated only when computed is false.
  std::string error;
};

/// Compare the unitaries of two straight-line, bounded-unitary kernels exactly.
///
/// Each dense unitary is built directly from the IR (no simulator, no target
/// pipeline), then compared element-wise and up to a global phase. Current
/// CUDA-Q circuit results are not global-phase observable for a complete
/// kernel, so equalUpToGlobalPhase is the acceptance signal while phase
/// phaseIsZero record the delta for callers that need it.
///
/// Callers should confirm both kernels are in the supported domain (see
/// checkBoundedUnitaryDomain) first. On a build failure or dimension
/// mismatch the result reports computed == false rather than a false
/// equivalence.
UnitaryComparisonResult compareUnitaries(mlir::func::FuncOp baseline,
                                         mlir::func::FuncOp candidate,
                                         double rtol = 1e-5,
                                         double atol = 1e-8);

/// Reasons a module can be rejected from the Clifford validation domain.
///
/// The scalable (tableau) equivalence oracle reasons only about Clifford
/// circuits (H, S/S-adjoint, the Paulis X/Y/Z, single-controlled Paulis
/// (CX/CY/CZ), SWAP), and the axis rotations rx/ry/rz/r1 at integer multiples
/// of pi/2. Anything outside that class is rejected here so the tableau oracle
/// never silently downgrades a non-Clifford circuit to an unsound equivalent
/// verdict.
enum class CliffordRejectionKind {
  /// A measurement operation (`quake.mz`/`mx`/`my`, etc.) is present.
  Measurement,
  /// A `quake.reset` operation is present.
  Reset,
  /// A noise channel (`quake.apply_noise`) is present.
  Noise,
  /// Classical control flow (`cc.if`/`cc.loop`) is present.
  DynamicControlFlow,
  /// An un-inlined call is present. Inline before validating.
  UnsupportedCall,
  /// A dynamically-sized `!quake.veq` is present. The qubit count is not
  /// statically knowable.
  DynamicQubitRegister,
  /// A non-Clifford gate (`t`/`t-adjoint`, `u2`, `u3`, `phased_rx`, a custom
  /// unitary, or any other operator outside the Clifford set).
  NonCliffordGate,
  /// An axis rotation (`rx`/`ry`/`rz`/`r1`) whose angle is not a
  /// statically-known integer multiple of pi/2.
  NonCliffordRotation,
  /// A control structure outside the Clifford set. Two or more controls
  /// (Toffoli-class), or any control on a non-Pauli gate (controlled
  /// H/S/SWAP/rotation).
  NonCliffordControl,
};

/// Return a stable, consumable slug for kind (e.g. "measurement").
/// Part of the `validator's` diagnostic contract; must stay stable.
llvm::StringRef toString(CliffordRejectionKind kind);

/// A reason a kernel was rejected from the Clifford domain, with context.
struct CliffordRejection {
  CliffordRejectionKind kind;
  /// The kernel (function) symbol name the rejection was found in.
  std::string kernel;
  /// Context (e.g. the offending op name or angle).
  std::string detail;
  /// Source location of the offending construct, when available.
  mlir::Location loc;
};

/// Result of a Clifford-domain `preflight` over a whole module.
struct CliffordDomainStatus {
  /// True iff every kernel with a body is a Clifford circuit.
  bool supported = true;
  /// The largest statically-known qubit count observed across kernels.
  std::size_t maxQubits = 0;
  /// All rejections found, in discovery order. Empty iff supported.
  llvm::SmallVector<CliffordRejection> rejections;
};

/// Determine whether every function-with-a-body in module is a Clifford
/// circuit suitable for exact tableau (stabilizer) equivalence checking.
CliffordDomainStatus checkCliffordDomain(mlir::ModuleOp module);

/// Result of an exact stabilizer-tableau comparison of two Clifford kernels.
struct CliffordComparisonResult {
  /// True iff both tableaux were built and cover the same qubit count. When
  /// false, no comparison was performed and error explains why.
  bool computed = false;
  /// True iff the two Clifford operations are equal. A stabilizer tableau does
  /// not represent global phase, so this is inherently an up-to-global-phase
  /// verdict (the same acceptance signal as the dense-unitary oracle).
  bool equivalent = false;
  /// Populated only when computed is false (a non-Clifford op slipped past the
  /// domain preflight, or the kernels differ in qubit count).
  std::string error;
};

/// Compare two straight-line Clifford kernels by their stabilizer tableaux.
///
/// Each kernel is compiled (no simulator, no target) into a stabilizer tableau
/// and the tableaux are compared for equality. Unlike compareUnitaries there is
/// no qubit bound: the tableau is polynomial in the qubit count. Because a
/// tableau does not track global phase, equality is inherently up to a global
/// phase.
CliffordComparisonResult compareTableaux(mlir::func::FuncOp baseline,
                                         mlir::func::FuncOp candidate);

} // namespace cudaq::opt
