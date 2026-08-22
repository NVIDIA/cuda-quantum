/****************************************************************-*- C++ -*-****
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
  /// Control flow is present, either structured (`cc.if`/`cc.loop`, an unwind
  /// out of one) or as a CFG branch between blocks (`cf.br`/`cf.cond_br`, which
  /// is what an early return lowers to). Only straight-line circuits are
  /// supported for now.
  DynamicControlFlow,
  /// An `un-inlined` call is present. The `callee's` body is not visible for
  /// exact unitary construction. Inline before validating.
  UnsupportedCall,
  /// A dynamically-sized `!quake.veq` is present. The qubit count is not
  /// statically knowable.
  DynamicQubitRegister,
  /// The kernel uses more qubits than the exact-unitary bound allows.
  TooManyQubits,
  /// The kernel's `ancillas` are not returned to the computational basis state
  /// they came in as, so the operator cannot be reduced to one on the system
  /// qubits alone. Unlike the kinds above this is not a `preflight` verdict:
  /// it is only visible once the unitary has been built, and it is reported
  /// through UnitaryComparisonResult.
  AncillaNotRestored,
};

/// Return a stable, machine-consumable slug for \p kind (e.g. "measurement").
/// These strings are part of the `validator's` diagnostic contract and must
/// stay stable across releases.
llvm::StringRef toString(DomainRejectionKind kind);

/// A reason a kernel was rejected from a validation domain, with enough
/// context to diagnose it. RejectionKindTy names the domain's rejection
/// enum. See DomainRejection and CliffordRejection below.
template <typename RejectionKindTy>
struct Rejection {
  RejectionKindTy kind;
  /// The kernel (function) symbol name the rejection was found in.
  std::string kernel;
  /// Context (e.g. the offending op name, angle, or qubit count).
  std::string detail;
  /// Source location of the offending construct, when available.
  mlir::Location loc;
};

/// A rejection from the bounded-unitary domain.
using DomainRejection = Rejection<DomainRejectionKind>;

/// Result of a bounded-unitary domain `preflight` over a whole module.
struct BoundedUnitaryDomainStatus {
  /// True iff every kernel with a body is in the supported domain.
  bool supported = true;
  /// The largest statically-known qubit count observed across kernels.
  std::size_t maxQubits = 0;
  /// Of those, the largest number contributed by allocations marked with
  /// `quake.ancilla`. Ancillas count against the same flat bound as any other
  /// qubit (a 2^n matrix does not care what a qubit is for).
  std::size_t maxAncillaQubits = 0;
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

/// What an equivalence verdict is a statement about.
enum class EquivalenceGuarantee {
  /// Neither kernel used `ancillas`. The verdict covers the whole operator.
  Exact,
  /// At least one kernel introduced `ancillas`, which were checked to be
  /// returned to the basis state they came in as and then projected out. The
  /// verdict is that the two kernels agree on the system qubits when the
  /// `ancillas` start in |0>. It says nothing about `ancillas` that arrive in
  /// some other state (the borrowed-ancilla claim), which is a stronger
  /// property this oracle does not check.
  CleanAncilla,
  /// The wider kernel was shown to be the narrower one `tensored` with the
  /// identity, so it never touches the extra qubits. Stronger than
  /// CleanAncilla: it holds whatever state the `ancillas` arrive in.
  BorrowedAncilla,
};

/// Return a stable slug for guarantee ("exact", "clean-ancilla",
/// "borrowed-ancilla").
llvm::StringRef toString(EquivalenceGuarantee guarantee);

/// Default tolerances for the element-wise unitary comparison, matching the
/// defaults of cudaq::isApproxEqual.
inline constexpr double kDefaultRelativeTolerance = 1e-5;
inline constexpr double kDefaultAbsoluteTolerance = 1e-8;

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
  /// What the verdict is a statement about. See EquivalenceGuarantee.
  EquivalenceGuarantee guarantee = EquivalenceGuarantee::Exact;
  /// True iff the candidate left its `ancillas` dirty. This is a negative
  /// verdict, not a failure to compare. Computed stays true and the kernels
  /// are reported as not equivalent. A baseline with dirty `ancillas` is a
  /// different matter (there is nothing to compare against) and reports
  /// computed == false instead.
  bool ancillaNotRestored = false;
  /// Qubits projected out as `ancillas` on each side.
  std::size_t baselineAncillas = 0;
  std::size_t candidateAncillas = 0;
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
UnitaryComparisonResult
compareUnitaries(mlir::func::FuncOp baseline, mlir::func::FuncOp candidate,
                 double rtol = kDefaultRelativeTolerance,
                 double atol = kDefaultAbsoluteTolerance);

/// Reasons a module can be rejected from the Clifford validation domain.
///
/// The scalable (tableau) equivalence oracle reasons only about Clifford
/// circuits (H, S/S-adjoint, the Paulis X/Y/Z, single-controlled Paulis
/// (CX/CY/CZ), SWAP), and the axis rotations `rx`/`ry`/`rz`/`r1` at integer
/// multiples of pi/2. Anything outside that class is rejected here so the
/// tableau oracle never silently downgrades a non-Clifford circuit to an
/// unsound equivalent verdict.
enum class CliffordRejectionKind {
  /// A measurement operation (`quake.mz`/`mx`/`my`, etc.) is present.
  Measurement,
  /// A `quake.reset` operation is present.
  Reset,
  /// A noise channel (`quake.apply_noise`) is present.
  Noise,
  /// Control flow is present, either structured (`cc.if`/`cc.loop`, an unwind
  /// out of one) or as a CFG branch between blocks (`cf.br`/`cf.cond_br`).
  DynamicControlFlow,
  /// An `un-inlined` call is present. Inline before validating.
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

/// A rejection from the Clifford domain.
using CliffordRejection = Rejection<CliffordRejectionKind>;

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
  /// What the verdict is a statement about. A width difference is checked as
  /// a tensor with the identity, which earns BorrowedAncilla rather than the
  /// dense oracle's weaker CleanAncilla.
  EquivalenceGuarantee guarantee = EquivalenceGuarantee::Exact;
  /// Populated only when computed is false (a non-Clifford op slipped past the
  /// domain `preflight`).
  std::string error;
};

/// Compare two straight-line Clifford kernels by their stabilizer tableaux.
///
/// Each kernel is compiled (no simulator, no target) into a stabilizer tableau
/// and the tableaux are compared for equality. Unlike compareUnitaries there is
/// no qubit bound: the tableau is polynomial in the qubit count. Because a
/// tableau does not track global phase, equality is inherently up to a global
/// phase.
///
/// A kernel that took on `ancillas` is first checked against the padded
/// tableau, which certifies it when the extra qubits are untouched
/// (`borrowed-ancilla`). Failing that, it is checked on the subspace where the
/// `ancillas` hold |0>, which certifies the weaker `clean-ancilla` claim. The
/// guarantee on the result says which one the verdict is about.
CliffordComparisonResult compareTableaux(mlir::func::FuncOp baseline,
                                         mlir::func::FuncOp candidate);

} // namespace cudaq::opt
