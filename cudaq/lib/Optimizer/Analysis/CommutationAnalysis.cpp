/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/CommutationAnalysis.h"
#include "QubitIdentityAnalysis.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Matchers.h"
#include <functional>
#include <optional>
#include <utility>

using namespace mlir;

using Pauli = cudaq::quake::Pauli;
using PauliWord = cudaq::quake::PauliWord;
using CommutationAnalysis = cudaq::quake::detail::CommutationAnalysis;
using commutation_reason = cudaq::quake::detail::commutation_reason;
using CommutationResult = cudaq::quake::detail::CommutationResult;
using commutation_status = cudaq::quake::detail::commutation_status;
using QubitIdentityAnalysis = cudaq::quake::detail::QubitIdentityAnalysis;

namespace {
using QubitId = QubitIdentityAnalysis::QubitId;
using OperationPair = std::pair<Operation *, Operation *>;

struct ControlUse {
  QubitId qubitId;
  bool negated;

  bool operator==(const ControlUse &) const = default;
};

enum class qubit_role { target, positive_control, negative_control };

// Query-local view of a supported Quake operation's controls, targets, and
// support, expressed using analysis-local qubit identifiers.
struct OperationView {
  explicit OperationView(Operation *operation)
      : operation(operation),
        operatorInterface(
            dyn_cast<cudaq::quake::OperatorInterface>(operation)) {}

  // Underlying operation used for kind-specific commutation rules.
  Operation *operation;
  // Quake interface used by unitary-channel rules. Measurement-instrument and
  // other non-unitary operation views leave it null.
  cudaq::quake::OperatorInterface operatorInterface;
  // Controls in operand order, including their positive or negative polarity.
  llvm::SmallVector<ControlUse> controls;
  // Targets in operand order, preserving positional gate semantics.
  llvm::SmallVector<QubitId> targets;
  // Unique role and control polarity for each supported qubit.
  llvm::DenseMap<QubitId, qubit_role> roles;
};

// Analysis-local Pauli product keyed by qubit rather than IR target order.
// This normalized form makes shared-qubit parity checks order-independent.
struct PauliAction {
  llvm::DenseMap<QubitId, Pauli> terms;
};
} // namespace

static CommutationResult commutes(commutation_reason reason) {
  return {commutation_status::commutes, reason};
}

static CommutationResult doesNotCommute(commutation_reason reason) {
  return {commutation_status::does_not_commute, reason};
}

static CommutationResult indeterminate(commutation_reason reason) {
  return {commutation_status::indeterminate, reason};
}

static OperationPair getCanonicalCacheKey(Operation *lhs, Operation *rhs) {
  // std::less provides a total order for unrelated pointers. The order has no
  // semantic meaning; it only makes the symmetric cache key canonical.
  if (std::less<Operation *>{}(rhs, lhs))
    return {rhs, lhs};
  return {lhs, rhs};
}

static bool isSupportedViewOperation(Operation *operation) {
  return isa<cudaq::quake::OperatorInterface,
             cudaq::quake::MeasurementInterface, cudaq::quake::ResetOp,
             cudaq::quake::SinkOp>(operation);
}

// Identify built-in operations for which shared-support rules are implemented.
static bool isSupportedSharedOperation(Operation *operation) {
  return isa<cudaq::quake::HOp, cudaq::quake::XOp, cudaq::quake::YOp,
             cudaq::quake::ZOp, cudaq::quake::SOp, cudaq::quake::TOp,
             cudaq::quake::SwapOp, cudaq::quake::R1Op, cudaq::quake::RxOp,
             cudaq::quake::RyOp, cudaq::quake::RzOp, cudaq::quake::PhasedRxOp,
             cudaq::quake::U2Op, cudaq::quake::U3Op, cudaq::quake::ExpPauliOp>(
      operation);
}

// Exact Pauli operators support a negative commutation proof from odd parity.
// ExpPauli rotations do not because their angles may make them commute.
static bool isPauliOperator(Operation *operation) {
  return isa<cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(
      operation);
}

static bool isXAxis(Operation *operation) {
  return isa<cudaq::quake::XOp, cudaq::quake::RxOp>(operation);
}

static bool isYAxis(Operation *operation) {
  return isa<cudaq::quake::YOp, cudaq::quake::RyOp>(operation);
}

static bool isZAxis(Operation *operation) {
  return isa<cudaq::quake::ZOp, cudaq::quake::SOp, cudaq::quake::TOp,
             cudaq::quake::R1Op, cudaq::quake::RzOp>(operation);
}

// Identify operations covered by the computational-diagonal rule.
static bool isComputationalDiagonal(Operation *operation) {
  // The initial rule set recognizes the shared single-target Z-axis family.
  return isZAxis(operation);
}

// Require parameter identity or equal constant attributes; do not approximate.
static bool areExactParameterValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (lhs.getType() != rhs.getType())
    return false;

  Attribute lhsConstant;
  Attribute rhsConstant;
  return matchPattern(lhs, m_Constant(&lhsConstant)) &&
         matchPattern(rhs, m_Constant(&rhsConstant)) &&
         lhsConstant == rhsConstant;
}

// Compare all OperatorInterface parameters using exact structural equality.
static bool haveExactParameters(cudaq::quake::OperatorInterface lhs,
                                cudaq::quake::OperatorInterface rhs) {
  auto lhsParameters = lhs.getParameters();
  auto rhsParameters = rhs.getParameters();
  return lhsParameters.size() == rhsParameters.size() &&
         llvm::equal(lhsParameters, rhsParameters, areExactParameterValues);
}

// Compare ordered targets, except for Swap's unordered target pair.
static bool haveSameTargets(const OperationView &lhs,
                            const OperationView &rhs) {
  if (lhs.targets.size() != rhs.targets.size())
    return false;
  // Swap is symmetric in its two targets, so reversed target order represents
  // the same operation.
  if (isa<cudaq::quake::SwapOp>(lhs.operation) &&
      isa<cudaq::quake::SwapOp>(rhs.operation))
    return lhs.targets.size() == 2 && ((lhs.targets[0] == rhs.targets[0] &&
                                        lhs.targets[1] == rhs.targets[1]) ||
                                       (lhs.targets[0] == rhs.targets[1] &&
                                        lhs.targets[1] == rhs.targets[0]));
  return lhs.targets == rhs.targets;
}

// Decode the literal Pauli word needed by structural ExpPauli rules.
static std::optional<PauliWord> getLiteralPaulis(const OperationView &view) {
  auto expPauli = dyn_cast<cudaq::quake::ExpPauliOp>(view.operation);
  if (!expPauli)
    return std::nullopt;
  auto literal = expPauli.getPauliLiteralAttr();
  if (!literal)
    return std::nullopt;
  return cudaq::quake::symbolizePauliWord(literal.getValue());
}

// Reject dynamic ExpPauli words before rules that require literal symbols.
static bool hasSupportedPauliWord(const OperationView &view) {
  return !isa<cudaq::quake::ExpPauliOp>(view.operation) ||
         getLiteralPaulis(view).has_value();
}

// Compare the generator or matrix symbol that defines a custom unitary.
static bool haveSameCustomUnitaryDefinition(Operation *lhs, Operation *rhs) {
  if (auto lhsCall = dyn_cast<cudaq::quake::CustomUnitaryCallOp>(lhs)) {
    auto rhsCall = dyn_cast<cudaq::quake::CustomUnitaryCallOp>(rhs);
    return rhsCall && lhsCall.getGeneratorAttr() == rhsCall.getGeneratorAttr();
  }
  if (auto lhsConstant = dyn_cast<cudaq::quake::CustomUnitaryConstantOp>(lhs)) {
    auto rhsConstant = dyn_cast<cudaq::quake::CustomUnitaryConstantOp>(rhs);
    return rhsConstant &&
           lhsConstant.getMatrixAttr() == rhsConstant.getMatrixAttr();
  }
  return false;
}

// Recognize matching structural operations or exact adjoints on the same
// qubits.
static bool haveSameOperation(const OperationView &lhs,
                              const OperationView &rhs) {
  if (!lhs.operatorInterface || !rhs.operatorInterface)
    return false;

  // Match the operation kind and every action-bearing interface value. Adjoint
  // state may differ because an operation commutes with its exact inverse.
  bool sameRecognizedKind =
      (isSupportedSharedOperation(lhs.operation) &&
       isSupportedSharedOperation(rhs.operation) &&
       lhs.operation->getName() == rhs.operation->getName()) ||
      haveSameCustomUnitaryDefinition(lhs.operation, rhs.operation);
  if (!sameRecognizedKind || lhs.controls != rhs.controls ||
      !haveSameTargets(lhs, rhs) ||
      !haveExactParameters(lhs.operatorInterface, rhs.operatorInterface))
    return false;

  // ExpPauli stores part of its action in the Pauli word rather than among the
  // OperatorInterface parameters.
  if (isa<cudaq::quake::ExpPauliOp>(lhs.operation)) {
    auto lhsPaulis = getLiteralPaulis(lhs);
    auto rhsPaulis = getLiteralPaulis(rhs);
    return lhsPaulis && rhsPaulis && lhsPaulis == rhsPaulis;
  }
  return true;
}

// Recognize equal single-target rotation axes for the same-axis rule.
static bool haveSameAxisTargetAction(const OperationView &lhs,
                                     const OperationView &rhs) {
  if (lhs.targets.size() != 1 || rhs.targets.size() != 1 ||
      lhs.targets.front() != rhs.targets.front())
    return false;
  // Gates in the same standard axis family commute even when their rotation
  // angles differ.
  if ((isXAxis(lhs.operation) && isXAxis(rhs.operation)) ||
      (isYAxis(lhs.operation) && isYAxis(rhs.operation)) ||
      (isZAxis(lhs.operation) && isZAxis(rhs.operation)))
    return true;

  // This rule proves commutation when the axis-defining PhasedRx phase values
  // match exactly; rotation angles may differ.
  auto lhsPhasedRx = dyn_cast<cudaq::quake::PhasedRxOp>(lhs.operation);
  auto rhsPhasedRx = dyn_cast<cudaq::quake::PhasedRxOp>(rhs.operation);
  return lhsPhasedRx && rhsPhasedRx &&
         areExactParameterValues(lhsPhasedRx.getParameters()[1],
                                 rhsPhasedRx.getParameters()[1]);
}

// Normalize an exact Pauli operator or literal ExpPauli word into Pauli symbols
// keyed by the block-local qubits on which they act.
static std::optional<PauliAction> getPauliAction(const OperationView &view) {
  std::optional<Pauli> pauli;
  if (isa<cudaq::quake::XOp>(view.operation))
    pauli = Pauli::X;
  else if (isa<cudaq::quake::YOp>(view.operation))
    pauli = Pauli::Y;
  else if (isa<cudaq::quake::ZOp>(view.operation))
    pauli = Pauli::Z;

  if (pauli) {
    if (view.targets.size() != 1)
      return std::nullopt;
    PauliAction action;
    action.terms.try_emplace(view.targets.front(), *pauli);
    return action;
  }

  auto paulis = getLiteralPaulis(view);
  if (!paulis)
    return std::nullopt;
  PauliAction action;
  action.terms.reserve(view.targets.size());
  for (auto [qubitId, symbol] : llvm::zip(view.targets, *paulis))
    action.terms.try_emplace(qubitId, symbol);
  return action;
}

// Compute whether shared Pauli factors contain an odd number of anti-commuting
// nonidentity pairs.
static bool hasOddPauliAnticommutationParity(const PauliAction &lhs,
                                             const PauliAction &rhs) {
  const auto *smaller = &lhs.terms;
  const auto *larger = &rhs.terms;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);

  bool hasOddParity = false;
  for (auto [qubitId, pauli] : *smaller) {
    auto other = larger->find(qubitId);
    if (other != larger->end() && pauli != Pauli::I &&
        other->second != Pauli::I && pauli != other->second)
      hasOddParity = !hasOddParity;
  }
  return hasOddParity;
}

// Supported operations with no shared control or target qubit commute because
// their induced maps act on disjoint quantum factors.
static bool haveDisjointQuantumSupport(const OperationView &lhs,
                                       const OperationView &rhs) {
  const auto *smaller = &lhs.roles;
  const auto *larger = &rhs.roles;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  return llvm::none_of(*smaller, [&](const auto &entry) {
    return larger->contains(entry.first);
  });
}

// Detect a qubit used as a target by one operation and a control by the other.
static bool hasTargetControlCrossover(const OperationView &lhs,
                                      const OperationView &rhs) {
  auto isControl = [](const OperationView &view, QubitId qubitId) {
    auto role = view.roles.find(qubitId);
    return role != view.roles.end() && role->second != qubit_role::target;
  };
  return llvm::any_of(
             lhs.targets,
             [&](QubitId qubitId) { return isControl(rhs, qubitId); }) ||
         llvm::any_of(rhs.targets,
                      [&](QubitId qubitId) { return isControl(lhs, qubitId); });
}

// Check whether all shared support of a computational-basis-diagonal operation
// occurs only among the other operation's controls.
static bool diagonalOverlapsOnlyControls(const OperationView &diagonal,
                                         const OperationView &controlled) {
  if (!isComputationalDiagonal(diagonal.operation) ||
      controlled.controls.empty())
    return false;

  bool hasOverlap = false;
  auto checkQubit = [&](QubitId qubitId) {
    auto role = controlled.roles.find(qubitId);
    if (role == controlled.roles.end())
      return true;
    hasOverlap = true;
    return role->second != qubit_role::target;
  };

  for (ControlUse control : diagonal.controls)
    if (!checkQubit(control.qubitId))
      return false;
  for (QubitId target : diagonal.targets)
    if (!checkQubit(target))
      return false;
  return hasOverlap;
}

// Controlled operations may share controls; this predicate checks only whether
// their target actions are disjoint.
static bool haveDisjointTargetSupport(const OperationView &lhs,
                                      const OperationView &rhs) {
  const auto *smaller = &lhs;
  const auto *larger = &rhs;
  if (larger->targets.size() < smaller->targets.size())
    std::swap(smaller, larger);
  return llvm::none_of(smaller->targets, [&](QubitId qubitId) {
    auto role = larger->roles.find(qubitId);
    return role != larger->roles.end() && role->second == qubit_role::target;
  });
}

// Test target-only commutation after controlled-rule preconditions are met.
static bool targetActionsCommute(const OperationView &lhs,
                                 const OperationView &rhs) {
  if (haveDisjointTargetSupport(lhs, rhs))
    return true;
  if (lhs.operation->getName() == rhs.operation->getName() &&
      haveSameTargets(lhs, rhs) &&
      haveExactParameters(lhs.operatorInterface, rhs.operatorInterface)) {
    if (!isa<cudaq::quake::ExpPauliOp>(lhs.operation) ||
        getLiteralPaulis(lhs) == getLiteralPaulis(rhs))
      return true;
  }
  if (isComputationalDiagonal(lhs.operation) &&
      isComputationalDiagonal(rhs.operation))
    return true;
  if (haveSameAxisTargetAction(lhs, rhs))
    return true;

  auto lhsPauli = getPauliAction(lhs);
  auto rhsPauli = getPauliAction(rhs);
  return lhsPauli && rhsPauli &&
         !hasOddPauliAnticommutationParity(*lhsPauli, *rhsPauli);
}

// Opposite polarity on any shared control gives orthogonal projectors
// |0><0| |1><1| = 0, so the control predicates cannot both be satisfied.
static bool haveMutuallyExclusiveControls(const OperationView &lhs,
                                          const OperationView &rhs) {
  const auto *smaller = &lhs;
  const auto *larger = &rhs;
  if (larger->controls.size() < smaller->controls.size())
    std::swap(smaller, larger);
  for (ControlUse control : smaller->controls) {
    auto other = larger->roles.find(control.qubitId);
    if (other != larger->roles.end() && other->second != qubit_role::target &&
        control.negated != (other->second == qubit_role::negative_control))
      return true;
  }
  return false;
}

// Maps on disjoint quantum factors compose independently in either order.
static std::optional<CommutationResult>
tryDisjointSupport(const OperationView &lhs, const OperationView &rhs) {
  if (haveDisjointQuantumSupport(lhs, rhs))
    return commutes(commutation_reason::disjoint_support);
  return std::nullopt;
}

// U commutes with itself and its exact adjoint because UU^-1 = U^-1U = I.
static std::optional<CommutationResult>
trySameOperation(const OperationView &lhs, const OperationView &rhs) {
  if (haveSameOperation(lhs, rhs))
    return commutes(commutation_reason::same_operation);
  return std::nullopt;
}

static std::optional<CommutationResult>
tryInstrumentOrReset(const OperationView &lhs, const OperationView &rhs) {
  const bool lhsIsOperator = static_cast<bool>(lhs.operatorInterface);
  const bool rhsIsOperator = static_cast<bool>(rhs.operatorInterface);
  if (lhsIsOperator == rhsIsOperator)
    return std::nullopt;

  const OperationView &unitaryChannel = lhsIsOperator ? lhs : rhs;
  const OperationView &nonUnitaryOperation = lhsIsOperator ? rhs : lhs;
  if (!isa<cudaq::quake::MeasurementInterface, cudaq::quake::ResetOp>(
          nonUnitaryOperation.operation))
    return std::nullopt;
  if (!unitaryChannel.controls.empty())
    return std::nullopt;
  if (nonUnitaryOperation.targets.size() != 1 ||
      unitaryChannel.targets.size() != 1)
    return std::nullopt;
  if (nonUnitaryOperation.targets.front() != unitaryChannel.targets.front())
    return std::nullopt;

  // Mz returns a computational-basis wire. A Z-axis unitary preserves both
  // its outcome projectors and each conditional output state. Mx and My also
  // return computational-basis wires, so matching their observed axis alone
  // does not prove instrument commutation.
  if (isa<cudaq::quake::MzOp>(nonUnitaryOperation.operation) &&
      isZAxis(unitaryChannel.operation))
    return commutes(commutation_reason::measurement_instrument_basis);

  // Reset is R(rho) = |0><0| Tr(rho). A Z-axis unitary preserves |0> up to
  // phase, so R(U rho U^dagger) = U R(rho) U^dagger = R(rho).
  if (isa<cudaq::quake::ResetOp>(nonUnitaryOperation.operation) &&
      isZAxis(unitaryChannel.operation))
    return commutes(commutation_reason::preserved_reset_state);
  return std::nullopt;
}

// Computational-basis diagonal matrices satisfy D1 D2 = D2 D1 because their
// products are pointwise scalar products in the same basis.
static std::optional<CommutationResult>
tryComputationalDiagonal(const OperationView &lhs, const OperationView &rhs) {
  if (isComputationalDiagonal(lhs.operation) &&
      isComputationalDiagonal(rhs.operation))
    return commutes(commutation_reason::computational_diagonal);
  return std::nullopt;
}

// Operators that are functions of the same Pauli axis P commute because
// f(P) g(P) = g(P) f(P). This rule recognizes PhasedRx axes only when their
// phase values match exactly.
static std::optional<CommutationResult> trySameAxis(const OperationView &lhs,
                                                    const OperationView &rhs) {
  if (lhs.controls.empty() && rhs.controls.empty() &&
      haveSameAxisTargetAction(lhs, rhs))
    return commutes(commutation_reason::same_axis);
  return std::nullopt;
}

// Pauli products obey PQ = (-1)^m QP, where m is the number of aligned
// anti-commuting factors. Odd parity proves a negative only for exact Pauli
// operators, not parameterized ExpPauli rotations.
static std::optional<CommutationResult>
tryPauliParity(const OperationView &lhs, const OperationView &rhs) {
  if (!lhs.controls.empty() || !rhs.controls.empty())
    return std::nullopt;
  auto lhsPauli = getPauliAction(lhs);
  auto rhsPauli = getPauliAction(rhs);
  if (!lhsPauli || !rhsPauli)
    return std::nullopt;
  if (!hasOddPauliAnticommutationParity(*lhsPauli, *rhsPauli))
    return commutes(commutation_reason::even_pauli_parity);
  if (isPauliOperator(lhs.operation) && isPauliOperator(rhs.operation))
    return doesNotCommute(commutation_reason::odd_pauli_parity);
  return std::nullopt;
}

// Quake control polarity selects a computational-basis projector P. A diagonal
// action D on that control satisfies DP = PD for either polarity, so the proof
// holds for every input state. This applies when every shared qubit is only a
// control of the other operation, never one of its targets.
// This rule cannot recognize a control basis established by surrounding basis
// changes, such as H-C(U)-H. Sequence-level basis tracking would cover those
// cases, while reusable per-operand commuting-basis properties would avoid
// hard-coding the supported individual operations.
static std::optional<CommutationResult>
tryDiagonalOnControls(const OperationView &lhs, const OperationView &rhs) {
  if (diagonalOverlapsOnlyControls(lhs, rhs) ||
      diagonalOverlapsOnlyControls(rhs, lhs))
    return commutes(commutation_reason::diagonal_on_controls);
  return std::nullopt;
}

// With no target-control crossover, commuting target actions and commuting
// control projectors make every term of the controlled products commute.
static std::optional<CommutationResult>
tryCompatibleControlledTargets(const OperationView &lhs,
                               const OperationView &rhs) {
  if ((!lhs.controls.empty() || !rhs.controls.empty()) &&
      !hasTargetControlCrossover(lhs, rhs) && targetActionsCommute(lhs, rhs))
    return commutes(commutation_reason::compatible_controlled_targets);
  return std::nullopt;
}

// Opposite polarity on a shared control gives disjoint projectors (PQ = 0), so
// the controlled operations commute regardless of their target actions.
static std::optional<CommutationResult>
tryMutuallyExclusiveControls(const OperationView &lhs,
                             const OperationView &rhs) {
  if (!lhs.controls.empty() && !rhs.controls.empty() &&
      !hasTargetControlCrossover(lhs, rhs) &&
      haveMutuallyExclusiveControls(lhs, rhs))
    return commutes(commutation_reason::mutually_exclusive_controls);
  return std::nullopt;
}

using CommutationRule = std::optional<CommutationResult> (*)(
    const OperationView &, const OperationView &);

// Apply general rules, reject unsupported shared-support cases, then apply the
// remaining shared-support rules in stable proof-reason precedence order.
static CommutationResult dispatchRules(const OperationView &lhs,
                                       const OperationView &rhs) {
  if ((isa<cudaq::quake::MeasurementInterface>(lhs.operation) &&
       lhs.targets.size() != 1) ||
      (isa<cudaq::quake::MeasurementInterface>(rhs.operation) &&
       rhs.targets.size() != 1))
    return indeterminate(commutation_reason::no_applicable_rule);
  if (auto result = tryDisjointSupport(lhs, rhs))
    return *result;
  if (!lhs.operatorInterface || !rhs.operatorInterface) {
    if (auto result = tryInstrumentOrReset(lhs, rhs))
      return *result;
    return indeterminate(commutation_reason::no_applicable_rule);
  }
  if (auto result = trySameOperation(lhs, rhs))
    return *result;
  if (!isSupportedSharedOperation(lhs.operation) ||
      !isSupportedSharedOperation(rhs.operation))
    return indeterminate(commutation_reason::no_applicable_rule);
  if (!hasSupportedPauliWord(lhs) || !hasSupportedPauliWord(rhs))
    return indeterminate(commutation_reason::unsupported_pauli_word);

  // Rule order determines which successful proof reason is reported.
  static constexpr CommutationRule orderedRules[] = {
      tryComputationalDiagonal,
      trySameAxis,
      tryPauliParity,
      tryDiagonalOnControls,
      tryCompatibleControlledTargets,
      tryMutuallyExclusiveControls,
  };
  for (CommutationRule rule : orderedRules)
    if (auto result = rule(lhs, rhs))
      return *result;
  return indeterminate(commutation_reason::no_applicable_rule);
}

// Populate the normalized view used by commutation rules. Resolve scalar wire
// controls and targets to analysis-local qubit IDs, record their roles and
// control polarities, and reject unmapped or duplicate qubit uses.
static std::optional<commutation_reason>
populateOperationView(OperationView &view,
                      const QubitIdentityAnalysis &qubitIdentity) {
  // A supported operation view may use a qubit in only one control or target
  // role, so the role index also rejects duplicates across both groups.
  llvm::SmallVector<Value> targets;

  if (auto quantumOperator = view.operatorInterface) {
    // Operators contribute ordered controls, their polarities, and ordered
    // targets because later rules distinguish each qubit's role.
    // Valid Quake IR guarantees that polarity metadata, when present, has one
    // entry per control operand.
    auto negatedControls = quantumOperator.getNegatedControls();
    auto controls = quantumOperator.getControls();

    // Preserve control operand order while also building the role index
    // required by controlled-operation rules.
    view.controls.reserve(controls.size());
    for (auto [index, control] : llvm::enumerate(controls)) {
      if (!isa<cudaq::quake::WireType>(control.getType()))
        return commutation_reason::unsupported_quantum_operand_type;
      auto qubitId = qubitIdentity.getQubitId(control);
      if (!qubitId)
        return commutation_reason::unmapped_qubit_id;
      bool negated = negatedControls && (*negatedControls)[index];
      if (!view.roles
               .try_emplace(*qubitId, negated ? qubit_role::negative_control
                                              : qubit_role::positive_control)
               .second)
        return commutation_reason::duplicate_qubit_operand;
      view.controls.push_back({*qubitId, negated});
    }
    llvm::append_range(targets, quantumOperator.getTargets());
  } else if (auto measurementInstrument =
                 dyn_cast<cudaq::quake::MeasurementInterface>(view.operation)) {
    // Only the measured quantum targets contribute to qubit support. Classical
    // outcomes are instrument results, not qubit identities.
    llvm::append_range(targets, measurementInstrument.getTargets());
  } else if (auto resetChannel =
                 dyn_cast<cudaq::quake::ResetOp>(view.operation)) {
    // Reset is a single-target channel with no control role.
    targets.push_back(resetChannel.getTargets());
  } else if (auto sink = dyn_cast<cudaq::quake::SinkOp>(view.operation)) {
    // Sink consumes the target identity. Recording that target makes shared
    // support a conservative boundary.
    targets.push_back(sink.getTarget());
  } else {
    llvm_unreachable("operation kind was validated before normalization");
  }

  // Preserve target order for positional gate semantics and build the target
  // membership lookup used by overlap and crossover rules.
  view.targets.reserve(targets.size());
  for (Value target : targets) {
    if (!isa<cudaq::quake::WireType>(target.getType()))
      return commutation_reason::unsupported_quantum_operand_type;
    auto qubitId = qubitIdentity.getQubitId(target);
    if (!qubitId)
      return commutation_reason::unmapped_qubit_id;
    if (!view.roles.try_emplace(*qubitId, qubit_role::target).second)
      return commutation_reason::duplicate_qubit_operand;
    view.targets.push_back(*qubitId);
  }
  return std::nullopt;
}

// Validate and normalize a query, then apply general and shared-support rules.
static CommutationResult evaluate(Operation *lhs, Operation *rhs,
                                  const QubitIdentityAnalysis &qubitIdentity) {
  if (!isSupportedViewOperation(lhs) || !isSupportedViewOperation(rhs))
    return indeterminate(commutation_reason::unsupported_operation_kind);

  OperationView lhsView{lhs};
  OperationView rhsView{rhs};
  auto lhsFailure = populateOperationView(lhsView, qubitIdentity);
  auto rhsFailure = populateOperationView(rhsView, qubitIdentity);
  if (lhsFailure || rhsFailure) {
    // Operand representation is a prerequisite for identity resolution, which
    // is a prerequisite for checking whether one identity has multiple roles.
    // This semantic order keeps the reason independent of query operand order.
    static constexpr commutation_reason normalizationFailurePrecedence[] = {
        commutation_reason::unsupported_quantum_operand_type,
        commutation_reason::unmapped_qubit_id,
        commutation_reason::duplicate_qubit_operand,
    };
    for (commutation_reason reason : normalizationFailurePrecedence)
      if (lhsFailure == reason || rhsFailure == reason)
        return indeterminate(reason);
    llvm_unreachable("unhandled operation-view normalization failure");
  }

  return dispatchRules(lhsView, rhsView);
}

llvm::StringRef
cudaq::quake::detail::getCommutationReasonId(commutation_reason reason) {
  switch (reason) {
  case commutation_reason::disjoint_support:
    return "disjoint-support";
  case commutation_reason::same_operation:
    return "same-operation";
  case commutation_reason::computational_diagonal:
    return "computational-diagonal";
  case commutation_reason::same_axis:
    return "same-axis";
  case commutation_reason::measurement_instrument_basis:
    return "measurement-instrument-basis";
  case commutation_reason::preserved_reset_state:
    return "preserved-reset-state";
  case commutation_reason::even_pauli_parity:
    return "even-pauli-parity";
  case commutation_reason::odd_pauli_parity:
    return "odd-pauli-parity";
  case commutation_reason::diagonal_on_controls:
    return "diagonal-on-controls";
  case commutation_reason::compatible_controlled_targets:
    return "compatible-controlled-targets";
  case commutation_reason::mutually_exclusive_controls:
    return "mutually-exclusive-controls";
  case commutation_reason::null_operation:
    return "null-operation";
  case commutation_reason::different_blocks:
    return "different-blocks";
  case commutation_reason::unsupported_operation_kind:
    return "unsupported-operation-kind";
  case commutation_reason::unsupported_quantum_operand_type:
    return "unsupported-quantum-operand-type";
  case commutation_reason::unmapped_qubit_id:
    return "unmapped-qubit-id";
  case commutation_reason::duplicate_qubit_operand:
    return "duplicate-qubit-operand";
  case commutation_reason::unsupported_pauli_word:
    return "unsupported-pauli-word";
  case commutation_reason::no_applicable_rule:
    return "no-applicable-rule";
  }
  llvm_unreachable("unhandled commutation reason");
}

CommutationAnalysis::CommutationAnalysis(Block &block)
    : block(&block),
      qubitIdentity(std::make_unique<QubitIdentityAnalysis>(block)) {}

CommutationAnalysis::~CommutationAnalysis() = default;

bool CommutationAnalysis::haveSameOrderedQuantumOperands(Operation *lhs,
                                                         Operation *rhs) const {
  auto lhsInterface = dyn_cast_if_present<cudaq::quake::OperatorInterface>(lhs);
  auto rhsInterface = dyn_cast_if_present<cudaq::quake::OperatorInterface>(rhs);
  if (!lhsInterface || !rhsInterface)
    return false;

  return qubitIdentity->haveSameOrderedQubitIdentities(
             lhsInterface.getControls(), rhsInterface.getControls()) &&
         qubitIdentity->haveSameOrderedQubitIdentities(
             lhsInterface.getTargets(), rhsInterface.getTargets());
}

bool CommutationAnalysis::registerIdentityPreservingOperation(
    Operation *operation) {
  return operation && operation->getBlock() == block &&
         qubitIdentity->registerOperation(*operation);
}

bool CommutationAnalysis::prepareIdentityPreservingReplacement(
    Operation *operation, ValueRange replacement) {
  if (!operation || operation->getBlock() != block ||
      !qubitIdentity->replacementPreservesIdentities(*operation, replacement))
    return false;
  invalidateOperation(operation);
  return true;
}

void CommutationAnalysis::invalidateOperation(Operation *operation) {
  auto dependency = cacheDependencies.find(operation);
  if (dependency == cacheDependencies.end())
    return;

  llvm::SmallVector<OperationPair> incidentPairs(dependency->second.begin(),
                                                 dependency->second.end());
  cacheDependencies.erase(dependency);
  for (OperationPair pair : incidentPairs) {
    cache.erase(pair);
    Operation *other = pair.first == operation ? pair.second : pair.first;
    if (other == operation)
      continue;
    auto otherDependency = cacheDependencies.find(other);
    if (otherDependency == cacheDependencies.end())
      continue;
    otherDependency->second.erase(pair);
    if (otherDependency->second.empty())
      cacheDependencies.erase(otherDependency);
  }
}

void CommutationAnalysis::eraseOperation(Operation *operation) {
  if (!operation || operation->getBlock() != block)
    return;
  invalidateOperation(operation);
  qubitIdentity->eraseOperation(*operation);
}

CommutationResult CommutationAnalysis::getResult(Operation *lhs,
                                                 Operation *rhs) {
  if (!lhs || !rhs)
    return indeterminate(commutation_reason::null_operation);
  if (lhs->getBlock() != block || rhs->getBlock() != block)
    return indeterminate(commutation_reason::different_blocks);

  OperationPair key = getCanonicalCacheKey(lhs, rhs);
  auto cached = cache.find(key);
  if (cached != cache.end())
    return cached->second;
  auto result = evaluate(lhs, rhs, *qubitIdentity);
  auto [entry, inserted] = cache.try_emplace(key, result);
  if (inserted) {
    cacheDependencies[key.first].insert(key);
    cacheDependencies[key.second].insert(key);
  }
  return entry->second;
}

bool CommutationAnalysis::canCommute(Operation *lhs, Operation *rhs) {
  return getResult(lhs, rhs).status == commutation_status::commutes;
}
