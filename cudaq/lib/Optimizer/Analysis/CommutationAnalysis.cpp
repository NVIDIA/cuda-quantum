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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Matchers.h"
#include <functional>
#include <optional>
#include <utility>

using namespace mlir;

using cudaq::quake::Pauli;
using cudaq::quake::PauliWord;
using cudaq::quake::detail::CommutationAnalysis;
using cudaq::quake::detail::CommutationReason;
using cudaq::quake::detail::CommutationResult;
using cudaq::quake::detail::CommutationStatus;
using cudaq::quake::detail::QubitIdentityAnalysis;

namespace {
using QubitId = QubitIdentityAnalysis::QubitId;
using OperationPair = std::pair<Operation *, Operation *>;

struct ControlUse {
  QubitId qubitId;
  bool negated;

  bool operator==(const ControlUse &) const = default;
};

// Query-local view of a Quake operator's controls, targets, and support,
// expressed using analysis-local qubit identifiers.
struct OperationView {
  OperationView(Operation *operation,
                cudaq::quake::OperatorInterface operatorInterface)
      : operation(operation), interface(operatorInterface) {}

  // Underlying operation used for kind-specific commutation rules.
  Operation *operation;
  // Quake interface used to access parameters and quantum operands.
  cudaq::quake::OperatorInterface interface;
  // Controls in operand order, including their positive or negative polarity.
  llvm::SmallVector<ControlUse> controls;
  // Targets in operand order, preserving positional gate semantics.
  llvm::SmallVector<QubitId> targets;
  // Unique union of control and target qubits.
  llvm::DenseSet<QubitId> support;
  // Target lookup used by overlap and crossover rules.
  llvm::DenseSet<QubitId> targetQubitIds;
  // Control lookup and polarity used by controlled-operation rules.
  llvm::DenseMap<QubitId, bool> controlPolarities;
};

// Analysis-local Pauli product keyed by qubit rather than IR target order.
// This normalized form makes shared-qubit parity checks order-independent.
struct PauliAction {
  llvm::DenseMap<QubitId, Pauli> terms;
};
} // namespace

static CommutationResult commutes(CommutationReason reason) {
  return {CommutationStatus::Commutes, reason};
}

static CommutationResult doesNotCommute(CommutationReason reason) {
  return {CommutationStatus::DoesNotCommute, reason};
}

static CommutationResult indeterminate(CommutationReason reason) {
  return {CommutationStatus::Indeterminate, reason};
}

static OperationPair getCanonicalPair(Operation *lhs, Operation *rhs) {
  // std::less provides a total order for unrelated pointers. The order has no
  // semantic meaning; it only makes the symmetric cache key canonical.
  if (std::less<Operation *>{}(rhs, lhs))
    std::swap(lhs, rhs);
  return {lhs, rhs};
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

// Identify operations whose target action is on the Pauli-X axis.
static bool isXAxis(Operation *operation) {
  return isa<cudaq::quake::XOp, cudaq::quake::RxOp>(operation);
}

// Identify operations whose target action is on the Pauli-Y axis.
static bool isYAxis(Operation *operation) {
  return isa<cudaq::quake::YOp, cudaq::quake::RyOp>(operation);
}

// Identify operations whose target action is on the Pauli-Z axis.
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

// Prove that two views describe the same action on the same qubits.
static bool haveSameOperation(const OperationView &lhs,
                              const OperationView &rhs) {
  // Match the operation kind and every action-bearing interface value. Adjoint
  // state may differ because an operation commutes with its exact inverse.
  bool sameRecognizedKind =
      (isSupportedSharedOperation(lhs.operation) &&
       isSupportedSharedOperation(rhs.operation) &&
       lhs.operation->getName() == rhs.operation->getName()) ||
      haveSameCustomUnitaryDefinition(lhs.operation, rhs.operation);
  if (!sameRecognizedKind || lhs.controls != rhs.controls ||
      !haveSameTargets(lhs, rhs) ||
      !haveExactParameters(lhs.interface, rhs.interface))
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

// Compute whether shared Pauli factors contain an odd number of mismatches.
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

// Operations with no shared control or target qubit commute independently of
// their gate semantics.
static bool haveDisjointQuantumSupport(const OperationView &lhs,
                                       const OperationView &rhs) {
  const auto *smaller = &lhs.support;
  const auto *larger = &rhs.support;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  return llvm::none_of(
      *smaller, [&](QubitId qubitId) { return larger->contains(qubitId); });
}

// Detect a qubit used as a target by one operation and a control by the other.
static bool hasTargetControlCrossover(const OperationView &lhs,
                                      const OperationView &rhs) {
  return llvm::any_of(lhs.targetQubitIds,
                      [&](QubitId qubitId) {
                        return rhs.controlPolarities.contains(qubitId);
                      }) ||
         llvm::any_of(rhs.targetQubitIds, [&](QubitId qubitId) {
           return lhs.controlPolarities.contains(qubitId);
         });
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
    bool isSharedControl = controlled.controlPolarities.contains(qubitId);
    bool isSharedTarget = controlled.targetQubitIds.contains(qubitId);
    hasOverlap |= isSharedControl || isSharedTarget;
    return !isSharedTarget;
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
  const auto *smaller = &lhs.targetQubitIds;
  const auto *larger = &rhs.targetQubitIds;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  return llvm::none_of(
      *smaller, [&](QubitId qubitId) { return larger->contains(qubitId); });
}

// Test target-only commutation after controlled-rule preconditions are met.
static bool targetActionsCommute(const OperationView &lhs,
                                 const OperationView &rhs) {
  if (haveDisjointTargetSupport(lhs, rhs))
    return true;
  if (lhs.operation->getName() == rhs.operation->getName() &&
      haveSameTargets(lhs, rhs) &&
      haveExactParameters(lhs.interface, rhs.interface)) {
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
  const auto *smaller = &lhs.controlPolarities;
  const auto *larger = &rhs.controlPolarities;
  if (larger->size() < smaller->size())
    std::swap(smaller, larger);
  for (auto [qubitId, negated] : *smaller) {
    auto other = larger->find(qubitId);
    if (other != larger->end() && negated != other->second)
      return true;
  }
  return false;
}

// Operators on disjoint qubits commute because
// (A tensor I)(I tensor B) = A tensor B = (I tensor B)(A tensor I).
static std::optional<CommutationResult>
tryDisjointSupport(const OperationView &lhs, const OperationView &rhs) {
  if (haveDisjointQuantumSupport(lhs, rhs))
    return commutes(CommutationReason::DisjointSupport);
  return std::nullopt;
}

// U commutes with itself and its exact adjoint because UU^-1 = U^-1U = I.
static std::optional<CommutationResult>
trySameOperation(const OperationView &lhs, const OperationView &rhs) {
  if (haveSameOperation(lhs, rhs))
    return commutes(CommutationReason::SameOperation);
  return std::nullopt;
}

// Computational-basis diagonal matrices satisfy D1 D2 = D2 D1 because their
// products are pointwise scalar products in the same basis.
static std::optional<CommutationResult>
tryComputationalDiagonal(const OperationView &lhs, const OperationView &rhs) {
  if (isComputationalDiagonal(lhs.operation) &&
      isComputationalDiagonal(rhs.operation))
    return commutes(CommutationReason::ComputationalDiagonal);
  return std::nullopt;
}

// Operators that are functions of the same Pauli axis P commute because
// f(P) g(P) = g(P) f(P). This rule recognizes PhasedRx axes only when their
// phase values match exactly.
static std::optional<CommutationResult> trySameAxis(const OperationView &lhs,
                                                    const OperationView &rhs) {
  if (lhs.controls.empty() && rhs.controls.empty() &&
      haveSameAxisTargetAction(lhs, rhs))
    return commutes(CommutationReason::SameAxis);
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
    return commutes(CommutationReason::EvenPauliParity);
  if (isPauliOperator(lhs.operation) && isPauliOperator(rhs.operation))
    return doesNotCommute(CommutationReason::OddPauliParity);
  return std::nullopt;
}

// Quake control polarity selects a computational-basis projector P. A diagonal
// action D on that control satisfies DP = PD for either polarity, so the proof
// holds for every input state. This applies when every shared qubit is only a
// control of the other operation, never one of its targets.
// TODO: This rule cannot recognize a control basis established by surrounding
// basis changes, such as H-C(U)-H. Sequence-level basis tracking would cover
// those cases, while reusable per-operand commuting-basis properties would
// avoid hard-coding the supported individual operations.
static std::optional<CommutationResult>
tryDiagonalOnControls(const OperationView &lhs, const OperationView &rhs) {
  if (diagonalOverlapsOnlyControls(lhs, rhs) ||
      diagonalOverlapsOnlyControls(rhs, lhs))
    return commutes(CommutationReason::DiagonalOnControls);
  return std::nullopt;
}

// With no target-control crossover, commuting target actions and commuting
// control projectors make every term of the controlled products commute.
static std::optional<CommutationResult>
tryCompatibleControlledTargets(const OperationView &lhs,
                               const OperationView &rhs) {
  if ((!lhs.controls.empty() || !rhs.controls.empty()) &&
      !hasTargetControlCrossover(lhs, rhs) && targetActionsCommute(lhs, rhs))
    return commutes(CommutationReason::CompatibleControlledTargets);
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
    return commutes(CommutationReason::MutuallyExclusiveControls);
  return std::nullopt;
}

using CommutationRule = std::optional<CommutationResult> (*)(
    const OperationView &, const OperationView &);

// Apply general rules, reject unsupported shared-support cases, then apply the
// remaining shared-support rules in stable proof-reason precedence order.
static CommutationResult dispatchRules(const OperationView &lhs,
                                       const OperationView &rhs) {
  if (auto result = tryDisjointSupport(lhs, rhs))
    return *result;
  if (auto result = trySameOperation(lhs, rhs))
    return *result;
  if (!isSupportedSharedOperation(lhs.operation) ||
      !isSupportedSharedOperation(rhs.operation))
    return indeterminate(CommutationReason::NoApplicableRule);
  if (!hasSupportedPauliWord(lhs) || !hasSupportedPauliWord(rhs))
    return indeterminate(CommutationReason::UnsupportedPauliWord);

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
  return indeterminate(CommutationReason::NoApplicableRule);
}

// Populate the normalized view used by commutation rules. Resolve supported
// scalar controls and targets to analysis-local qubit IDs, record their roles
// and control polarities, and reject unmapped or duplicate qubit uses.
static std::optional<CommutationReason>
populateOperationView(OperationView &view,
                      const QubitIdentityAnalysis &qubitIdentity) {
  // Valid Quake IR guarantees that polarity metadata, when present, has one
  // entry per control operand.
  auto negatedControls = view.interface.getNegatedControls();
  auto controls = view.interface.getControls();

  // A supported operator may use a qubit in only one control or target role.
  // Track all resolved IDs here so duplicates are rejected across both groups.
  llvm::DenseSet<QubitId> seenQubitIds;

  // Preserve control operand order while also building the support and
  // identity-to-polarity lookup required by controlled-operation rules.
  view.controls.reserve(controls.size());
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (!isa<cudaq::quake::WireType, cudaq::quake::ControlType>(
            control.getType()))
      return CommutationReason::UnsupportedQuantumOperandType;
    auto qubitId = qubitIdentity.getQubitId(control);
    if (!qubitId)
      return CommutationReason::UnmappedQubitId;
    if (!seenQubitIds.insert(*qubitId).second)
      return CommutationReason::DuplicateQubitOperand;
    view.controls.push_back(
        {*qubitId, negatedControls && (*negatedControls)[index]});
    view.support.insert(*qubitId);
    view.controlPolarities.try_emplace(*qubitId, negatedControls &&
                                                     (*negatedControls)[index]);
  }

  // Preserve target order for positional gate semantics and build the target
  // membership lookup used by overlap and crossover rules.
  auto targets = view.interface.getTargets();
  view.targets.reserve(targets.size());
  for (Value target : targets) {
    if (!isa<cudaq::quake::WireType>(target.getType()))
      return CommutationReason::UnsupportedQuantumOperandType;
    auto qubitId = qubitIdentity.getQubitId(target);
    if (!qubitId)
      return CommutationReason::UnmappedQubitId;
    if (!seenQubitIds.insert(*qubitId).second)
      return CommutationReason::DuplicateQubitOperand;
    view.targets.push_back(*qubitId);
    view.support.insert(*qubitId);
    view.targetQubitIds.insert(*qubitId);
  }
  return std::nullopt;
}

// Validate and normalize a query, then apply general and shared-support rules.
static CommutationResult evaluate(Operation *lhs, Operation *rhs,
                                  const QubitIdentityAnalysis &qubitIdentity) {
  auto lhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(lhs);
  auto rhsInterface = dyn_cast<cudaq::quake::OperatorInterface>(rhs);
  if (!lhsInterface || !rhsInterface)
    return indeterminate(CommutationReason::UnsupportedOperationKind);

  OperationView lhsView{lhs, lhsInterface};
  OperationView rhsView{rhs, rhsInterface};
  if (auto reason = populateOperationView(lhsView, qubitIdentity))
    return indeterminate(*reason);
  if (auto reason = populateOperationView(rhsView, qubitIdentity))
    return indeterminate(*reason);

  return dispatchRules(lhsView, rhsView);
}

llvm::StringRef
cudaq::quake::detail::getCommutationReasonId(CommutationReason reason) {
  switch (reason) {
  case CommutationReason::DisjointSupport:
    return "disjoint-support";
  case CommutationReason::SameOperation:
    return "same-operation";
  case CommutationReason::ComputationalDiagonal:
    return "computational-diagonal";
  case CommutationReason::SameAxis:
    return "same-axis";
  case CommutationReason::EvenPauliParity:
    return "even-pauli-parity";
  case CommutationReason::OddPauliParity:
    return "odd-pauli-parity";
  case CommutationReason::DiagonalOnControls:
    return "diagonal-on-controls";
  case CommutationReason::CompatibleControlledTargets:
    return "compatible-controlled-targets";
  case CommutationReason::MutuallyExclusiveControls:
    return "mutually-exclusive-controls";
  case CommutationReason::NullOperation:
    return "null-operation";
  case CommutationReason::DifferentBlocks:
    return "different-blocks";
  case CommutationReason::UnsupportedOperationKind:
    return "unsupported-operation-kind";
  case CommutationReason::UnsupportedQuantumOperandType:
    return "unsupported-quantum-operand-type";
  case CommutationReason::UnmappedQubitId:
    return "unmapped-qubit-id";
  case CommutationReason::DuplicateQubitOperand:
    return "duplicate-qubit-operand";
  case CommutationReason::UnsupportedPauliWord:
    return "unsupported-pauli-word";
  case CommutationReason::NoApplicableRule:
    return "no-applicable-rule";
  }
  llvm_unreachable("unhandled commutation reason");
}

CommutationAnalysis::CommutationAnalysis(Block &block)
    : block(&block),
      qubitIdentity(std::make_unique<QubitIdentityAnalysis>(block)) {}

CommutationAnalysis::~CommutationAnalysis() = default;

CommutationResult CommutationAnalysis::getResult(Operation *lhs,
                                                 Operation *rhs) {
  if (!lhs || !rhs)
    return indeterminate(CommutationReason::NullOperation);
  if (lhs->getBlock() != block || rhs->getBlock() != block)
    return indeterminate(CommutationReason::DifferentBlocks);

  OperationPair key = getCanonicalPair(lhs, rhs);
  auto cached = cache.find(key);
  if (cached != cache.end())
    return cached->second;
  auto result = evaluate(key.first, key.second, *qubitIdentity);
  cache.try_emplace(key, result);
  return result;
}

bool CommutationAnalysis::canCommute(Operation *lhs, Operation *rhs) {
  return static_cast<bool>(getResult(lhs, rhs));
}
