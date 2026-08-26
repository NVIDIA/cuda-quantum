/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"
#include <cstdint>

namespace cudaq::opt {
#define GEN_PASS_DEF_PHASEFOLDING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "phase-folding"

using namespace mlir;

// AXIS-SPECIFIC: Defines which operations break a circuit into subcircuits
#define CIRCUIT_BREAKERS(MACRO)                                                \
  MACRO(YOp), MACRO(HOp), MACRO(R1Op), MACRO(RxOp), MACRO(PhasedRxOp),         \
      MACRO(RyOp), MACRO(U2Op), MACRO(U3Op)
#define Z_AXIS_ROTATIONS(MACRO) MACRO(RzOp), MACRO(SOp), MACRO(TOp), MACRO(ZOp)
#define RAW(X) cudaq::quake::X
#define RAW_CIRCUIT_BREAKERS CIRCUIT_BREAKERS(RAW)
#define RAW_Z_AXIS_ROTATIONS Z_AXIS_ROTATIONS(RAW)

namespace {

// ============================================================================
// Phase algebra
// ============================================================================

struct PhaseVariable {
  size_t idx;
  PhaseVariable(size_t index) : idx(index) {}
  bool operator==(PhaseVariable other) { return idx == other.idx; }
};

using PhaseKey = std::pair<SmallVector<unsigned, 4>, bool>;

struct PhaseKeyInfo {
  static PhaseKey getEmptyKey() { return {{unsigned(-1)}, false}; }
  static PhaseKey getTombstoneKey() { return {{unsigned(-1)}, true}; }
  static unsigned getHashValue(const PhaseKey &k) {
    auto h = llvm::hash_value(k.second);
    for (auto idx : k.first)
      h = llvm::hash_combine(h, idx);
    return (unsigned)h;
  }
  static bool isEqual(const PhaseKey &a, const PhaseKey &b) { return a == b; }
};

/// A `Phase` is an exclusive sum of `PhaseVariable`s plus an inversion flag.
/// Two rotations on qubits with equal Phases can be merged into one rotation.
class Phase {
  SetVector<PhaseVariable *> vars;
  bool isInverted;

public:
  Phase() : isInverted(false) {}
  Phase(PhaseVariable *var) : isInverted(false) { vars.insert(var); }

  bool operator==(Phase other) {
    for (auto var : vars)
      if (!other.vars.contains(var))
        return false;
    for (auto var : other.vars)
      if (!vars.contains(var))
        return false;
    return isInverted == other.isInverted;
  }

  static Phase sum(const Phase &p1, const Phase &p2) {
    Phase p;
    for (auto var : p1.vars)
      p.vars.insert(var);
    for (auto var : p2.vars)
      if (p.vars.contains(var))
        p.vars.remove(var);
      else
        p.vars.insert(var);
    p.isInverted = (p1.isInverted != p2.isInverted);
    return p;
  }

  static Phase invert(const Phase &p1) {
    Phase p;
    p.vars.insert(p1.vars.begin(), p1.vars.end());
    p.isInverted = !p1.isInverted;
    return p;
  }

  PhaseKey toKey() const {
    SmallVector<unsigned, 4> indices;
    for (auto *var : vars)
      indices.push_back(var->idx);
    sort(indices);
    return {indices, isInverted};
  }
};

/// Compatible rotations and the original operation where their replacement
/// can be materialized after analysis.
struct RotationGroup {
  SmallVector<cudaq::quake::OperatorInterface> rotations;
  Operation *anchor = nullptr;
  std::int64_t namedQuarterTurns = 0;
  bool hasDynamicRotation = false;
};

/// Deferred rotation groups for one subcircuit.
using FoldPlan = SmallVector<RotationGroup>;

static std::optional<std::int64_t>
getQuarterPiUnits(cudaq::quake::OperatorInterface rotation) {
  auto *op = rotation.getOperation();
  if (isa<cudaq::quake::ZOp>(op))
    return 4;
  if (isa<cudaq::quake::SOp>(op))
    return rotation.isAdj() ? -2 : 2;
  if (isa<cudaq::quake::TOp>(op))
    return rotation.isAdj() ? -1 : 1;
  return std::nullopt;
}

static unsigned normalizeQuarterTurns(std::int64_t units) {
  auto normalized = units % 8;
  if (normalized < 0)
    normalized += 8;
  return normalized;
}

static std::int64_t getSignedQuarterTurns(std::int64_t units) {
  auto normalized = normalizeQuarterTurns(units);
  return normalized > 4 ? normalized - 8 : normalized;
}

class PhaseStorage {
  DenseMap<PhaseKey, unsigned, PhaseKeyInfo> phaseToGroup;
  SmallVector<RotationGroup> groups;
  DominanceInfo &domInfo;

  bool canUseRotationAt(cudaq::quake::OperatorInterface rotation,
                        Operation *insertionPoint) {
    auto *rotationOp = rotation.getOperation();
    if (getQuarterPiUnits(rotation))
      return true;
    if (auto rzOp = dyn_cast<cudaq::quake::RzOp>(rotationOp)) {
      Value angleValue = rzOp.getOperand(0);
      return domInfo.dominates(angleValue, insertionPoint);
    }
    return false;
  }

  bool canUseAccumulatedAngleAt(const RotationGroup &group,
                                Operation *insertionPoint) {
    if (!group.hasDynamicRotation)
      return true;
    if (group.rotations.size() == 1)
      return canUseRotationAt(group.rotations.front(), insertionPoint);

    // Once a sum is required, its value is local to the selected anchor.
    return domInfo.dominates(group.anchor, insertionPoint);
  }

  void startGroup(const PhaseKey &key,
                  cudaq::quake::OperatorInterface rotation) {
    unsigned index = groups.size();
    auto &group = groups.emplace_back();
    group.rotations.push_back(rotation);
    group.anchor = rotation.getOperation();
    if (auto units = getQuarterPiUnits(rotation))
      group.namedQuarterTurns = *units;
    else
      group.hasDynamicRotation = true;
    phaseToGroup[key] = index;
  }

public:
  PhaseStorage(DominanceInfo &domInfo) : domInfo(domInfo) {}

  void addRotation(cudaq::quake::OperatorInterface rotation, Phase phase) {
    auto key = phase.toKey();
    auto it = phaseToGroup.find(key);
    if (it == phaseToGroup.end()) {
      startGroup(key, rotation);
      return;
    }

    auto &group = groups[it->second];
    auto *rotationOp = rotation.getOperation();
    auto quarterTurns = getQuarterPiUnits(rotation);
    if (rotation.getTarget(0).getDefiningOp() == group.anchor ||
        canUseAccumulatedAngleAt(group, rotationOp)) {
      group.anchor = rotationOp;
    } else if (!canUseRotationAt(rotation, group.anchor)) {
      phaseToGroup.erase(it);
      startGroup(key, rotation);
      return;
    }

    group.rotations.push_back(rotation);
    if (quarterTurns)
      group.namedQuarterTurns += *quarterTurns;
    else
      group.hasDynamicRotation = true;

    // End exact named-gate cancellations here. A later rotation starts a new
    // group instead of materializing an unnecessary zero-angle sum.
    if (!group.hasDynamicRotation &&
        normalizeQuarterTurns(group.namedQuarterTurns) == 0)
      phaseToGroup.erase(it);
  }

  FoldPlan takePlan() {
    FoldPlan plan;
    for (auto &group : groups)
      if (group.rotations.size() > 1)
        plan.push_back(std::move(group));
    return plan;
  }
};

static Operation *createNamedRotation(OpBuilder &builder, Location loc,
                                      Value wireIn, unsigned quarterTurns) {
  auto wireTy = cudaq::quake::WireType::get(builder.getContext());
  switch (quarterTurns) {
  case 0:
    return nullptr;
  case 1:
    return cudaq::quake::TOp::create(builder, loc, TypeRange{wireTy}, false,
                                     ValueRange{}, ValueRange{},
                                     ValueRange{wireIn}, {});
  case 2:
    return cudaq::quake::SOp::create(builder, loc, TypeRange{wireTy}, false,
                                     ValueRange{}, ValueRange{},
                                     ValueRange{wireIn}, {});
  case 3: {
    Value wire = cudaq::quake::SOp::create(builder, loc, TypeRange{wireTy},
                                           false, ValueRange{}, ValueRange{},
                                           ValueRange{wireIn}, {})
                     ->getResult(0);
    return cudaq::quake::TOp::create(builder, loc, TypeRange{wireTy}, false,
                                     ValueRange{}, ValueRange{},
                                     ValueRange{wire}, {});
  }
  case 4:
    return cudaq::quake::ZOp::create(builder, loc, TypeRange{wireTy}, false,
                                     ValueRange{}, ValueRange{},
                                     ValueRange{wireIn}, {});
  case 5: {
    Value wire = cudaq::quake::ZOp::create(builder, loc, TypeRange{wireTy},
                                           false, ValueRange{}, ValueRange{},
                                           ValueRange{wireIn}, {})
                     ->getResult(0);
    return cudaq::quake::TOp::create(builder, loc, TypeRange{wireTy}, false,
                                     ValueRange{}, ValueRange{},
                                     ValueRange{wire}, {});
  }
  case 6:
    return cudaq::quake::SOp::create(builder, loc, TypeRange{wireTy}, true,
                                     ValueRange{}, ValueRange{},
                                     ValueRange{wireIn}, {});
  case 7:
    return cudaq::quake::TOp::create(builder, loc, TypeRange{wireTy}, true,
                                     ValueRange{}, ValueRange{},
                                     ValueRange{wireIn}, {});
  }
  llvm_unreachable("quarter turns must be reduced modulo eight");
}

static Value materializeMixedAngle(OpBuilder &builder,
                                   const RotationGroup &group) {
  Value angle;
  auto namedUnits = getSignedQuarterTurns(group.namedQuarterTurns);
  bool emittedNamedAngle = false;

  for (auto rotation : group.rotations) {
    auto rz = dyn_cast<cudaq::quake::RzOp>(rotation.getOperation());
    Value nextAngle;
    if (rz) {
      nextAngle = rz.getOperand(0);
      if (rotation.isAdj())
        nextAngle = arith::NegFOp::create(builder, rz->getLoc(), nextAngle);
    } else if (namedUnits != 0 && !emittedNamedAngle) {
      nextAngle = cudaq::opt::factory::createF64Constant(
          group.anchor->getLoc(), builder, namedUnits * M_PI_4);
      emittedNamedAngle = true;
    } else {
      continue;
    }
    if (!angle)
      angle = nextAngle;
    else
      angle = arith::AddFOp::create(builder, group.anchor->getLoc(), angle,
                                    nextAngle);
  }
  assert(angle && "mixed rotation group must contain an Rz angle");
  return angle;
}

static void applyRotationGroup(RotationGroup &group) {
  if (group.rotations.size() < 2)
    return;

  auto anchor = cast<cudaq::quake::OperatorInterface>(group.anchor);
  OpBuilder builder(group.anchor);
  auto loc = group.anchor->getLoc();
  Value wireIn = anchor.getTarget(0);
  Operation *replacement = nullptr;
  if (group.hasDynamicRotation) {
    auto wireTy = cudaq::quake::WireType::get(builder.getContext());
    Value angle = materializeMixedAngle(builder, group);
    replacement = cudaq::quake::RzOp::create(
        builder, loc, TypeRange{wireTy}, false, ValueRange{angle}, ValueRange{},
        ValueRange{wireIn}, {});
  } else {
    replacement = createNamedRotation(
        builder, loc, wireIn, normalizeQuarterTurns(group.namedQuarterTurns));
  }
  Value replacementWire = replacement ? replacement->getResult(0) : wireIn;
  group.anchor->getResult(0).replaceAllUsesWith(replacementWire);
  for (auto it = group.rotations.rbegin(); it != group.rotations.rend(); ++it) {
    Operation *rotationOp = it->getOperation();
    if (rotationOp == group.anchor)
      continue;
    rotationOp->getResult(0).replaceAllUsesWith(it->getTarget(0));
    rotationOp->erase();
  }
  group.anchor->erase();
}

static void applyFoldPlan(FoldPlan &plan) {
  for (auto &group : plan)
    applyRotationGroup(group);
}

// ============================================================================
// Wire semantics implementation
// ============================================================================

namespace wire {

static unsigned calculateSkip(Operation *op) {
  unsigned i = 0;
  for (auto type : op->getOperandTypes()) {
    if (isa<cudaq::quake::WireType>(type))
      return i;
    i++;
  }
  return i;
}

static Value getNextOperand(Value v) {
  auto result = dyn_cast<OpResult>(v);
  auto op = result.getDefiningOp();
  auto skip = calculateSkip(op);
  return op->getOperand(result.getResultNumber() + skip);
}

static OpResult getNextResult(Value v) {
  assert(v.hasOneUse());
  auto correspondingOperand = v.getUses().begin();
  auto op = correspondingOperand.getUser();
  auto skip = calculateSkip(op);
  return op->getResult(correspondingOperand.getOperand()->getOperandNumber() -
                       skip);
}

// AXIS-SPECIFIC: could allow controlled y and z here
static bool isControlledOp(Operation *op) {
  if (!isa<cudaq::quake::XOp>(op))
    return false;
  auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
  if (!opi || opi.getControls().size() != 1)
    return false;
  for (auto operand : cudaq::quake::getQuantumOperands(op))
    if (!isa<cudaq::quake::WireType>(operand.getType()))
      return false;
  return true;
}

static Block *getPhaseFoldingBlock(Operation *op) {
  auto *block = op->getBlock();
  auto *parent = block->getParentOp();
  // A wire captured by an ordinary scope reaches nested operations without
  // passing through the ScopeOp, so compare the enclosing folding domains
  // rather than relying on the wire walk to encounter the boundary itself.
  while (auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(parent)) {
    if (scope.getAtomicQuantumRegionAttr() || !scope.getRegion().hasOneBlock())
      break;
    block = scope->getBlock();
    parent = block->getParentOp();
  }
  return block;
}

static bool isSubCircuitTerminationPoint(Operation *op) {
  if (!op)
    return true;
  if (!isQuakeOperation(op))
    return true;
  if (isa<RAW_CIRCUIT_BREAKERS>(op))
    return true;
  if (isa<cudaq::quake::NullWireOp>(op))
    return true;
  auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
  if (!opi)
    return true;
  // Only allow single control (for CNOT/NOT); Z-rotations must be uncontrolled
  if (opi.getControls().size() > 0 && !isa<cudaq::quake::XOp>(op))
    return true;
  if (isa<RAW_Z_AXIS_ROTATIONS>(op) && !opi.getControls().empty())
    return true;
  // TODO: support other qubit types (ref, veq) in phase folding
  for (auto operand : cudaq::quake::getQuantumOperands(op))
    if (cudaq::quake::isQuantumType(operand.getType()) &&
        !isa<cudaq::quake::WireType>(operand.getType()))
      return true;
  return false;
}

class Subcircuit {
protected:
  SetVector<Operation *> ops;
  SetVector<Value> initial_wires;
  SetVector<Value> terminal_wires;
  Block *start;
  // TODO: these three are really intermediate state for constructing the
  // subcircuit; would be nice to turn them into local arguments instead
  SetVector<Value> termination_points;
  SetVector<Value> anchor_points;
  SetVector<Value> seen;
  DenseMap<Value, Value> scope_result_to_continue_operand;

  bool isTerminationPoint(Operation *op) {
    if (!op)
      return true;
    if (isSubCircuitTerminationPoint(op))
      return true;
    return getPhaseFoldingBlock(op) != start;
  }

  bool isAfterTerminationPoint(Value wire) {
    return isTerminationPoint(wire.getDefiningOp());
  }

  void addAnchorPoint(Value v) { anchor_points.insert(v); }
  void addTerminationPoint(Value v) { termination_points.insert(v); }

  void calculateSubcircuitForQubitForward(Value v) {
    if (seen.contains(v))
      return;
    seen.insert(v);
    if (!v.hasOneUse()) {
      addTerminationPoint(v);
      return;
    }
    OpOperand *use = &*v.getUses().begin();
    Operation *op = use->getOwner();
    if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(op)) {
      // Ordinary single-block scopes preserve wire identity, so their yielded
      // wire can remain in this subcircuit. Marked or CFG-bearing scopes do not
      // have that transparent-boundary contract.
      auto scope = dyn_cast<cudaq::cc::ScopeOp>(cont->getParentOp());
      if (!scope || scope.getAtomicQuantumRegionAttr() ||
          !scope.getInitRegion().hasOneBlock() ||
          scope.getInitRegion().front().getTerminator() != op ||
          cont.getNumOperands() != scope->getNumResults() ||
          use->getOperandNumber() >= scope->getNumResults()) {
        addTerminationPoint(v);
        return;
      }
      auto nextResult = scope->getResult(use->getOperandNumber());
      if (!isa<cudaq::quake::WireType>(nextResult.getType())) {
        addTerminationPoint(v);
        return;
      }
      scope_result_to_continue_operand[nextResult] = use->get();
      calculateSubcircuitForQubitForward(nextResult);
      return;
    }
    if (isTerminationPoint(op)) {
      addTerminationPoint(v);
      return;
    }
    ops.insert(op);
    auto nextResult = getNextResult(v);
    // Controlled not: add an anchor point for the other qubit
    if (op->getResults().size() > 1) {
      auto control = op->getResult(0);
      auto target = op->getResult(1);
      if (nextResult == control)
        addAnchorPoint(target);
      else
        addAnchorPoint(control);
    }
    calculateSubcircuitForQubitForward(nextResult);
  }

  void calculateSubcircuitForQubitBackward(Value v) {
    if (seen.contains(v))
      return;
    seen.insert(v);
    Operation *op = v.getDefiningOp();
    if (auto scope = dyn_cast_or_null<cudaq::cc::ScopeOp>(op)) {
      // The ScopeOp result hides the quantum operation that produced the
      // yielded wire. Recover that wire so the backward walk follows the same
      // path as the forward walk.
      if (scope.getAtomicQuantumRegionAttr() ||
          !scope.getRegion().hasOneBlock()) {
        addTerminationPoint(v);
        return;
      }
      auto resultIndex = cast<OpResult>(v).getResultNumber();
      auto continueOp = dyn_cast_or_null<cudaq::cc::ContinueOp>(
          scope.getInitRegion().front().getTerminator());
      if (!continueOp || continueOp.getNumOperands() <= resultIndex) {
        addTerminationPoint(v);
        return;
      }
      Value continueWire = continueOp.getOperand(resultIndex);
      scope_result_to_continue_operand[v] = continueWire;
      calculateSubcircuitForQubitBackward(continueWire);
      return;
    }
    if (isTerminationPoint(op)) {
      addTerminationPoint(v);
      return;
    }
    ops.insert(op);
    auto nextOperand = getNextOperand(v);
    // Controlled not: add an anchor point for the other qubit
    // Use getResults() as Rz has two operands but only one result
    if (op->getResults().size() > 1) {
      auto control = op->getOperand(0);
      auto target = op->getOperand(1);
      if (nextOperand == control)
        addAnchorPoint(target);
      else
        addAnchorPoint(control);
    }
    calculateSubcircuitForQubitBackward(nextOperand);
  }

  void calculateInitialSubcircuit(Operation *op) {
    // AXIS-SPECIFIC: This could be any controlled operation
    auto cnot = dyn_cast<cudaq::quake::XOp>(op);
    assert(cnot && cnot.getWires().size() == 2);
    ops.insert(cnot);
    anchor_points.insert(cnot->getResult(1));
    calculateSubcircuitForQubitForward(cnot->getResult(0));
    calculateSubcircuitForQubitBackward(cnot->getOperand(0));
    while (!anchor_points.empty()) {
      auto next = anchor_points.back();
      anchor_points.pop_back();
      if (seen.contains(next))
        continue;
      calculateSubcircuitForQubitForward(next);
      // Remove next from seen for working backwards
      seen.remove(next);
      calculateSubcircuitForQubitBackward(next);
    }
  }

public:
  Subcircuit(Operation *cnot, DenseSet<Operation *> &processedOps) {
    // The anchor's folding domain does not change while building the
    // subcircuit, so cache it.
    start = getPhaseFoldingBlock(cnot);
    calculateInitialSubcircuit(cnot);
    // TODO: there is a performance issue that the current pruning definition
    // will always preference earlier operations, so a large interconnected
    // circuit will always tend towards the same subcircuit early in the circuit
    // and will not process later CNOTs, so the same early subcircuit will be
    // processed repeatedly. For value semantics, we don't actually need to
    // prune because we can just assign fresh phase variables for wires after
    // circuit breaking ops. Another possible option is to stop at ops in
    // `processedOps`. This is likely also an issue in the ref semantics form.
    for (auto *op : ops)
      processedOps.insert(op);

    for (auto w : termination_points)
      if (isAfterTerminationPoint(w))
        initial_wires.insert(w);
      else
        terminal_wires.insert(w);
  }

  SetVector<Value> getInitialWires() { return initial_wires; }
  size_t getNumOps() { return ops.size(); }

  float getRotationWeight() {
    if (ops.empty())
      return 0.0f;
    size_t rotCount = 0;
    for (auto *op : ops)
      if (isa<RAW_Z_AXIS_ROTATIONS>(op))
        rotCount++;
    return (float)rotCount / (float)ops.size();
  }

  SmallVector<Operation *> getOrderedOps() {
    DenseMap<Operation *, unsigned> predecessorCounts;
    DenseMap<Operation *, SmallVector<Operation *>> users;

    // The subcircuit already identifies every relevant operation. Order that
    // slice from its quantum def-use edges instead of scanning ancestor blocks.
    for (auto *op : ops) {
      llvm::SmallPtrSet<Operation *, 2> predecessors;
      for (Value operand : cudaq::quake::getQuantumOperands(op)) {
        auto *predecessor = resolveScopeResult(operand).getDefiningOp();
        if (predecessor && ops.contains(predecessor))
          predecessors.insert(predecessor);
      }
      predecessorCounts[op] = predecessors.size();
      for (auto *predecessor : predecessors)
        users[predecessor].push_back(op);
    }

    SmallVector<Operation *> ready;
    for (auto *op : ops)
      if (predecessorCounts[op] == 0)
        ready.push_back(op);

    SmallVector<Operation *> ordered;
    for (size_t next = 0; next < ready.size(); ++next) {
      Operation *op = ready[next];
      ordered.push_back(op);
      for (auto *user : users[op]) {
        auto &count = predecessorCounts[user];
        assert(count > 0 && "ready operation released more than once");
        if (--count == 0)
          ready.push_back(user);
      }
    }
    assert(ordered.size() == ops.size() &&
           "wire def-use graph must be acyclic");
    return ordered;
  }

  Value resolveScopeResult(Value v) {
    auto it = scope_result_to_continue_operand.find(v);
    while (it != scope_result_to_continue_operand.end()) {
      v = it->second;
      it = scope_result_to_continue_operand.find(v);
    }
    return v;
  }
};

} // namespace wire

// ============================================================================
// Pass
// ============================================================================

/// Phase-polynomial rotation merging pass.
///
/// Expected input: wire-semantics IR (`!quake.wire` SSA values) where each
/// qubit is represented as a separate wire definition. Run
/// `factor-quantum-alloc` followed by `memtoreg{quantum=1}` before this pass
/// to split veq allocations into individual wires and convert to wire form.
/// Ops with non-wire quantum operands (`!quake.ref`, `!quake.veq`) are
/// treated as subcircuit termination points and are left unmodified.
///
/// A subcircuit is a maximal region anchored on a single-control-X (CNOT)
/// gate and bounded by termination points. The following op classes are
/// allowed inside a subcircuit:
///   - Single-qubit NOT (quake.x, uncontrolled): inverts a wire's phase.
///   - CNOT (quake.x, single control): XORs control phase into target phase.
///   - Swap (quake.swap): exchanges the phases of two wires.
///   - Z-axis rotations (quake.rz, quake.s, quake.t, quake.z, and their
///     adjoints), uncontrolled: rotation candidates for merging.
/// All other ops (H, Y, Rx, Ry, R1, ...) terminate the subcircuit.
///
/// Compatible Z-axis rotations with the same phase are collected before the
/// IR is changed. Named gates are accumulated exactly in quarter turns and
/// materialized as a canonical named gate sequence. A group containing
/// `quake.rz` is materialized as one `quake.rz` whose angle combines its
/// dynamic operands with the net named-gate contribution. Each group is placed
/// at an original operation where every required angle is available.
class PhaseFoldingPass
    : public cudaq::opt::impl::PhaseFoldingBase<PhaseFoldingPass> {
  using PhaseFoldingBase::PhaseFoldingBase;

  FoldPlan planWirePhaseFolding(wire::Subcircuit *subcircuit,
                                DominanceInfo &domInfo) {
    DenseMap<Value, Phase> wirePhase;
    auto getWirePhase = [&](Value v) -> Phase {
      // A scope result and its continue operand name the same wire but are
      // distinct SSA keys. Normalize them before consulting the phase map.
      return wirePhase[subcircuit->resolveScopeResult(v)];
    };
    SmallVector<std::unique_ptr<PhaseVariable>> vars;
    PhaseStorage store(domInfo);
    size_t i = 0;

    for (auto w : subcircuit->getInitialWires()) {
      auto &var = vars.emplace_back(std::make_unique<PhaseVariable>(i++));
      wirePhase[w] = Phase(var.get());
    }

    for (auto *op : subcircuit->getOrderedOps()) {
      if (wire::isControlledOp(op)) {
        auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
        Phase ctrlPhase = getWirePhase(opi.getControls().front());
        Phase tgtPhase = getWirePhase(opi.getTarget(0));
        wirePhase[op->getResult(0)] = ctrlPhase;
        wirePhase[op->getResult(1)] = Phase::sum(ctrlPhase, tgtPhase);
      } else if (isa<cudaq::quake::XOp>(op)) {
        // AXIS-SPECIFIC: Would want to handle y and z gates here too
        auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
        wirePhase[op->getResult(0)] =
            Phase::invert(getWirePhase(opi.getTarget(0)));
      } else if (isa<RAW_Z_AXIS_ROTATIONS>(op)) {
        auto opi = cast<cudaq::quake::OperatorInterface>(op);
        Phase p = getWirePhase(opi.getTarget(0));
        store.addRotation(opi, p);
        wirePhase[op->getResult(0)] = p;
      } else if (auto swap = dyn_cast<cudaq::quake::SwapOp>(op)) {
        Phase p0 = getWirePhase(swap.getTarget(0));
        Phase p1 = getWirePhase(swap.getTarget(1));
        wirePhase[op->getResult(0)] = p1;
        wirePhase[op->getResult(1)] = p0;
      }
    }
    return store.takePlan();
  }

public:
  void runOnOperation() override {
    auto func = dyn_cast<func::FuncOp>(getOperation());
    if (!func)
      return;
    if (func->hasAttr(cudaq::runtime::disableQuantumOpts))
      return;

    DominanceInfo domInfo(func);

    SmallVector<cudaq::quake::XOp> cnots;
    func.walk([&](cudaq::quake::XOp xop) {
      if (wire::isControlledOp(xop))
        cnots.push_back(xop);
    });

    DenseSet<Operation *> processedOps;
    SmallVector<FoldPlan> plans;
    for (auto xop : cnots) {
      if (processedOps.count(xop))
        continue;
      wire::Subcircuit subcircuit(xop, processedOps);
      if (subcircuit.getNumOps() < minimumBlockLength ||
          subcircuit.getRotationWeight() < minimumrzWeight) {
        LLVM_DEBUG(llvm::dbgs() << "Subcircuit below threshold, skipping!\n");
        continue;
      }
      plans.push_back(planWirePhaseFolding(&subcircuit, domInfo));
    }

    // Dominance queries and operation ordering run only on the original IR.
    // Applying all plans afterward prevents rewrites in one subcircuit from
    // forcing order-index maintenance while another subcircuit is analyzed.
    for (auto &plan : plans)
      applyFoldPlan(plan);
  }
};
} // namespace
