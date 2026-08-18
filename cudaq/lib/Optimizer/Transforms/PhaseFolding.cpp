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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/Passes.h"

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

class PhaseStorage {
  DenseMap<PhaseKey, cudaq::quake::OperatorInterface, PhaseKeyInfo> phaseToRot;
  size_t numCombined = 0;

  // Returns the angle of a named Z-axis gate as a multiple of pi/4 (mod 8),
  // or nullopt for quake.rz (angle not statically known as a named gate).
  static std::optional<int>
  getQuarterPiUnits(cudaq::quake::OperatorInterface rot) {
    auto *op = rot.getOperation();
    if (isa<cudaq::quake::ZOp>(op))
      return 4;
    if (isa<cudaq::quake::SOp>(op))
      return rot.isAdj() ? 6 : 2;
    if (isa<cudaq::quake::TOp>(op))
      return rot.isAdj() ? 7 : 1;
    return std::nullopt;
  }

  // Returns the angle of a Z-axis rotation as an MLIR Value, creating a float
  // constant for named gates (S/T/Z) or reusing the existing operand for Rz.
  static Value getRotAngleValue(OpBuilder &builder,
                                cudaq::quake::OperatorInterface rot) {
    auto *op = rot.getOperation();
    if (isa<cudaq::quake::RzOp>(op))
      return op->getOperand(0);
    double angle;
    if (isa<cudaq::quake::ZOp>(op))
      angle = M_PI;
    else if (isa<cudaq::quake::SOp>(op))
      angle = rot.isAdj() ? -M_PI_2 : M_PI_2;
    else
      angle = rot.isAdj() ? -M_PI_4 : M_PI_4;
    return cudaq::opt::factory::createF64Constant(op->getLoc(), builder, angle);
  }

  // Combine rot1 (stored) and rot2 (new, the surviving position).
  // rot2's wire input (= rot1's output) is used for the new op.
  // rot1 is bypassed (its output replaced by its own input), then erased.
  // Returns the new combined op, or nullptr if they cancel to identity.
  Operation *combineRotations(cudaq::quake::OperatorInterface rot1,
                              cudaq::quake::OperatorInterface rot2) {
    auto *op1 = rot1.getOperation();
    auto *op2 = rot2.getOperation();
    OpBuilder builder(op2);
    auto loc = op2->getLoc();
    auto *ctx = op2->getContext();
    auto wireTy = cudaq::quake::WireType::get(ctx);
    Value wireIn = rot2.getTarget(0); // rot1's result (B)
    Value prevIn = rot1.getTarget(0); // rot1's input (A)
    numCombined++;

    auto finalize = [&](Operation *newOp) -> Operation * {
      op2->getResult(0).replaceAllUsesWith(newOp ? newOp->getResult(0)
                                                 : wireIn);
      op2->erase();
      op1->getResult(0).replaceAllUsesWith(prevIn);
      op1->erase();
      return newOp;
    };

    // If both are named gates (S/T/Z), combine via exact integer arithmetic
    // on quarter-pi units (0..7 mod 8) — no floating-point comparison.
    auto u1 = getQuarterPiUnits(rot1);
    auto u2 = getQuarterPiUnits(rot2);
    if (u1 && u2) {
      int combined = (*u1 + *u2) & 7;
      switch (combined) {
      case 0: // 0 = identity
        return finalize(nullptr);
      case 1: // π/4 = T
        return finalize(cudaq::quake::TOp::create(
            builder, loc, TypeRange{wireTy}, false, ValueRange{}, ValueRange{},
            ValueRange{wireIn}, {}));
      case 2: // π/2 = S
        return finalize(cudaq::quake::SOp::create(
            builder, loc, TypeRange{wireTy}, false, ValueRange{}, ValueRange{},
            ValueRange{wireIn}, {}));
      case 3: { // 3π/4 = S then T
        Value w = cudaq::quake::SOp::create(builder, loc, TypeRange{wireTy},
                                            false, ValueRange{}, ValueRange{},
                                            ValueRange{wireIn}, {})
                      ->getResult(0);
        return finalize(cudaq::quake::TOp::create(
            builder, loc, TypeRange{wireTy}, false, ValueRange{}, ValueRange{},
            ValueRange{w}, {}));
      }
      case 4: // π = Z
        return finalize(cudaq::quake::ZOp::create(
            builder, loc, TypeRange{wireTy}, false, ValueRange{}, ValueRange{},
            ValueRange{wireIn}, {}));
      case 5: { // 5π/4 = Z then T
        Value w = cudaq::quake::ZOp::create(builder, loc, TypeRange{wireTy},
                                            false, ValueRange{}, ValueRange{},
                                            ValueRange{wireIn}, {})
                      ->getResult(0);
        return finalize(cudaq::quake::TOp::create(
            builder, loc, TypeRange{wireTy}, false, ValueRange{}, ValueRange{},
            ValueRange{w}, {}));
      }
      case 6: // 3π/2 = S†
        return finalize(cudaq::quake::SOp::create(
            builder, loc, TypeRange{wireTy}, true, ValueRange{}, ValueRange{},
            ValueRange{wireIn}, {}));
      case 7: // 7π/4 = T†
        return finalize(cudaq::quake::TOp::create(
            builder, loc, TypeRange{wireTy}, true, ValueRange{}, ValueRange{},
            ValueRange{wireIn}, {}));
      }
    }

    // Rz + anything, or named-gate combo not landing on a named gate: addf
    Value angle1 = getRotAngleValue(builder, rot1);
    Value angle2 = getRotAngleValue(builder, rot2);
    auto sumAngle = arith::AddFOp::create(builder, loc, angle1, angle2);
    return finalize(
        cudaq::quake::RzOp::create(builder, loc, TypeRange{wireTy}, false,
                                   ValueRange{sumAngle.getResult()},
                                   ValueRange{}, ValueRange{wireIn}, {}));
  }

public:
  // Returns the stored or combined op (nullptr if identity cancellation).
  Operation *addOrCombineRotationForPhase(cudaq::quake::OperatorInterface rot,
                                          Phase phase) {
    auto key = phase.toKey();
    auto it = phaseToRot.find(key);
    if (it != phaseToRot.end()) {
      auto *newOp = combineRotations(it->second, rot);
      if (newOp)
        it->second = cast<cudaq::quake::OperatorInterface>(newOp);
      else
        phaseToRot.erase(it);
      return newOp;
    }
    phaseToRot[key] = rot;
    return rot.getOperation();
  }

  size_t getNumCombined() { return numCombined; }
};

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

static bool isTerminationPoint(Operation *op) {
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
  Operation *start;
  // TODO: these three are really intermediate state for constructing the
  // subcircuit; would be nice to turn them into local arguments instead
  SetVector<Value> termination_points;
  SetVector<Value> anchor_points;
  SetVector<Value> seen;

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
    Operation *op = v.getUses().begin().getUser();
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
    start = cnot;
    for (auto w : termination_points)
      if (isAfterTerminationPoint(w))
        initial_wires.insert(w);
      else
        terminal_wires.insert(w);
  }

  SetVector<Value> getInitialWires() { return initial_wires; }
  bool isInSubcircuit(Operation *op) { return ops.contains(op); }
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
    SmallVector<Operation *> ordered(ops.begin(), ops.end());
    sort(ordered,
         [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    return ordered;
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
/// When two Z-axis rotations share the same phase their combined angle is
/// checked (mod 2*pi) against named gate thresholds (epsilon = 1e-9):
///   0        -> identity; both ops removed
///   +/-pi/4  -> quake.t / quake.t<adj>
///   +/-pi/2  -> quake.s / quake.s<adj>
///   pi       -> quake.z
///   other    -> quake.rz with the raw summed constant, or an arith.addf of
///               the two angle values when either input angle is non-constant.
class PhaseFoldingPass
    : public cudaq::opt::impl::PhaseFoldingBase<PhaseFoldingPass> {
  using PhaseFoldingBase::PhaseFoldingBase;

  void doWirePhaseFolding(wire::Subcircuit *subcircuit) {
    DenseMap<Value, Phase> wirePhase;
    SmallVector<std::unique_ptr<PhaseVariable>> vars;
    PhaseStorage store;
    size_t i = 0;

    for (auto w : subcircuit->getInitialWires()) {
      auto &var = vars.emplace_back(std::make_unique<PhaseVariable>(i++));
      wirePhase[w] = Phase(var.get());
    }

    for (auto *op : subcircuit->getOrderedOps()) {
      if (wire::isControlledOp(op)) {
        auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
        Phase ctrlPhase = wirePhase[opi.getControls().front()];
        Phase tgtPhase = wirePhase[opi.getTarget(0)];
        wirePhase[op->getResult(0)] = ctrlPhase;
        wirePhase[op->getResult(1)] = Phase::sum(ctrlPhase, tgtPhase);
      } else if (isa<cudaq::quake::XOp>(op)) {
        // AXIS-SPECIFIC: Would want to handle y and z gates here too
        auto opi = dyn_cast<cudaq::quake::OperatorInterface>(op);
        wirePhase[op->getResult(0)] =
            Phase::invert(wirePhase[opi.getTarget(0)]);
      } else if (isa<RAW_Z_AXIS_ROTATIONS>(op)) {
        auto opi = cast<cudaq::quake::OperatorInterface>(op);
        Phase p = wirePhase[opi.getTarget(0)];
        auto *newOp = store.addOrCombineRotationForPhase(opi, p);
        if (newOp)
          wirePhase[newOp->getResult(0)] = p;
      } else if (auto swap = dyn_cast<cudaq::quake::SwapOp>(op)) {
        Phase p0 = wirePhase[swap.getTarget(0)];
        Phase p1 = wirePhase[swap.getTarget(1)];
        wirePhase[op->getResult(0)] = p1;
        wirePhase[op->getResult(1)] = p0;
      }
    }
  }

public:
  void runOnOperation() override {
    auto func = dyn_cast<func::FuncOp>(getOperation());
    if (!func)
      return;
    if (func->hasAttr(cudaq::runtime::disableQuantumOpts))
      return;

    // Collect CNOTs first to avoid iterator invalidation: combineRotations
    // erases Rz ops which may be the stored next-pointer in the walk iterator.
    SmallVector<cudaq::quake::XOp> cnots;
    func.walk([&](cudaq::quake::XOp xop) {
      if (wire::isControlledOp(xop))
        cnots.push_back(xop);
    });

    // Subcircuits are built and folded one at a time so that Rz ops erased
    // during folding are gone from the IR before the next subcircuit is built.
    // TODO: Parallel folding would require tracking which Rz ops have been
    // erased so that subcircuits built concurrently can skip stale references.
    DenseSet<Operation *> processedOps;
    for (auto xop : cnots) {
      if (processedOps.count(xop))
        continue;
      wire::Subcircuit subcircuit(xop, processedOps);
      if (subcircuit.getNumOps() < minimumBlockLength ||
          subcircuit.getRotationWeight() < minimumrzWeight) {
        LLVM_DEBUG(llvm::dbgs() << "Subcircuit below threshold, skipping!\n");
        continue;
      }
      doWirePhaseFolding(&subcircuit);
    }
  }
};
} // namespace
