/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "Subcircuit.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_PHASEFOLDING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "phase-folding"

using namespace mlir;

namespace {
struct PhaseVariable {
public:
  size_t idx;
  // Q: do we really need the initial_wire here? I think it's just useful for
  // debugging
  Value initial_wire;
  PhaseVariable(size_t index, Value wire) : idx(index), initial_wire(wire) {}

  bool operator==(PhaseVariable other) { return idx == other.idx; }
};

/// A `Phase` is an exclusive sum of all of the `PhaseVariable`s involved in the
/// current state of a qubit, as well as 1, representing inversion from a Not
/// gate. The simplest Phase contains exactly the `PhaseVariable` representing
/// the initial state of a qubit in a subcircuit. There are two operations on
/// `Phase`s to generate new `Phase`s: `Phase::sum` sums two Phases,
/// corresponding to the effect of a CNot on the target qubit. `Phase::invert`
/// inverts a Phase, corresponding to the effect of a Not a qubit.
///
/// Generally, a Phase is an exclusive sum of products.
/// However, our Phases are currently only exclusive sums;
/// products are not currently supported.
///
/// Any two rotations on qubits with equal Phases can be merged into one
/// rotation.
class Phase {
  SetVector<PhaseVariable *> vars;
  bool isInverted;

public:
  Phase() : isInverted(false) {}

  Phase(PhaseVariable *var) : isInverted(false) { vars.insert(var); }

  /// @brief Two phases are equal if they contain exactly the same vars
  /// and have the same inversion flag.
  bool operator==(Phase other) {
    for (auto var : vars)
      if (!other.vars.contains(var))
        return false;
    for (auto var : other.vars)
      if (!vars.contains(var))
        return false;
    return isInverted == other.isInverted;
  }

  /// @brief Returns a new phase equal to the sum of `p1` and `p2`
  /// @returns A new phase containing the exclusive or of `p1` and `p2`
  static Phase sum(Phase &p1, Phase &p2) {
    Phase new_phase = Phase();
    for (auto var : p1.vars)
      new_phase.vars.insert(var);
    for (auto var : p2.vars)
      if (new_phase.vars.contains(var))
        new_phase.vars.remove(var);
      else
        new_phase.vars.insert(var);
    new_phase.isInverted = (p1.isInverted != p2.isInverted);
    return new_phase;
  }

  /// @brief Returns a new phase equal to `p1` with the opposite inversion flag
  static Phase invert(Phase &p1) {
    auto new_phase = Phase();
    for (auto var : p1.vars)
      new_phase.vars.insert(var);
    new_phase.isInverted = !p1.isInverted;
    return new_phase;
  }

  void dump() {
    llvm::outs() << "Phase: ";
    if (isInverted)
      llvm::outs() << "!";
    llvm::outs() << "{";
    auto first = true;
    for (auto var : vars) {
      if (!first)
        llvm::outs() << " ^ ";
      llvm::outs() << var->idx;
      first = false;
    }
    llvm::outs() << "}\n";
  }

  std::optional<int64_t> getIntRepresentation() {
    int64_t sum = 0;
    for (auto var : vars) {
      if (var->idx > sizeof(int64_t) - 1)
        return std::nullopt;
      sum += 1 << var->idx;
    }
  }
};

class PhaseStorage {
  SmallVector<Phase> phases;
  SmallVector<quake::RzOp> rotations;
  size_t numCombined = 0;

  // Merges the rotation at prev_idx with rzop by adding their
  // rotation angles and overwriting rzop's angle with the new
  // angle. The old rotation is erased. We keep the latter rotation
  // to ensure that dynamic rotation angles (e.g., dependent on
  // measurement results) are indeed available, as earlier angles
  // will always be available later, but not vice-versa.
  void combineRotations(size_t prev_idx, quake::RzOp rzop) {
    auto old_rzop = rotations[prev_idx];
    auto rot_arg1 = old_rzop.getOperand(0);
    auto rot_arg2 = rzop.getOperand(0);
    auto builder = OpBuilder(rzop);
    auto new_rot_arg =
        builder.create<arith::AddFOp>(rzop.getLoc(), rot_arg1, rot_arg2);
    rzop->setOperand(0, new_rot_arg.getResult());
    old_rzop.erase();
    rotations[prev_idx] = rzop;
    numCombined++;
  }

public:
  /// @brief registers a new rotation op for the given phase
  /// @returns true if the rotation was combined, false otherwise
  bool addOrCombineRotationForPhase(quake::RzOp op, Phase phase) {
    for (size_t i = 0; i < phases.size(); i++)
      if (phases[i] == phase) {
        combineRotations(i, op);
        return true;
      }

    phases.push_back(phase);
    rotations.push_back(op);
    return false;
  }

  size_t getNumCombined() { return numCombined; }
};

class PhaseFoldingPass
    : public cudaq::opt::impl::PhaseFoldingBase<PhaseFoldingPass> {
  using PhaseFoldingBase::PhaseFoldingBase;

  // Perform the phase folding optimization over a subcircuit
  void doPhaseFolding(Subcircuit *subcircuit) {
    SmallVector<PhaseVariable *> phase_vars;
    SmallVector<Phase> current_phases;
    PhaseStorage store;
    size_t i = 0;
    // Initial the phases and phase variables for each qubit in the circuit, the
    // initial phase contains only the phase variable for that qubit
    for (auto ref : subcircuit->getRefs()) {
      if (ref.getType() != quake::RefType::get(getOperation().getContext()))
        break;
      auto phase_idx = i++;
      auto new_phase_var = new PhaseVariable(phase_idx, ref);
      auto defop = ref.getDefiningOp();
      assert(defop);
      auto builder = OpBuilder(defop);
      // Q: is using attributes actually better than using a map of some sort?
      // Does it matter?
      defop->setAttr("phaseidx", builder.getUI32IntegerAttr(phase_idx));
      phase_vars.push_back(new_phase_var);
      current_phases.push_back(Phase(new_phase_var));
    }

    // A helper function to look up the phase for a quake.ref
    auto getPhase = [&](Value ref) {
      auto defop = ref.getDefiningOp();
      auto idx = defop->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      return current_phases[idx];
    };

    // A helper function to set the phase for a quake.ref
    auto setPhase = [&](Value ref, Phase phase) {
      auto defop = ref.getDefiningOp();
      auto idx = defop->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      current_phases[idx] = phase;
    };

    // Process all ops in the subcircuit, tracking phases and greedily trying to
    // merge rotations
    for (auto op : subcircuit->getOrderedOps()) {
      if (::isControlledOp(op)) {
        // Controlled not, set new target phase to XOR of control and old target
        // phases
        auto control = op->getOperand(0);
        auto target = op->getOperand(1);
        auto control_phase = getPhase(control);
        auto target_phase = getPhase(target);
        auto new_target_phase = Phase::sum(target_phase, control_phase);
        setPhase(target, new_target_phase);
      } else if (isa<quake::XOp>(op)) {
        // Simple not, invert target phase
        // AXIS-SPECIFIC: Would want to handle y and z gates here too
        auto target = op->getOperand(0);
        auto target_phase = getPhase(target);
        auto new_target_phase = Phase::invert(target_phase);
        setPhase(target, new_target_phase);
      } else if (auto rzop = dyn_cast<quake::RzOp>(op)) {
        // Rotation, try to merge by looking up in store
        auto target = op->getOperand(1);
        auto target_phase = getPhase(target);
        store.addOrCombineRotationForPhase(rzop, target_phase);
      } else if (auto swap = dyn_cast<quake::SwapOp>(op)) {
        // Swap phases
        auto target1 = op->getOperand(0);
        auto target2 = op->getOperand(1);
        auto target1_phase = getPhase(target1);
        auto target2_phase = getPhase(target2);
        setPhase(target1, target2_phase);
        setPhase(target2, target1_phase);
      }
    }
  }

public:
  // Assumption: all quake operators interface operands are `!quake.ref`s,
  // either from `extract_ref` or `alloca`
  void runOnOperation() override {
    auto func = getOperation();
    auto nlc = NetlistContainer(func);
    SmallVector<Subcircuit *> subcircuits;
    func.walk([&](quake::XOp op) {
      // AXIS-SPECIFIC: controlled not only
      if (!::isControlledOp(op) || ::processed(op))
        return;

      auto subcircuit = new Subcircuit(op, &nlc);
      subcircuits.push_back(subcircuit);
    });

    // Performing the actual optimization over subcircuits after collecting them
    // A) allows for parallelization of the optimization, and
    // B) avoids rewriting the AST as it is being walked above.
    for (auto subcircuit : subcircuits) {
      doPhaseFolding(subcircuit);
      delete subcircuit;
    }
  }
};
} // namespace
