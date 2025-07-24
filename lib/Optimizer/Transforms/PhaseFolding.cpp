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
  // TODO: do we really need the initial_wire here?
  // I think it's just useful for debugging
  Value initial_wire;
  PhaseVariable(size_t index, Value wire) : idx(index), initial_wire(wire) {}

  bool operator==(PhaseVariable other) { return idx == other.idx; }
};

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

  static Phase combine(Phase &p1, Phase &p2) {
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

  void doPhaseFolding(Subcircuit *subcircuit) {
    SmallVector<PhaseVariable *> phase_vars;
    SmallVector<Phase> current_phases;
    PhaseStorage store;
    size_t i = 0;
    for (auto ref : subcircuit->getRefs()) {
      if (ref.getType() != quake::RefType::get(getOperation().getContext()))
        break;
      auto phase_idx = i++;
      auto new_phase_var = new PhaseVariable(phase_idx, ref);
      auto defop = ref.getDefiningOp();
      assert(defop);
      auto builder = OpBuilder(defop);
      defop->setAttr("phaseidx", builder.getUI32IntegerAttr(phase_idx));
      phase_vars.push_back(new_phase_var);
      current_phases.push_back(Phase(new_phase_var));
    }

    auto getPhase = [&](Value ref) {
      auto defop = ref.getDefiningOp();
      auto idx = defop->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      return current_phases[idx];
    };

    auto setPhase = [&](Value ref, Phase phase) {
      auto defop = ref.getDefiningOp();
      auto idx = defop->getAttrOfType<IntegerAttr>("phaseidx").getUInt();
      current_phases[idx] = phase;
    };

    for (auto op : subcircuit->getOrderedOps()) {
      if (::isControlledOp(op)) {
        auto control = op->getOperand(0);
        auto target = op->getOperand(1);
        auto control_phase = getPhase(control);
        auto target_phase = getPhase(target);
        auto new_target_phase = Phase::combine(target_phase, control_phase);
        setPhase(target, new_target_phase);
      } else if (isa<quake::XOp>(op)) {
        // Simple not, invert phase
        // AXIS-SPECIFIC: Would want to handle y and z gates here too
        auto target = op->getOperand(0);
        auto target_phase = getPhase(target);
        auto new_target_phase = Phase::invert(target_phase);
        setPhase(target, new_target_phase);
      } else if (auto rzop = dyn_cast<quake::RzOp>(op)) {
        auto target = op->getOperand(1);
        auto target_phase = getPhase(target);
        store.addOrCombineRotationForPhase(rzop, target_phase);
      } else if (auto swap = dyn_cast<quake::SwapOp>(op)) {
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
  // One assumption is that calls are fully inlined so no references are
  // function arguments
  void runOnOperation() override {
    auto func = getOperation();
    auto nlc = NetlistContainer(func);
    SmallVector<Subcircuit *> subcircuits;
    // AXIS-SPECIFIC: controlled not only
    func.walk([&](quake::XOp op) {
      if (!::isControlledOp(op) || ::processed(op))
        return;

      auto subcircuit = new Subcircuit(op, &nlc);
      subcircuits.push_back(subcircuit);
    });

    for (auto subcircuit : subcircuits) {
      doPhaseFolding(subcircuit);
      delete subcircuit;
    }
  }
};
} // namespace
