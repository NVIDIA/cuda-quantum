/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PhaseFolding.h"
#include "PassDetails.h"
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
class PhaseFoldingPass
    : public cudaq::opt::impl::PhaseFoldingBase<PhaseFoldingPass> {
  using PhaseFoldingBase::PhaseFoldingBase;

  // Perform the phase folding optimization over a subcircuit
  void doPhaseFolding(Subcircuit *subcircuit) {
    SmallVector<PhaseVariable *> phase_vars;
    SmallVector<Phase> current_phases;
    PhaseStorage store;
    size_t i = 0;
    // Initialize the phases and phase variables for each qubit in the circuit,
    // the initial phase contains only the phase variable for that qubit
    for (auto ref : subcircuit->getRefs()) {
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
      if (::isCNOT(op)) {
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
  void runOnOperation() override {
    auto func = getOperation();
    // Get the netlist represention for the qubits in the function,
    // this will walk the whole function once
    auto nl = Netlist(func);
    SmallVector<Subcircuit *> subcircuits;

    func.walk([&](quake::XOp op) {
      // AXIS-SPECIFIC: controlled not only
      if (!::isCNOT(op) || ::processed(op))
        return;

      if (!isSupportedValue(op.getOperand(0)) ||
          !isSupportedValue(op.getOperand(1)))
        return;

      // Build a subcircuit from the CNot
      auto subcircuit = new Subcircuit(op, &nl);
      subcircuits.push_back(subcircuit);
    });

    // Performing the actual optimization over subcircuits after collecting them
    // A) allows for eventually parallelizing the optimization, and
    // B) avoids rewriting the AST as it is being walked above, causing an
    // error. This does introduce a requirement that each operation belongs to
    // at most one subcircuit.
    for (auto subcircuit : subcircuits) {
      doPhaseFolding(subcircuit);
      // Clean up
      delete subcircuit;
    }
  }
};
} // namespace
