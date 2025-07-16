/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "Subcircuit.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_PHASEPOLYNOMIALROTATIONMERGING
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "phase-polynomial-rotation-merging"

using namespace mlir;

namespace {
class PhasePolynomialRotationMergingPass
    : public cudaq::opt::impl::PhasePolynomialRotationMergingBase<
          PhasePolynomialRotationMergingPass> {
  using PhasePolynomialRotationMergingBase::PhasePolynomialRotationMergingBase;

  // struct PhaseVariable {
  // public:
  //   size_t idx;
  //   // TODO: do we really need the initial_wire here?
  //   // I think it's just useful for debugging
  //   Value initial_wire;
  //   PhaseVariable(size_t index, Value wire) : idx(index), initial_wire(wire)
  //   {}

  //   bool operator==(PhaseVariable other) { return idx == other.idx; }
  // };

  // class Phase {
  //   SetVector<PhaseVariable *> vars;
  //   bool isInverted;

  // public:
  //   Phase() : isInverted(false) {}

  //   Phase(PhaseVariable *var) : isInverted(false) { vars.insert(var); }

  //   bool operator==(Phase other) {
  //     for (auto var : vars)
  //       if (!other.vars.contains(var))
  //         return false;
  //     for (auto var : other.vars)
  //       if (!vars.contains(var))
  //         return false;
  //     return isInverted == other.isInverted;
  //   }

  //   static Phase combine(Phase &p1, Phase &p2) {
  //     Phase new_phase = Phase();
  //     for (auto var : p1.vars)
  //       new_phase.vars.insert(var);
  //     for (auto var : p2.vars)
  //       if (new_phase.vars.contains(var))
  //         new_phase.vars.remove(var);
  //       else
  //         new_phase.vars.insert(var);
  //     new_phase.isInverted = (p1.isInverted != p2.isInverted);
  //     return new_phase;
  //   }

  //   static Phase invert(Phase &p1) {
  //     auto new_phase = Phase();
  //     for (auto var : p1.vars)
  //       new_phase.vars.insert(var);
  //     new_phase.isInverted = !p1.isInverted;
  //     return new_phase;
  //   }

  //   void dump() {
  //     llvm::outs() << "Phase: ";
  //     if (isInverted)
  //       llvm::outs() << "!";
  //     llvm::outs() << "{";
  //     auto first = true;
  //     for (auto var : vars) {
  //       if (!first)
  //         llvm::outs() << " ^ ";
  //       llvm::outs() << var->idx;
  //       first = false;
  //     }
  //     llvm::outs() << "}\n";
  //   }

  //   std::optional<int64_t> getIntRepresentation() {
  //     int64_t sum = 0;
  //     for (auto var : vars) {
  //       if (var->idx > sizeof(int64_t) - 1)
  //         return std::nullopt;
  //       sum += 1 << var->idx;
  //     }
  //   }
  // };

  // class PhaseStorage {
  //   SmallVector<Phase> phases;
  //   SmallVector<quake::RzOp> rotations;

  //   void combineRotations(size_t prev_idx, quake::RzOp rzop) {
  //     auto old_rzop = rotations[prev_idx];
  //     auto builder = OpBuilder(old_rzop);
  //     auto rot_arg1 = old_rzop.getOperand(0);
  //     auto rot_arg2 = rzop.getOperand(0);
  //     auto new_rot_arg =
  //         builder.create<arith::AddFOp>(old_rzop.getLoc(), rot_arg1,
  //         rot_arg2);
  //     old_rzop->setOperand(0, new_rot_arg.getResult());
  //     rzop.getResult(0).replaceAllUsesWith(rzop.getOperand(1));
  //     rzop.erase();
  //   }

  // public:
  //   /// @brief registers a new rotation op for the given phase
  //   /// @returns true if the rotation was combined, false otherwise
  //   bool addOrCombineRotationForPhase(quake::RzOp op, Phase phase) {
  //     for (size_t i = 0; i < phases.size(); i++)
  //       if (phases[i] == phase) {
  //         combineRotations(i, op);
  //         return true;
  //       }

  //     phases.push_back(phase);
  //     rotations.push_back(op);
  //     return false;
  //   }
  // };

  // class PhaseStepper {
  //   Value wire;
  //   Subcircuit *subcircuit;
  //   PhaseStorage *store;
  //   Phase current_phase;

  // public:
  //   class StepperContainer {
  //     SmallVector<PhaseStepper *> steppers;
  //     PhaseStorage *store;
  //     SmallVector<PhaseVariable *> vars;

  //     PhaseStepper *getStepperForValue(Value v) {
  //       for (auto *stepper : steppers)
  //         if (stepper->wire == v)
  //           return stepper;
  //       return nullptr;
  //     }

  //   public:
  //     // Caller is responsible for cleaning up circuit
  //     StepperContainer(Subcircuit *circuit) {
  //       store = new PhaseStorage();
  //       size_t i = 0;
  //       for (auto wire : circuit->getInitialWires()) {
  //         auto *new_var = new PhaseVariable(i++, wire);
  //         // StepperContainer is responsible for cleaning up PhaseSteppers
  //         steppers.push_back(new PhaseStepper(circuit, store, wire,
  //         new_var));
  //         // StepperContainer is responsible for cleaning up PhaseVariables
  //         vars.push_back(new_var);
  //       }
  //     }

  //     ~StepperContainer() {
  //       delete store;
  //       for (auto stepper : steppers)
  //         delete stepper;
  //       for (auto var : vars)
  //         delete var;
  //     }

  //     static bool isPhaseInvariant(Block *b) {
  //       llvm::outs() << "Inspecting ";
  //       b->dump();

  //       auto subcircuit = Subcircuit::constructFromBlock(b);

  //       if (!subcircuit)
  //         return false;

  //       llvm::outs() << "Valid subcircuit!\n";

  //       auto stepper = StepperContainer(subcircuit);

  //       while (!stepper.isStopped())
  //         stepper.stepAll();

  //       for (size_t i = 0; i < stepper.steppers.size(); i++)
  //         if (stepper.steppers[i]->current_phase != stepper.vars[i])
  //           return false;

  //       return true;
  //     }

  //     bool isStopped() {
  //       for (auto *stepper : steppers)
  //         if (!stepper->isStopped())
  //           return false;
  //       return true;
  //     }

  //     void stepAll() {
  //       if (isStopped())
  //         return;
  //       for (auto *stepper : steppers)
  //         stepper->step(this);
  //     }

  //     std::optional<Phase> maybeGetControlPhase(quake::OperatorInterface opi)
  //     {
  //       assert(isControlledOp(opi.getOperation()));
  //       auto control = opi.getControls().front();
  //       auto *stepper = getStepperForValue(control);
  //       if (stepper)
  //         return stepper->current_phase;
  //       return std::nullopt;
  //     }

  //     /// @brief handles a swap between two wires, swapping their phases
  //     /// @returns `true` if the swap has been handled and stepping can
  //     /// continue, `false` otherwise
  //     bool maybeHandleSwap(quake::SwapOp swap) {
  //       auto wire0 = swap.getTarget(0);
  //       auto wire1 = swap.getTarget(1);
  //       if (wireVisited(wire0) || wireVisited(wire1))
  //         return true;

  //       auto stepper0 = getStepperForValue(wire0);
  //       auto stepper1 = getStepperForValue(wire1);
  //       if (!stepper0 || !stepper1)
  //         return false;

  //       auto tmp = stepper0->current_phase;
  //       stepper0->current_phase = stepper1->current_phase;
  //       stepper1->current_phase = tmp;
  //       return true;
  //     }

  //     bool wireVisited(Value wire) {
  //       auto next_result = getNextResult(wire);
  //       // Wait until target wire stepper steps to ensure
  //       // control phase is available
  //       return !!getStepperForValue(next_result);
  //     }
  //   };

  //   PhaseStepper(Subcircuit *circuit, PhaseStorage *store, Value initial,
  //                PhaseVariable *var) {
  //     subcircuit = circuit;
  //     this->store = store;
  //     wire = initial;
  //     current_phase = Phase(var);
  //   }

  //   bool isStopped() {
  //     Operation *op = *wire.getUsers().begin();
  //     assert(op);
  //     // Have to have explicit check for termination point
  //     // because rotation merging may have removed old termination
  //     // point
  //     return isTerminationPoint(op);
  //   }

  //   void step(StepperContainer *container) {
  //     if (isStopped())
  //       return;
  //     assert(wire.hasOneUse());

  //     Operation *op = *wire.getUsers().begin();
  //     assert(op);
  //     auto opi = dyn_cast<quake::OperatorInterface>(op);
  //     assert(opi);

  //     if (isControlledOp(op)) {
  //       // Controlled not, and we are the target, so update phase
  //       if (opi.getTarget(0) == wire) {
  //         auto phase_opt = container->maybeGetControlPhase(opi);
  //         // Wait until we have the phase for the other wire
  //         if (!phase_opt.has_value())
  //           return;
  //         current_phase = Phase::combine(current_phase, phase_opt.value());
  //       } else {
  //         // Wait until the target has visited the operation so it can
  //         // access our phase (the control phase)
  //         if (!container->wireVisited(opi.getTarget(0)))
  //           return;
  //       }
  //     } else if (isa<quake::XOp>(op) && opi.getControls().size() == 0) {
  //       // Simple not, invert phase
  //       // AXIS-SPECIFIC: Would want to handle y and z gates here too
  //       current_phase = Phase::invert(current_phase);
  //     } else if (auto rzop = dyn_cast<quake::RzOp>(op)) {
  //       if (store->addOrCombineRotationForPhase(rzop, current_phase))
  //         return;
  //     } else if (auto swap = dyn_cast<quake::SwapOp>(op)) {
  //       if (!container->maybeHandleSwap(swap))
  //         return;
  //     }

  //     wire = getNextResult(wire);
  //   }
  // };

public:
  void runOnOperation() override {
    auto func = getOperation();

    if (!func.getOperation()->hasAttr("subcircuit"))
      return;

    // auto subcircuit = Subcircuit::constructFromFunc(func);
    // if (!subcircuit)
    //   return;
    // auto container = PhaseStepper::StepperContainer(subcircuit);

    // while (!container.isStopped())
    //   container.stepAll();
    // delete subcircuit;

    // func.walk([&](Operation *op){
    //   if (auto loop = dyn_cast<cudaq::cc::LoopOp>(op))
    //     if
    //     (PhaseStepper::StepperContainer::isPhaseInvariant(&loop.getLoopBody().front()))
    //     {
    //       llvm::outs() << "Phase invariant!: ";
    //       loop.dump();
    //     }
    // });
  }
};
} // namespace
