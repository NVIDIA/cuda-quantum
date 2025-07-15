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
#define GEN_PASS_DEF_PHASEPOLYNOMIALPREPROCESS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "phase-polynomial-preprocess"

using namespace mlir;

namespace {
class PhasePolynomialPreprocessPass
    : public cudaq::opt::impl::PhasePolynomialPreprocessBase<
          PhasePolynomialPreprocessPass> {
  using PhasePolynomialPreprocessBase::PhasePolynomialPreprocessBase;

  // TODO: I think this could potentially be generalized nicely
  class WireStepper {
    Value old_wire;
    Value new_wire;
    Subcircuit *subcircuit;
    bool stopped = false;

  public:
    WireStepper(Subcircuit *circuit, Value initial, Value arg) {
      subcircuit = circuit;
      old_wire = initial;
      new_wire = arg;
    }

    bool isStopped() {
      return stopped ||
             (stopped = subcircuit->getTerminalWires().contains(old_wire));
    }

    Value getNewWire() { return new_wire; }

    Value getOldWire() { return old_wire; }

    void step(DenseMap<Operation *, Operation *> &cloned, OpBuilder &builder,
              std::function<Value(Value)> addFuncArg) {
      // TODO: Something more elegant here would be nice.
      // The problem is that the old_wire may have two uses,
      // one in the original block, and one in the new function by the cloned
      // op. We want to ignore the cloned op here.
      Operation *op = nullptr;
      size_t opnum = 0;
      for (auto &use : old_wire.getUses()) {
        if (use.getOwner()->hasAttr("clone"))
          continue;
        op = use.getOwner();
        opnum = use.getOperandNumber();
      }

      assert(op);

      if (cloned.count(op) == 1) {
        cloned[op]->setOperand(opnum, new_wire);
        assert(old_wire.hasOneUse());
        old_wire = getNextResult(old_wire);
        new_wire = getNextResult(new_wire);
        return;
      }

      // Make sure all dependencies have been cloned
      for (auto dependency : op->getOperands()) {
        if (!isa<quake::WireType>(dependency.getType()))
          continue;
        auto dop = dependency.getDefiningOp();
        if (cloned.count(dop) != 1 &&
            !subcircuit->getInitialWires().contains(dependency))
          return;
      }

      auto clone = builder.clone(*op);
      clone->setOperand(opnum, new_wire);
      clone->setAttr("clone", builder.getUnitAttr());

      // Make classical values arguments to the function,
      // to allow non-constant rotation angles
      builder.setInsertionPointToStart(clone->getBlock());
      for (size_t i = 0; i < clone->getNumOperands(); i++) {
        auto dependency = clone->getOperand(i);
        if (!isa<quake::WireType>(dependency.getType())) {
          auto new_arg = addFuncArg(dependency);
          clone->setOperand(i, new_arg);
        }
      }
      builder.setInsertionPointAfter(clone);

      cloned[op] = clone;
      assert(old_wire.hasOneUse());
      old_wire = getNextResult(old_wire);
      new_wire = getNextResult(new_wire);
    }
  };

  void removeOld(Subcircuit &subcircuit,
                 SmallVector<Operation *> &removal_order, Operation *next) {
    if (!subcircuit.getOps().contains(next) ||
        std::find(removal_order.begin(), removal_order.end(), next) !=
            removal_order.end())
      return;

    for (auto result : next->getResults())
      for (auto *user : result.getUsers())
        removeOld(subcircuit, removal_order, user);

    removal_order.push_back(next);
  }

  void shiftAfter(Operation *pivot, Operation *to_shift) {
    if (pivot->isBeforeInBlock(to_shift))
      return;
    to_shift->moveAfter(pivot);
    for (auto user : to_shift->getUsers())
      shiftAfter(to_shift, user);
  }

  void moveToFunc(Subcircuit *subcircuit, size_t subcircuit_num) {
    auto module = getOperation();
    SmallVector<Type> types(subcircuit->getInitialWires().size(),
                            quake::WireType::get(module.getContext()));
    auto name = std::string("subcircuit") + std::to_string(subcircuit_num);
    auto fun = cudaq::opt::factory::createFunction(name, types, types, module);
    fun.setPrivate();
    auto entry = fun.addEntryBlock();
    OpBuilder builder(fun);
    fun.getOperation()->setAttr("subcircuit", builder.getUnitAttr());
    fun.getOperation()->setAttr(
        "num_cnots", builder.getUI32IntegerAttr(subcircuit->numCNots()));

    DenseMap<Operation *, Operation *> cloned;

    // Need to keep ordering to match returns with arguments
    SmallVector<Value> args;
    SmallVector<WireStepper *> steppers;
    for (auto wire : subcircuit->getInitialWires()) {
      args.push_back(wire);
      steppers.push_back(
          new WireStepper(subcircuit, wire, fun.getArgument(steppers.size())));
    }

    auto add_arg_fun = [&](Value v) {
      auto idx = args.size();
      args.push_back(v);
      fun.insertArgument(idx, v.getType(), {}, v.getDefiningOp()->getLoc());
      return fun.getArgument(idx);
    };

    builder.setInsertionPointToStart(entry);
    while (true) {
      auto stepped = false;
      for (auto stepper : steppers) {
        if (stepper->isStopped())
          continue;
        stepped = true;
        stepper->step(cloned, builder, add_arg_fun);
      }

      if (!stepped)
        break;
    }

    SmallVector<Value> new_wires;
    for (size_t i = 0; i < steppers.size(); i++)
      new_wires.push_back(steppers[i]->getNewWire());

    builder.create<cudaq::cc::ReturnOp>(fun.getLoc(), new_wires);

    auto cnot = subcircuit->getStart();
    auto latest = cnot;
    for (auto arg : args) {
      if (!isa<quake::WireType>(arg.getType()))
        continue;
      auto dop = arg.getDefiningOp();
      if (dop && latest->isBeforeInBlock(dop))
        latest = dop;
    }
    builder.setInsertionPointAfter(latest);

    fun.walk([&](Operation *op) {
      op->removeAttr("clone");
      op->removeAttr("processed");
    });

    auto call = builder.create<func::CallOp>(cnot->getLoc(), types,
                                             fun.getSymNameAttr(), args);
    for (size_t i = 0; i < steppers.size(); i++)
      steppers[i]->getOldWire().replaceAllUsesWith(call.getResult(i));

    for (auto user : call->getUsers())
      shiftAfter(call, user);

    for (auto stepper : steppers)
      delete stepper;
  }

public:
  void runOnOperation() override {
    auto module = getOperation();
    size_t i = 0;
    SetVector<Subcircuit *> subcircuits;
    for (auto &op : module) {
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        func.walk([&](quake::XOp op) {
          if (!::isControlledOp(op) || ::processed(op))
            return;

          auto *subcircuit = new Subcircuit(op);
          moveToFunc(subcircuit, i++);
          // Add the subcircuit to erase from the function after we
          // finish walking it, as we don't want to erase ops from a
          // function we are currently walking
          subcircuits.insert(subcircuit);
        });
      }
    }

    for (auto *subcircuit : subcircuits) {
      for (auto op : subcircuit->getOps()) {
        op->dropAllUses();
        op->erase();
      }
      delete subcircuit;
    }
  }
};
} // namespace

static void createPhasePolynomialOptPipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  // opt::LoopUnrollOptions luo;
  // luo.threshold = 2048;
  // pm.addNestedPass<func::FuncOp>(opt::createLoopUnroll(luo));
  // pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createFactorQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(cudaq::opt::createPhasePolynomialPreprocess());
  pm.addNestedPass<func::FuncOp>(
      cudaq::opt::createPhasePolynomialRotationMerging());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeSimplify());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  cudaq::opt::addAggressiveEarlyInlining(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createRegToMem());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
}

void cudaq::opt::registerPhasePolynomialOptimizationPipeline() {
  PassPipelineRegistration<>(
      "phase-polynomial-opt-pipeline",
      "Apply phase polynomial based rotation merging.",
      [](OpPassManager &pm) { createPhasePolynomialOptPipeline(pm); });
}
