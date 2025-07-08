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

  public:
    WireStepper(Subcircuit *circuit, Value initial, Value arg) {
      subcircuit = circuit;
      old_wire = initial;
      new_wire = arg;
    }

    bool isStopped() {
      return subcircuit->getTerminalWires().contains(old_wire);
    }

    Value getNewWire() { return new_wire; }

    Value getOldWire() { return old_wire; }

    void step(DenseMap<Operation *, Operation *> &cloned, OpBuilder &builder,
              std::function<Value(Value)> addFuncArg) {
      if (isStopped())
        return;

      // TODO: Something more elegant here would be nice
      Operation *op = nullptr;
      size_t opnum = 0;
      for (auto &use : old_wire.getUses()) {
        if (!subcircuit->getOps().contains(use.getOwner()))
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
        if (!stepper->isStopped())
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

    builder.setInsertionPointAfter(cnot);
    auto call = builder.create<func::CallOp>(cnot->getLoc(), types,
                                             fun.getSymNameAttr(), args);
    for (size_t i = 0; i < steppers.size(); i++)
      steppers[i]->getOldWire().replaceAllUsesWith(call.getResult(i));

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
