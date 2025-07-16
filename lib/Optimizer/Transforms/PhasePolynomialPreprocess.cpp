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

  void moveToFunc(Subcircuit *subcircuit, size_t subcircuit_num) {
    auto module = getOperation();
    auto name = std::string("subcircuit") + std::to_string(subcircuit_num);
    auto fun = cudaq::opt::factory::createFunction(name, {}, {}, module);
    fun.setPrivate();
    auto entry = fun.addEntryBlock();
    OpBuilder builder(fun);
    fun.getOperation()->setAttr("subcircuit", builder.getUnitAttr());

    SmallVector<Value> args;
    auto add_arg = [&](Value v) {
      auto idx = args.size();
      args.push_back(v);
      fun.insertArgument(idx, v.getType(), {}, v.getDefiningOp()->getLoc());
      return fun.getArgument(idx);
    };

    auto ops = subcircuit->getOps();
    auto ordered = SmallVector<Operation *>(ops.begin(), ops.end());
    std::sort(ordered.begin(), ordered.end(), [&](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    builder.setInsertionPointToStart(entry);
    for (auto op : ordered) {
      auto clone = builder.clone(*op);
      for (size_t i = 0; i < clone->getOperands().size(); i++) {
        auto operand = clone->getOperand(i);
        if (!quake::isQuantumType(operand.getType())) {
          auto arg = add_arg(operand);
          clone->setOperand(i, arg);
        }
      }
    }

    for (auto ref : subcircuit->getRefs()) {
      auto arg = add_arg(ref);
      ref.replaceUsesWithIf(arg, [&](OpOperand &use) {
        return use.getOwner()->getBlock() == entry;
      });
    }

    builder.create<cudaq::cc::ReturnOp>(fun.getLoc());

    auto cnot = subcircuit->getStart();
    auto latest = cnot;
    // for (auto arg : args) {
    //   if (!isa<quake::WireType>(arg.getType()))
    //     continue;
    //   auto dop = arg.getDefiningOp();
    //   if (dop && latest->isBeforeInBlock(dop))
    //     latest = dop;
    // }
    builder.setInsertionPointAfter(latest);
    builder.create<func::CallOp>(cnot->getLoc(), fun, args);
  }

public:
  void runOnOperation() override {
    auto module = getOperation();
    size_t i = 0;
    SetVector<Subcircuit *> subcircuits;

    auto nlc = new NetlistContainer();

    for (auto &op : module) {
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        func.walk([&](Operation *op) {
          if (auto refop = dyn_cast<quake::ExtractRefOp>(op)) {
            nlc->allocNetlist(refop);
            return;
          }

          if (!::isControlledOp(op) || ::processed(op))
            return;

          auto subcircuit = new Subcircuit(op, nlc);
          // Add the subcircuit to erase from the function after we
          // finish walking it, as we don't want to erase ops from a
          // function we are currently walking
          subcircuits.insert(subcircuit);
          moveToFunc(subcircuit, i++);
        });
      }
    }

    for (auto subcircuit : subcircuits) {
      for (auto op : subcircuit->getOps())
        op->erase();
      delete subcircuit;
    }

    delete nlc;
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
