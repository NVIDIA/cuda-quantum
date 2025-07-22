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

public:
  void runOnOperation() override {
    auto module = getOperation();
    size_t i = 0;
    SetVector<Subcircuit *> subcircuits;

    for (auto &op : module) {
      if (auto func = dyn_cast<func::FuncOp>(op)) {
        auto nlc = NetlistContainer(func);
        func.walk([&](Operation *op) {
          if (!::isControlledOp(op) || ::processed(op))
            return;

          auto subcircuit = new Subcircuit(op, &nlc);
          // Add the subcircuit to erase from the function after we
          // finish walking it, as we don't want to erase ops from a
          // function we are currently walking
          auto name = std::string("subcircuit") + std::to_string(i++);
          if (subcircuit->getNumRotations() > 1 &&
              subcircuit->moveToFunc(module, name))
            subcircuits.insert(subcircuit);
          else
            delete subcircuit;
        });

        // Clean up
        for (auto subcircuit : subcircuits) {
          for (auto op : subcircuit->getOrderedOps())
            op->erase();
          delete subcircuit;
          subcircuits.clear();
        }
      }
    }
  }
};
} // namespace

static void createPhaseFoldingPipeline(OpPassManager &pm) {
  // pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(cudaq::opt::createPhasePolynomialPreprocess());
  pm.addNestedPass<func::FuncOp>(
      cudaq::opt::createPhasePolynomialRotationMerging());
  cudaq::opt::addAggressiveEarlyInlining(pm);
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  // pm.addNestedPass<func::FuncOp>(cudaq::opt::createFactorQuantumAllocations());
  // pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // pm.addNestedPass<func::FuncOp>(createCSEPass());
}

void cudaq::opt::registerPhaseFoldingPipeline() {
  PassPipelineRegistration<>(
      "phase-folding-pipeline",
      "Apply phase polynomial based rotation merging.",
      [](OpPassManager &pm) { createPhaseFoldingPipeline(pm); });
}
